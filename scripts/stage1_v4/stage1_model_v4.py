# stage1_model_v4.py
import torch
import torch.nn as nn
import torch.nn.functional as F
import timm

# ----------------------
# Small helpers
# ----------------------
def conv1x1(in_ch, out_ch):
    return nn.Conv2d(in_ch, out_ch, kernel_size=1, bias=True)

def conv3x3(in_ch, out_ch, stride=1):
    return nn.Conv2d(in_ch, out_ch, kernel_size=3, stride=stride, padding=1, bias=True)

def soft_argmax_2d(logits: torch.Tensor, tau: float = 1.0):
    if torch.isnan(logits).any():
        print("NaN detected in Stage-1 logits")
    # logits32 = logits.to(torch.float32)
    B, _, H, W = logits.shape
    flat = logits.view(B, H*W) / max(tau, 1e-6)
    prob = F.softmax(flat, dim=1).view(B, 1, H, W)

    if torch.isnan(prob).any():
        print("NaN detected in Stage-1 prob")

    ys = torch.linspace(0.5/H, 1-0.5/H, H, device=logits.device)
    xs = torch.linspace(0.5/W, 1-0.5/W, W, device=logits.device)
    yy, xx = torch.meshgrid(ys, xs, indexing="ij")

    x = (prob * xx).sum(dim=(2,3))
    y = (prob * yy).sum(dim=(2,3))
    xy = torch.cat([x,y], dim=1)  # [B,2]
    return xy, prob

# ----------------------
# Auxiliary heatmap
# ----------------------
@torch.no_grad()
def build_aux_heatmap(last_pos, size=75, sigma=0.05):
    """
    last_pos: [B,3] (x,y,vis) in [0,1]
    Returns: [B,1,H,W]
    """
    B = last_pos.size(0)
    H = W = size

    xs = torch.linspace(0.5/W, 1-0.5/W, W, device=last_pos.device, dtype=last_pos.dtype)
    ys = torch.linspace(0.5/H, 1-0.5/H, H, device=last_pos.device, dtype=last_pos.dtype)
    yy, xx = torch.meshgrid(ys, xs, indexing="ij")   # [H,W]
    grid = torch.stack([xx, yy], dim=-1)             # [H,W,2]

    # [B,1,1,2] last positions
    centers = last_pos[:, :2].unsqueeze(1).unsqueeze(1)  # [B,1,1,2]
    vis_mask = (last_pos[:, 2] > 0.5).float().view(B,1,1) # [B,1,1]

    # distances squared: [B,H,W]
    d2 = ((grid.unsqueeze(0) - centers)**2).sum(dim=-1)   # [B,H,W]
    bump = torch.exp(-d2 / (2*sigma**2)) * vis_mask       # zero if invisible

    return bump.unsqueeze(1)  # [B,1,H,W]

# ----------------------
# BiFPN Block
# ----------------------
class BiFPNFirst(nn.Module):
    def __init__(self, out_ch=64):
        super().__init__()
        self.l2 = conv1x1(40,  out_ch)
        self.l3 = conv1x1(56,  out_ch)
        self.l4 = conv1x1(136, out_ch)
        self.l5 = conv1x1(232, out_ch)
        self.out2 = conv3x3(out_ch, out_ch)
        self.out3 = conv3x3(out_ch, out_ch)
        self.out4 = conv3x3(out_ch, out_ch)
        self.out5 = conv3x3(out_ch, out_ch)

    def forward(self, c2, c3, c4, c5):
        # project to 64
        c2 = self.l2(c2); c3 = self.l3(c3); c4 = self.l4(c4); c5 = self.l5(c5)
        # top–down
        p5 = c5
        p4 = c4 + F.interpolate(p5, size=c4.shape[-2:], mode="bilinear", align_corners=False)
        p3 = c3 + F.interpolate(p4, size=c3.shape[-2:], mode="bilinear", align_corners=False)
        p2 = c2 + F.interpolate(p3, size=c2.shape[-2:], mode="bilinear", align_corners=False)
        # bottom–up (stride-2 3×3 “downsample”)
        p3 = p3 + F.max_pool2d(p2, kernel_size=3, stride=2, padding=1)  # 75→38
        p4 = p4 + F.max_pool2d(p3, kernel_size=3, stride=2, padding=1)  # 38→19
        p5 = p5 + F.max_pool2d(p4, kernel_size=3, stride=2, padding=1)  # 19→10
        return self.out2(p2), self.out3(p3), self.out4(p4), self.out5(p5)


class BiFPNIter(nn.Module):
    def __init__(self, ch=64):
        super().__init__()
        # all inputs already ch=64
        self.out2 = conv3x3(ch, ch)
        self.out3 = conv3x3(ch, ch)
        self.out4 = conv3x3(ch, ch)
        self.out5 = conv3x3(ch, ch)

    def forward(self, p2, p3, p4, p5):
        # top–down
        t5 = p5
        t4 = p4 + F.interpolate(t5, size=p4.shape[-2:], mode="bilinear", align_corners=False)
        t3 = p3 + F.interpolate(t4, size=p3.shape[-2:], mode="bilinear", align_corners=False)
        t2 = p2 + F.interpolate(t3, size=p2.shape[-2:], mode="bilinear", align_corners=False)
        # bottom–up
        b3 = t3 + F.max_pool2d(t2, kernel_size=3, stride=2, padding=1)
        b4 = t4 + F.max_pool2d(b3, kernel_size=3, stride=2, padding=1)
        b5 = t5 + F.max_pool2d(b4, kernel_size=3, stride=2, padding=1)
        return self.out2(t2), self.out3(b3), self.out4(b4), self.out5(b5)

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD  = [0.229, 0.224, 0.225]

# ----------------------
# Stage1 Model (BiFPN+Heatmap)
# ----------------------
class Stage1ModelBiFPN(nn.Module):
    def __init__(self, tau=1.0):
        super().__init__()
        self.register_buffer("mean", torch.tensor(IMAGENET_MEAN).view(1, 3, 1, 1))
        self.register_buffer("std", torch.tensor(IMAGENET_STD).view(1, 3, 1, 1))
        self.backbone = timm.create_model("tf_efficientnetv2_b3", pretrained=True, features_only=True)
        self.bifpn_first = BiFPNFirst(64)
        self.bifpn_iters = nn.ModuleList([BiFPNIter(64) for _ in range(3)])  # 1st + 3 iters = 4 total
        self.tau = tau
        self.aux_gate = nn.Parameter(torch.tensor(0.0))

        # head
        self.head = nn.Sequential(
            conv1x1(512,256), nn.ReLU(inplace=True),
            conv3x3(256,256), nn.ReLU(inplace=True),
            conv1x1(256,64), nn.ReLU(inplace=True),
            conv3x3(64,64), nn.ReLU(inplace=True),
            conv1x1(64,16), nn.ReLU(inplace=True),
            nn.Conv2d(16,1,kernel_size=1)
        )

    def forward(self, current, past, positions):
        """
        frames: [B,4,3,300,300]
        positions: [B,30,3]
        """
        frames = torch.cat([current.unsqueeze(1), past], dim=1)
        B, T, C, H, W = frames.shape  # [B,4,3,300,300]
        frames = frames.view(B * T, C, H, W).contiguous()  # [B*4,3,300,300]

        # Normalize using ImageNet stats
        frames = (frames - self.mean) / self.std

        # FP16
        with torch.autocast(frames.device.type, dtype=torch.float16):
            feats = self.backbone(frames)  # list of 5
            # feats[1]: C2 = [B*4, 40, 75, 75]
            # feats[2]: C3 = [B*4, 56, 38, 38]
            # feats[3]: C4 = [B*4,136, 19, 19]
            # feats[4]: C5 = [B*4,232, 10, 10]

            c2, c3, c4, c5 = feats[1], feats[2], feats[3], feats[4]
            # C2..C5 from backbone
            p2, p3, p4, p5 = self.bifpn_first(c2, c3, c4, c5)
            for block in self.bifpn_iters:
                p2, p3, p4, p5 = block(p2, p3, p4, p5)

            # combine P2+P3
            p3_up = F.interpolate(p3, size=p2.shape[-2:], mode="bilinear", align_corners=False)  # [B*4,64,75,75]
            # temporal concat
            fmap = torch.cat([p2, p3_up], dim=1)  # [B*4,128,75,75]

            # fmap = fmap
            fmap = fmap.contiguous().view(B, T * 128, 75, 75)  # [B,512,75,75]
        # head
        H_logits = self.head(fmap)  # [B,1,75,75]
        # H_logits = H_logits

        # aux heatmap
        last_pos = positions[:,-1,:]  # [B,3]
        H_aux = build_aux_heatmap(last_pos, size=75, sigma=0.05)
        H = H_logits + F.softplus(self.aux_gate)*H_aux

        xy,prob = soft_argmax_2d(H, tau=self.tau)
        return {"xy":xy, "heatmap":H}
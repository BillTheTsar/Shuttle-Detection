# stage2_model_v6.py
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import timm

# -------------------------
# small helpers
# -------------------------
def conv1x1(in_ch, out_ch):
    return nn.Conv2d(in_ch, out_ch, kernel_size=1, bias=True)

def conv3x3(in_ch, out_ch, stride=1):
    return nn.Conv2d(in_ch, out_ch, kernel_size=3, stride=stride, padding=1, bias=True)

def soft_argmax_2d(logits: torch.Tensor, tau: float = 1.0):
    B, _, H, W = logits.shape
    flat = logits.view(B, H * W) / max(tau, 1e-6)
    prob = F.softmax(flat, dim=1).view(B, 1, H, W)
    ys = torch.linspace(0.5/H, 1 - 0.5/H, H, device=logits.device, dtype=logits.dtype)
    xs = torch.linspace(0.5/W, 1 - 0.5/W, W, device=logits.device, dtype=logits.dtype)
    yy, xx = torch.meshgrid(ys, xs, indexing="ij")
    x = (prob * xx).sum(dim=(2, 3))
    y = (prob * yy).sum(dim=(2, 3))
    xy = torch.cat([x, y], dim=1)
    return xy, prob

# -------------------------
# BiFPN blocks (B0 taps, 224x224 input)
# C2=32@56, C3=48@28, C4=112@14, C5=192@7
# -------------------------
class BiFPNFirst(nn.Module):
    def __init__(self, out_ch=64):
        super().__init__()
        self.l2 = conv1x1(32,  out_ch)
        self.l3 = conv1x1(48,  out_ch)
        self.l4 = conv1x1(112, out_ch)
        self.l5 = conv1x1(192, out_ch)

        self.out2 = conv3x3(out_ch, out_ch)
        self.out3 = conv3x3(out_ch, out_ch)
        self.out4 = conv3x3(out_ch, out_ch)
        self.out5 = conv3x3(out_ch, out_ch)

    def forward(self, c2, c3, c4, c5):
        # lateral 1×1 to unify channels
        c2 = self.l2(c2); c3 = self.l3(c3); c4 = self.l4(c4); c5 = self.l5(c5)

        # top-down: 7→14→28→56
        p5 = c5
        p4 = c4 + F.interpolate(p5, size=c4.shape[-2:], mode="bilinear", align_corners=False)   # 7→14
        p3 = c3 + F.interpolate(p4, size=c3.shape[-2:], mode="bilinear", align_corners=False)   # 14→28
        p2 = c2 + F.interpolate(p3, size=c2.shape[-2:], mode="bilinear", align_corners=False)   # 28→56

        # bottom-up: 56→28→14→7 (stride-2 pooling)
        p3 = p3 + F.max_pool2d(p2, kernel_size=3, stride=2, padding=1)  # 56→28
        p4 = p4 + F.max_pool2d(p3, kernel_size=3, stride=2, padding=1)  # 28→14
        p5 = p5 + F.max_pool2d(p4, kernel_size=3, stride=2, padding=1)  # 14→7

        return self.out2(p2), self.out3(p3), self.out4(p4), self.out5(p5)

class BiFPNIter(nn.Module):
    def __init__(self, ch=64):
        super().__init__()
        self.out2 = conv3x3(ch, ch)
        self.out3 = conv3x3(ch, ch)
        self.out4 = conv3x3(ch, ch)
        self.out5 = conv3x3(ch, ch)

    def forward(self, p2, p3, p4, p5):
        # top-down: 7→14→28→56
        t5 = p5
        t4 = p4 + F.interpolate(t5, size=p4.shape[-2:], mode="bilinear", align_corners=False)
        t3 = p3 + F.interpolate(t4, size=p3.shape[-2:], mode="bilinear", align_corners=False)
        t2 = p2 + F.interpolate(t3, size=p2.shape[-2:], mode="bilinear", align_corners=False)

        # bottom-up: 56→28→14→7
        b3 = t3 + F.max_pool2d(t2, kernel_size=3, stride=2, padding=1)
        b4 = t4 + F.max_pool2d(b3, kernel_size=3, stride=2, padding=1)
        b5 = t5 + F.max_pool2d(b4, kernel_size=3, stride=2, padding=1)

        return self.out2(t2), self.out3(b3), self.out4(b4), self.out5(b5)

# -------------------------
# Stage-2 v6 model (BiFPN + heatmap @ 14x14)
# -------------------------
class Stage2ModelBiFPN(nn.Module):
    """
    Inputs
      current_crop : [B, 3, 224, 224]   (0..1 floats)
      past_crops   : [B, 3, 3, 224, 224]
      positions    : [B, 30, 3]         (not used here, reserved)
    Outputs (dict)
      xy       : [B,2]        (soft-argmax on 14×14)
      heatmap  : [B,1,14,14]  (logits)
    """
    def __init__(self, tau: float = 1.0):
        super().__init__()
        self.backbone = timm.create_model("tf_efficientnetv2_b0", pretrained=True, features_only=True)

        # 4-iter BiFPN (first + 3 iters)
        self.bifpn_first = BiFPNFirst(64)
        self.bifpn_iters = nn.ModuleList([BiFPNIter(64) for _ in range(3)])

        # head (concat P4@14 and upsampled P5→14 across 4 frames → 512ch)
        self.head = nn.Sequential(
            conv1x1(512, 256), nn.ReLU(inplace=True),
            conv3x3(256, 256), nn.ReLU(inplace=True),
            conv1x1(256, 64),  nn.ReLU(inplace=True),
            conv3x3(64, 64),   nn.ReLU(inplace=True),
            conv1x1(64, 16),   nn.ReLU(inplace=True),
            nn.Conv2d(16, 1, kernel_size=1)  # logits
        )

        self.tau = tau

        # ImageNet normalization (you feed 0..1 crops)
        mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
        std  = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
        self.register_buffer("mean", mean, persistent=False)
        self.register_buffer("std",  std,  persistent=False)

    def _normalize(self, x: torch.Tensor) -> torch.Tensor:
        return (x - self.mean.to(x.dtype)) / self.std.to(x.dtype)

    def forward(self, current_crop, past_crops, positions, tau: float = None):
        # Build [B,4,3,224,224] → [B*4,3,224,224]
        frames = torch.cat([current_crop.unsqueeze(1), past_crops], dim=1)
        B, T, C, H, W = frames.shape
        frames = frames.view(B * T, C, H, W)
        frames = self._normalize(frames)

        # Backbone taps (224×224 input)
        feats = self.backbone(frames)
        # c2=[B*4,32,56,56], c3=[B*4,48,28,28], c4=[B*4,112,14,14], c5=[B*4,192,7,7]
        c2, c3, c4, c5 = feats[1], feats[2], feats[3], feats[4]

        # BiFPN (first + 3 iterations)
        p2, p3, p4, p5 = self.bifpn_first(c2, c3, c4, c5)
        for block in self.bifpn_iters:
            p2, p3, p4, p5 = block(p2, p3, p4, p5)

        # Fuse P4 (14×14) + upsampled P5→14×14
        p5_up = F.interpolate(p5, size=p4.shape[-2:], mode="bilinear", align_corners=False)
        per_frame = torch.cat([p4, p5_up], dim=1)              # [B*4,128,14,14]

        # Temporal concat across 4 frames
        fmap = per_frame.view(B, T * 128, 14, 14)              # [B,512,14,14]

        # Heatmap logits and soft-argmax
        H_logit = self.head(fmap).to(fmap.dtype)               # [B,1,14,14]
        xy, _ = soft_argmax_2d(H_logit, tau=self.tau if tau is None else tau)

        return {"xy": xy, "heatmap": H_logit}
import torch
import torch.nn as nn
import torch.nn.functional as F
import timm
from torchvision import models

# ----------------------
# Small helpers
# ----------------------
def conv1x1(in_ch, out_ch):
    return nn.Conv2d(in_ch, out_ch, kernel_size=1, bias=True)

def conv3x3(in_ch, out_ch):
    return nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1, bias=True)

def soft_argmax_2d(logits: torch.Tensor, tau: float = 1.0):
    """
    logits: [B, 1, H, W]  (heatmap *logits*)
    returns:
      xy: [B, 2] in [0,1]x[0,1]
      prob: [B, 1, H, W] (softmax over spatial)
    """
    # logits32 = logits.to(torch.float32)
    B, _, H, W = logits.shape
    flat = logits.view(B, H * W) / tau
    prob = F.softmax(flat, dim=1).view(B, 1, H, W)

    # grid centers in [0,1]
    ys = (torch.arange(H, device=logits.device) + 0.5) / H
    xs = (torch.arange(W, device=logits.device) + 0.5) / W
    yy, xx = torch.meshgrid(ys, xs, indexing="ij")   # [H, W]

    x = (prob * xx).sum(dim=(2, 3))  # [B,1]
    y = (prob * yy).sum(dim=(2, 3))  # [B,1]
    xy = torch.cat([x, y], dim=1)    # [B,2]
    return xy, prob

# ----------------------
# Auxiliary heatmap
# ----------------------
@torch.no_grad()
def build_aux_heatmap_s1(last_pos, size=75, sigma=0.05):
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

@torch.no_grad()
def build_aux_heatmap_s2(positions: torch.Tensor,
                            size: int = 14,
                            sigma_cells: float = 0.5) -> torch.Tensor:
    """
    positions: [B, 30, 3] with (x,y,vis) in [0,1]
    Returns:   [B, 1, size, size] auxiliary heatmaps
    Logic:
      - look at last 3 steps
      - if last k visible (k>=2): extrapolate to current step (linear for k=2, quadratic for k=3)
      - place a Gaussian bump at predicted location; else zeros
    """
    device = positions.device
    dtype = positions.dtype
    B = positions.size(0)

    # last 3 steps: [B, 3, 3], order = t-2, t-1, t
    last3 = positions[:, -3:, :]  # [..., 0]=x, 1=y, 2=vis
    x = last3[..., 0]  # [B, 3]
    y = last3[..., 1]
    v = last3[..., 2]

    # v_last3: [B, 3] booleans for [t-2, t-1, t]
    v_last3 = v > 0.5

    # Start from the end (t), count contiguous True values.
    k = torch.zeros(B, dtype=torch.int64, device=device)

    # t
    k = torch.where(v_last3[:, 2], torch.ones_like(k), k)

    # t-1 (only count if t was already counted)
    k = torch.where(v_last3[:, 1] & (k == 1), k + 1, k)

    # t-2 (only count if t and t-1 were already counted)
    k = torch.where(v_last3[:, 0] & (k == 2), k + 1, k)

    x_pred = torch.full((B,), -1.0, device=device, dtype=dtype)
    y_pred = torch.full((B,), -1.0, device=device, dtype=dtype)

    # linear (need last 2 visible: t-1 and t)
    mask_lin = (k >= 2)
    x3_lin = x[:, 2]; y3_lin = y[:, 2]
    x2_lin = x[:, 1]; y2_lin = y[:, 1]
    x_lin_pred = 2 * x3_lin - x2_lin
    y_lin_pred = 2 * y3_lin - y2_lin

    # Apply only to masked rows
    x_pred = torch.where(mask_lin, x_lin_pred, x_pred)
    y_pred = torch.where(mask_lin, y_lin_pred, y_pred)

    # quadratic (need last 3 visible)
    mask_quad = (k == 3)
    x1_q = x[:, 0]; x2_q = x[:, 1]; x3_q = x[:, 2]
    y1_q = y[:, 0]; y2_q = y[:, 1]; y3_q = y[:, 2]

    x_quad_pred = x1_q - 3 * x2_q + 3 * x3_q
    y_quad_pred = y1_q - 3 * y2_q + 3 * y3_q

    x_pred = torch.where(mask_quad, x_quad_pred, x_pred)
    y_pred = torch.where(mask_quad, y_quad_pred, y_pred)

    d23 = torch.sqrt((x[:, 1] - x[:, 2]) ** 2 + (y[:, 1] - y[:, 2]) ** 2)  # [B]
    d3p = torch.sqrt((x[:, 2] - x_pred) ** 2 + (y[:, 2] - y_pred) ** 2)  # [B]

    motion_ok = (d23 + d3p) > 0.02  # [B] boolean
    has_pred = (x_pred >= 0) & (y_pred >= 0)  # or another sentinel check
    in_bounds = (x_pred >= 0.0) & (x_pred <= 1.0) & (y_pred >= 0.0) & (y_pred <= 1.0)

    valid = has_pred & in_bounds & motion_ok  # [B]

    # Make grid of cell centers [1, size, size, 2]
    coords = (torch.arange(size, device=device, dtype=dtype) + 0.5) / size  # exact centers
    cy = coords
    cx = coords
    grid_y, grid_x = torch.meshgrid(cy, cx, indexing="ij")  # [size, size]
    centers = torch.stack([grid_x, grid_y], dim=-1).unsqueeze(0)  # [1, S, S, 2]

    # Gaussian sigma in absolute coords
    sigma = (1.0 / size) * sigma_cells
    inv_two_sigma2 = 1.0 / (2.0 * sigma * sigma)

    # Build per-sample heatmaps
    # Expand x_pred, y_pred to [B,1,1]
    xy_all = torch.stack([x_pred, y_pred], dim=-1).view(-1, 1, 1, 2)  # [B,1,1,2]
    # Compute full d2
    d2 = ((centers - xy_all) ** 2).sum(dim=-1)  # [B, S, S]
    # Compute full bump
    bump = torch.exp(-d2 * inv_two_sigma2)  # [B, S, S]
    # Mask out invalid predictions
    bump = torch.where(valid.view(-1, 1, 1), bump, torch.zeros_like(bump))
    # Assign to H_aux
    H_aux = bump.unsqueeze(1)  # [B, 1, S, S]
    return H_aux  # [B, 1, S, S]

# -----------------------
# Trajectory featurizer (as before)
# -----------------------
class TrajectoryFeaturizer(nn.Module):
    """
    features = [x1,y1,vis1, ..., x6,y6,vis6, dx1,dy1, ..., dx5,dy5]  -> 28 dims
    project -> 64 (default) for FiLM
    """
    def __init__(self, k_last=6, project_dim=64):
        super().__init__()
        assert k_last >= 2
        self.k_last = k_last
        in_dim = 3 * k_last + 2 * (k_last - 1)  # 28 for k_last=6
        self.proj = nn.Sequential(
            nn.Linear(in_dim, project_dim),
            nn.ReLU(inplace=True),
        )
        self.output_dim = project_dim

    @torch.no_grad()
    def _build_raw(self, pos):
        # pos: [B, 30, 3]
        B, T, C = pos.shape
        k = self.k_last
        last_k = pos[:, -k:, :]                    # [B,k,3]
        raw = last_k.reshape(B, -1)                # [B,3k]
        x, y, v = last_k[..., 0], last_k[..., 1], last_k[..., 2]
        v_pair = (v[:, :-1] > 0.5) & (v[:, 1:] > 0.5)
        dx = torch.where(v_pair, x[:, 1:] - x[:, :-1], torch.zeros_like(x[:, 1:]))
        dy = torch.where(v_pair, y[:, 1:] - y[:, :-1], torch.zeros_like(y[:, 1:]))
        deltas = torch.stack([dx, dy], dim=-1).reshape(B, -1)
        return torch.cat([raw, deltas], dim=1)     # [B,28]

    def forward(self, positions):
        feats = self._build_raw(positions).to(positions.dtype)
        return self.proj(feats)                    # [B,64]

# ----------------------
# BiFPN Block
# ----------------------
class BiFPNFirst_s1(nn.Module):
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

# -------------------------
# BiFPN blocks (B0 taps, 224x224 input)
# C2=32@56, C3=48@28, C4=112@14, C5=192@7
# -------------------------
class BiFPNFirst_s2(nn.Module):
    def __init__(self, out_ch=96):
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
        self.bifpn_first = BiFPNFirst_s1(64)
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
        # with torch.autocast(frames.device.type, dtype=torch.float16):
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

        fmap = fmap.contiguous().view(B, T * 128, 75, 75)  # [B,512,75,75]
        # fmap = fmap.float()
        # head. Note that self.head tends to blow up, thus we disable fp16 precision
        with torch.autocast(device_type=frames.device.type, enabled=False):
            H_logits = self.head(fmap)  # [B,1,75,75]
        # H_logits = H_logits.float()

        # aux heatmap
        last_pos = positions[:,-1,:]  # [B,3]
        H_aux = build_aux_heatmap_s1(last_pos, size=75, sigma=0.05)
        H = H_logits + F.softplus(self.aux_gate)*H_aux

        xy,prob = soft_argmax_2d(H, tau=self.tau)
        return {"xy":xy, "heatmap":H}


# Stage 2 Model
# -----------------------
# EfficientNet-B0 multi-scale extractor
# -----------------------
class EffB0MultiScale(nn.Module):
    """
    Returns:
      f7:  [B, 1280, 7, 7]
      f14: [B,  112, 14, 14]
    """
    def __init__(self):
        super().__init__()
        backbone = models.efficientnet_b0(weights="IMAGENET1K_V1")
        self.features = backbone.features  # Sequential of 9 blocks
        # we don't use backbone.classifier

    def forward(self, x):
        f14 = None
        for i, layer in enumerate(self.features):
            x = layer(x)
            if i == 5:   # after block 5 -> typically 14x14 on B0
                f14 = x
        f7 = x          # final features -> 7x7
        return f7, f14

# -----------------------
# Stage 2 Heatmap model
# -----------------------
class Stage2ModelHeatmap(nn.Module):
    """
    Inputs:
      current_crop : [B, 3, 224, 224]
      past_crops   : [B, 3, 3, 224, 224]
      positions    : [B, 30, 3]
    Outputs (dict):
      xy         : [B, 2]      (normalized to [0,1]^2 via soft-argmax over 14x14)
      vis_logit  : [B]         (from pooled logits)
      heatmap    : [B, 1, 14, 14] (logits)
    """
    def __init__(self, tau: float = 1.0):
        super().__init__()
        self.tau = tau
        self.aux_gate = nn.Parameter(torch.tensor(1.0))  # learnable weight for the prior heatmap
        # self.register_buffer("norm_mean", torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
        # self.register_buffer("norm_std", torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))

        # backbone
        self.backbone = EffB0MultiScale()

        # --- Per-crop heads (F7 & F14) ---
        # F7 per crop: 1280->640 (1x1) -> 640 (3x3) -> 320 (1x1)
        self.f7_head = nn.Sequential(
            nn.Sequential(conv1x1(1280, 640),
            nn.ReLU(inplace=True)),
            nn.Sequential(conv3x3(640, 640),
            nn.ReLU(inplace=True)),
            nn.Sequential(conv1x1(640, 320),
            nn.ReLU(inplace=True)),
        )
        # F14 per crop: 112->128 (1x1) -> 128 (3x3) -> 128 (3x3)
        self.f14_head = nn.Sequential(
            nn.Sequential(conv1x1(112, 128),
            nn.ReLU(inplace=True)),
            nn.Sequential(conv3x3(128, 128),
            nn.ReLU(inplace=True)),
            nn.Sequential(conv3x3(128, 128),
            nn.ReLU(inplace=True)),
        )

        # After concat across 4 crops:
        # F7: (320*4=1280, 7,7) → upsample → 1x1(640) → 3x3(640) → 1x1(256)  <-- your explicit step
        self.f7_post = nn.Sequential(
            nn.Sequential(conv1x1(1280, 640),
            nn.ReLU(inplace=True)),
            nn.Sequential(conv3x3(640, 640),
            nn.ReLU(inplace=True)),
            nn.Sequential(conv1x1(640, 256),
            nn.ReLU(inplace=True)),
        )
        # F14: (128*4=512, 14,14) → 1x1(256) → 3x3(256)
        self.f14_post = nn.Sequential(
            nn.Sequential(conv1x1(512, 256),
            nn.ReLU(inplace=True)),
            nn.Sequential(conv3x3(256, 256),
            nn.ReLU(inplace=True)),
        )

        # Fusion: concat (256+256) -> 512 → 3x3(256)
        self.fuse = nn.Sequential(
            conv3x3(512, 256),
            nn.ReLU(inplace=True),
        )

        # FiLM conditioning from trajectory (64 -> 256 gammas/betas)
        self.traj = TrajectoryFeaturizer(project_dim=64)
        self.gamma_fc = nn.Linear(64, 256)
        self.beta_fc  = nn.Linear(64, 256)
        # init FiLM to identity
        nn.init.zeros_(self.gamma_fc.weight); nn.init.ones_(self.gamma_fc.bias)
        nn.init.zeros_(self.beta_fc.weight);  nn.init.zeros_(self.beta_fc.bias)

        # Decoder to heatmap H (logits) at 14x14
        self.dec = nn.Sequential(
            nn.Sequential(conv1x1(256, 64),
            nn.ReLU(inplace=True)),
            nn.Sequential(conv3x3(64, 64),
            nn.ReLU(inplace=True)),
            nn.Sequential(conv1x1(64, 16),
            nn.ReLU(inplace=True)),
            nn.Sequential(conv3x3(16, 16),
            nn.ReLU(inplace=True)),
            nn.Conv2d(16, 1, kernel_size=1, bias=True),  # logits
        )

        # Visibility from pooled logits (s_max, s_avg) -> 1
        self.vis_head = nn.Linear(2, 1)
        # p = 0.65
        # with torch.no_grad():
        #     self.vis_head.weight.zero_()
        #     self.vis_head.bias.fill_(torch.logit(torch.tensor(p), eps=1e-6))

    def _extract_multi(self, x4: torch.Tensor):
        """
        x4: [B, 4, 3, 224, 224]
        returns per-crop heads concatenated across time:
          f7_cat  : [B, 1280, 7, 7]
          f14_cat : [B,  512, 14, 14]
        """
        B, T, C, H, W = x4.shape
        x4 = x4.view(B * T, C, H, W).contiguous()
        f7, f14 = self.backbone(x4)                  # [B*T,1280,7,7], [B*T,112,14,14]

        f7p  = self.f7_head(f7)                          # [B*T,320,7,7]
        f14p = self.f14_head(f14)                        # [B*T,128,14,14]

        f7p  = f7p.view(B, T, 320, 7, 7).transpose(1, 2).reshape(B, 320*T, 7, 7)     # -> [B,1280,7,7]
        f14p = f14p.view(B, T, 128,14,14).transpose(1, 2).reshape(B, 128*T,14,14)    # -> [B,512,14,14]
        return f7p, f14p

    def _film(self, F, traj_embed):
        """
        F:   [B, 256, 14, 14]
        traj_embed: [B, 64]
        """
        gamma = self.gamma_fc(traj_embed)  # [B,256]
        beta  = self.beta_fc(traj_embed)   # [B,256]
        gamma = gamma.view(-1, 256, 1, 1)
        beta  = beta.view(-1, 256, 1, 1)
        return gamma * F + beta

    # ---- forward ----
    def forward(self, current_crop, past_crops, positions, tau: float = None):
        # Stack 4 crops along "time"
        # if past_crops.dim() == 5:
        x4 = torch.cat([current_crop.unsqueeze(1), past_crops], dim=1)  # [B,4,3,H,W]
        f7_cat, f14_cat = self._extract_multi(x4)        # [B,1280,7,7], [B,512,14,14]

        # F7 path → upsample to 14
        f7_up = F.interpolate(f7_cat, size=(14, 14), mode="bilinear", align_corners=False)
        f7_up = self.f7_post(f7_up)                      # [B,256,14,14]

        # F14 path
        f14p = self.f14_post(f14_cat)                    # [B,256,14,14]

        # Fuse
        F_fused = self.fuse(torch.cat([f7_up, f14p], dim=1))  # [B,256,14,14]

        # FiLM with trajectory
        traj_embed = self.traj(positions)                # [B,64]
        F_mod = self._film(F_fused, traj_embed)          # [B,256,14,14]

        # Heatmap logits H
        H = self.dec(F_mod)                              # [B,1,14,14]
        H_aux = build_aux_heatmap_s2(positions, size=14, sigma_cells=0.5)  # same device/dtype handled
        H = H + F.softplus(self.aux_gate) * H_aux.to(H.dtype)  # self.aux_alpha = nn.Parameter(torch.tensor(0.3))

        # Soft-argmax to (x,y) and softmax prob
        xy, P = soft_argmax_2d(H, tau=self.tau if tau is None else tau)  # xy∈[0,1]^2

        # Visibility from pooled *logits*
        s_max = H.amax(dim=(2, 3)).squeeze(1)           # [B]
        s_avg = H.mean(dim=(2, 3)).squeeze(1)           # [B]
        vis_logit = self.vis_head(torch.stack([s_max, s_avg], dim=1)).squeeze(1)
        # vis_logit = self._is_near_corner(xy)

        return {"xy": xy, "vis_logit": vis_logit, "heatmap": H}


# -------------------------
# Stage-2 v7 model (BiFPN + heatmap @ 14x14)
# -------------------------
class Stage2ModelBiFPN(nn.Module):
    """
    Inputs
      current_crop : [B, 3, 224, 224]   (0..1 floats)
      past_crops   : [B, 3, 3, 224, 224]
      positions    : [B, 30, 3]         (not used here, reserved)
    Outputs (dict)
      xy       : [B,2]        (soft-argmax on 14×14)
      heatmap  : [B,1,28,28]  (logits)
    """
    def __init__(self, tau: float = 0.7):
        super().__init__()
        self.backbone = timm.create_model("tf_efficientnetv2_b0", pretrained=True, features_only=True)

        # 4-iter BiFPN (first + 3 iters)
        self.bifpn_first = BiFPNFirst_s2(96)
        self.bifpn_iters = nn.ModuleList([BiFPNIter(96) for _ in range(3)])

        # head (concat P3@28 and upsampled P4→28 across 4 frames → 768ch)
        self.head = nn.Sequential(
            conv1x1(768, 256), nn.ReLU(inplace=True),
            conv3x3(256, 256), nn.ReLU(inplace=True),
            conv1x1(256, 64),  nn.ReLU(inplace=True),
            conv3x3(64, 64),   nn.ReLU(inplace=True),
            conv1x1(64, 16),   nn.ReLU(inplace=True),
            nn.Conv2d(16, 1, kernel_size=1)  # logits
        )

        self.tau = tau

        # ImageNet normalization (you feed 0..1 crops)
        self.register_buffer("mean", torch.tensor(IMAGENET_MEAN).view(1, 3, 1, 1), persistent=False)
        self.register_buffer("std", torch.tensor(IMAGENET_STD).view(1, 3, 1, 1), persistent=False)
        # self.register_buffer("mean", mean, persistent=False)
        # self.register_buffer("std",  std,  persistent=False)

    def _normalize(self, x: torch.Tensor) -> torch.Tensor:
        return (x - self.mean.to(x.dtype)) / self.std.to(x.dtype)

    def forward(self, current_crop, past_crops, positions, tau: float = None):
        # Build [B,4,3,224,224] → [B*4,3,224,224]
        frames = torch.cat([current_crop.unsqueeze(1), past_crops], dim=1)
        B, T, C, H, W = frames.shape
        frames = frames.view(B * T, C, H, W).contiguous()
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
        p4_up = F.interpolate(p4, size=p3.shape[-2:], mode="bilinear", align_corners=False)
        per_frame = torch.cat([p3, p4_up], dim=1)              # [B*4,192,28,28]

        # Temporal concat across 4 frames
        fmap = per_frame.contiguous().view(B, T * 192, 28, 28)

        # Heatmap logits and soft-argmax
        H_logit = self.head(fmap).to(fmap.dtype)               # [B,1,28,28]
        H_aux = build_aux_heatmap_s2(positions, size=28, sigma_cells=0.5)  # same device/dtype handled
        H_logit = H_logit + 2 * H_aux.to(H_logit.dtype)
        xy, _ = soft_argmax_2d(H_logit, tau=self.tau if tau is None else tau)

        return {"xy": xy, "heatmap": H_logit}
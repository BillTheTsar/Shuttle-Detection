# stage2_model_v7.py
# Slight changes, channels from 64 -> 96, tapping P3 and P4 instead of P4 and P5, tau = 0.7
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

def soft_argmax_2d(logits: torch.Tensor, tau: float = 0.7):
    logits32 = logits.to(torch.float32)
    B, _, H, W = logits32.shape
    flat = logits32.view(B, H * W) / max(tau, 1e-6)
    prob = F.softmax(flat, dim=1).view(B, 1, H, W)
    ys = torch.linspace(0.5/H, 1 - 0.5/H, H, device=logits.device, dtype=torch.float32)
    xs = torch.linspace(0.5/W, 1 - 0.5/W, W, device=logits.device, dtype=torch.float32)
    yy, xx = torch.meshgrid(ys, xs, indexing="ij")
    x = (prob * xx).sum(dim=(2, 3))
    y = (prob * yy).sum(dim=(2, 3))
    xy = torch.cat([x, y], dim=1)
    return xy, prob

@torch.no_grad()
def build_aux_heatmap_batch(positions: torch.Tensor,
                            size: int = 28,
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

# -------------------------
# BiFPN blocks (B0 taps, 224x224 input)
# C2=32@56, C3=48@28, C4=112@14, C5=192@7
# -------------------------
class BiFPNFirst(nn.Module):
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
    def __init__(self, ch=96):
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
        self.bifpn_first = BiFPNFirst(96)
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
        mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
        std  = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
        self.register_buffer("mean", mean, persistent=False)
        self.register_buffer("std",  std,  persistent=False)

    def _normalize(self, x: torch.Tensor) -> torch.Tensor:
        return (x - self.mean.to(x.dtype)) / self.std.to(x.dtype)

    def forward(self, current_crop, past_crops, positions, tau: float = 0.7):
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
        fmap = per_frame.contiguous().view(B, T * 192, 28, 28)              # [B,512,14,14]

        # Heatmap logits and soft-argmax
        H_logit = self.head(fmap).to(fmap.dtype)               # [B,1,14,14]
        H_aux = build_aux_heatmap_batch(positions, size=28, sigma_cells=0.5)  # same device/dtype handled
        H_logit = H_logit + 2 * H_aux.to(H_logit.dtype)
        xy, _ = soft_argmax_2d(H_logit, tau=self.tau if tau is None else tau)

        return {"xy": xy, "heatmap": H_logit}
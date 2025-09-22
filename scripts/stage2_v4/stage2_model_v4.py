# stage2_model_v4.py
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models

# -----------------------
# Small helpers
# -----------------------
def conv1x1(in_ch, out_ch):
    return nn.Sequential(
        nn.Conv2d(in_ch, out_ch, kernel_size=1, bias=True),
        nn.ReLU(inplace=True),
    )

def conv3x3(in_ch, out_ch):
    return nn.Sequential(
        nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1, bias=True),
        nn.ReLU(inplace=True),
    )

def soft_argmax_2d(logits: torch.Tensor, tau: float = 1.0):
    """
    logits: [B, 1, H, W]  (heatmap *logits*)
    returns:
      xy: [B, 2] in [0,1]x[0,1]
      prob: [B, 1, H, W] (softmax over spatial)
    """
    logits32 = logits.to(torch.float32)
    B, _, H, W = logits32.shape
    flat = logits32.view(B, H * W) / max(tau, 1e-6)
    prob = F.softmax(flat, dim=1).view(B, 1, H, W)

    # grid centers in [0,1]
    ys = (torch.arange(H, device=logits.device, dtype=torch.float32) + 0.5) / H
    xs = (torch.arange(W, device=logits.device, dtype=torch.float32) + 0.5) / W
    yy, xx = torch.meshgrid(ys, xs, indexing="ij")   # [H, W]

    x = (prob * xx).sum(dim=(2, 3))  # [B,1]
    y = (prob * yy).sum(dim=(2, 3))  # [B,1]
    xy = torch.cat([x, y], dim=1)    # [B,2]
    return xy, prob

@torch.no_grad()
def build_aux_heatmap_batch(positions: torch.Tensor,
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
            conv1x1(1280, 640),
            conv3x3(640, 640),
            conv1x1(640, 320),
        )
        # F14 per crop: 112->128 (1x1) -> 128 (3x3) -> 128 (3x3)
        self.f14_head = nn.Sequential(
            conv1x1(112, 128),
            conv3x3(128, 128),
            conv3x3(128, 128),
        )

        # After concat across 4 crops:
        # F7: (320*4=1280, 7,7) → upsample → 1x1(640) → 3x3(640) → 1x1(256)  <-- your explicit step
        self.f7_post = nn.Sequential(
            conv1x1(1280, 640),
            conv3x3(640, 640),
            conv1x1(640, 256),
        )
        # F14: (128*4=512, 14,14) → 1x1(256) → 3x3(256)
        self.f14_post = nn.Sequential(
            conv1x1(512, 256),
            conv3x3(256, 256),
        )

        # Fusion: concat (256+256) -> 512 → 3x3(256)
        self.fuse = conv3x3(512, 256)

        # FiLM conditioning from trajectory (64 -> 256 gammas/betas)
        self.traj = TrajectoryFeaturizer(project_dim=64)
        self.gamma_fc = nn.Linear(64, 256)
        self.beta_fc  = nn.Linear(64, 256)
        # init FiLM to identity
        nn.init.zeros_(self.gamma_fc.weight); nn.init.ones_(self.gamma_fc.bias)
        nn.init.zeros_(self.beta_fc.weight);  nn.init.zeros_(self.beta_fc.bias)

        # Decoder to heatmap H (logits) at 14x14
        self.dec = nn.Sequential(
            conv1x1(256, 64),
            conv3x3(64, 64),
            conv1x1(64, 16),
            conv3x3(16, 16),
            nn.Conv2d(16, 1, kernel_size=1, bias=True),  # logits
        )

        # Visibility from pooled logits (s_max, s_avg) -> 1
        self.vis_head = nn.Linear(2, 1)
        p = 0.65
        with torch.no_grad():
            self.vis_head.weight.zero_()
            self.vis_head.bias.fill_(torch.logit(torch.tensor(p), eps=1e-6))

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
        H_aux = build_aux_heatmap_batch(positions, size=14, sigma_cells=0.5)  # same device/dtype handled
        H = H + F.softplus(self.aux_gate) * H_aux.to(H.dtype)  # self.aux_alpha = nn.Parameter(torch.tensor(0.3))

        # Soft-argmax to (x,y) and softmax prob
        xy, P = soft_argmax_2d(H, tau=self.tau if tau is None else tau)  # xy∈[0,1]^2

        # Visibility from pooled *logits*
        s_max = H.amax(dim=(2, 3)).squeeze(1)           # [B]
        s_avg = H.mean(dim=(2, 3)).squeeze(1)           # [B]
        vis_logit = self.vis_head(torch.stack([s_max, s_avg], dim=1)).squeeze(1)
        # vis_logit = self._is_near_corner(xy)

        return {"xy": xy, "vis_logit": vis_logit, "heatmap": H}

# stage2_model.py
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models


# ---------- Attention Pooling (same mechanism, different size) ----------
class AttentionPooling(nn.Module):
    """
    Computes weighted average of past frame features using learned attention scores.
    Input:  features [B, N, D]
    Output: [B, D]
    """
    def __init__(self, feature_dim):
        super().__init__()
        self.attention = nn.Sequential(
            nn.Linear(feature_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 1)  # Score for each frame
        )

    def forward(self, features):
        weights = self.attention(features)  # [B, N, 1]
        weights = F.softmax(weights, dim=1)  # Normalize across frames
        weighted_sum = (features * weights).sum(dim=1)  # Weighted avg
        return weighted_sum

# ---------- Trajectory Encoder ----------
class TrajectoryFeaturizer(nn.Module):
    """
    Build a small, fixed-order vector from the last K steps:
      features = [x1,y1,vis1, ..., x6,y6,vis6, dx1,dy1, ..., dx5,dy5]  -> 28 dims
    If both steps t and t+1 are visible (vis>0.5), use dx,dy; else 0.
    Optionally project to a higher dim with a tiny MLP.
    """
    def __init__(self, k_last=6, project_dim=None):
        super().__init__()
        assert k_last >= 2, "k_last must be >= 2 to compute deltas"
        self.k_last = k_last
        in_dim = 3 * k_last + 2 * (k_last - 1)  # 18 + 10 = 28 for k_last=6
        if project_dim is None:
            self.proj = None
            self.output_dim = in_dim
        else:
            self.proj = nn.Sequential(
                nn.Linear(in_dim, project_dim),
                nn.ReLU()
            )
            self.output_dim = project_dim

    @torch.no_grad()  # feature construction is simple; gradients not needed here
    def _build_raw_features(self, pos):
        """
        pos: [B, 30, 3] with columns (x, y, vis) in [0,1]
        Returns: [B, 28] for k_last=6
        """
        B, T, C = pos.shape
        k = self.k_last

        last_k = pos[:, -k:, :]                          # [B, k, 3]
        raw = last_k.reshape(B, -1)                      # [B, 3k]  (x,y,vis) flattened in order

        # deltas between consecutive visible steps
        x = last_k[..., 0]                               # [B, k]
        y = last_k[..., 1]
        v = last_k[..., 2]
        v_pair = (v[:, :-1] > 0.5) & (v[:, 1:] > 0.5)    # [B, k-1] both visible
        dx = x[:, 1:] - x[:, :-1]
        dy = y[:, 1:] - y[:, :-1]
        dx = torch.where(v_pair, dx, torch.zeros_like(dx))
        dy = torch.where(v_pair, dy, torch.zeros_like(dy))
        deltas = torch.stack([dx, dy], dim=-1).reshape(B, -1)  # [B, 2*(k-1)]

        feats = torch.cat([raw, deltas], dim=1)          # [B, 3k + 2(k-1)]
        return feats

    def forward(self, positions):
        # positions: [B, 30, 3]
        feats = self._build_raw_features(positions).to(positions.dtype)
        if self.proj is not None:
            return self.proj(feats)
        return feats  # raw 28-dim vector


# ---------- Stage 2 Model ----------
class Stage2Model(nn.Module):
    """
    Inputs:
      current_crop : [B, 3, H, W]         (e.g., 224x224)
      past_crops   : [B, N, 3, H, W]
      positions    : [B, 30, 3]           (adjusted (x,y,vis) for the crop)

    Output:
      logits [B, 3]  -> (x_logit, y_logit, vis_logit)
    """
    def __init__(self):
        super().__init__()
        # EfficientNet-B0 backbone
        backbone = models.efficientnet_b0(weights="IMAGENET1K_V1")
        self.feature_extractor = nn.Sequential(*list(backbone.children())[:-1])  # -> [B, 1280, 1, 1]
        self.feature_dim = 1280

        # Past-frame attention pooling (MLP scorer)
        self.attn_pool = AttentionPooling(self.feature_dim)

        self.traj_encoder = TrajectoryFeaturizer(project_dim=64)

        # Fusion MLP (order: [past | current | trajectory])
        fusion_in = self.feature_dim * 2 + self.traj_encoder.output_dim
        self.mlp = nn.Sequential(
            nn.Linear(fusion_in, 1024),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(1024, 256),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(256, 3)
        )

    def extract_features(self, img: torch.Tensor) -> torch.Tensor:
        # img: [B, 3, H, W] -> [B, 1280]
        x = self.feature_extractor(img)   # [B, 1280, 1, 1]
        return x.flatten(1)               # [B, 1280]

    def forward(self, current_crop, past_crops, positions):
        # Current features
        f_cur = self.extract_features(current_crop)     # [B, 1280]

        # Past features + attention
        B, N, C, H, W = past_crops.shape
        past_flat = past_crops.view(B * N, C, H, W)
        f_past = self.extract_features(past_flat).view(B, N, -1)  # [B, N, 1280]
        f_past = self.attn_pool(f_past)                           # [B, 1280]

        # Trajectory features
        f_traj = self.traj_encoder(positions)                     # [B, 64]

        # Fusion
        fused = torch.cat([f_past, f_cur, f_traj], dim=1)         # [B, 1280+1280+64]
        logits = self.mlp(fused)
        return logits

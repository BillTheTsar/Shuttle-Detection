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
class TrajectoryEncoder(nn.Module):
    """
    Encodes 30 (x, y, visibility) vectors into a 64-d motion feature.
    Uses Conv1D + Global Average Pooling.
    """
    def __init__(self, input_channels=3, hidden_dim=64):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv1d(input_channels, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv1d(32, hidden_dim, kernel_size=3, padding=1),
            nn.ReLU()
        )
        self.global_pool = nn.AdaptiveAvgPool1d(1)  # Collapse sequence into 1 value per channel
        self.output_dim = hidden_dim

    def forward(self, x):
        x = x.permute(0, 2, 1)  # [B, 3, 30]
        x = self.encoder(x)     # [B, 64, 30]
        x = self.global_pool(x).squeeze(-1)  # [B, 64]
        return x


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

        # Trajectory encoder (wider than stage-1)
        self.traj_encoder = TrajectoryEncoder()

        # Fusion MLP (order: [past | current | trajectory])
        fusion_in = self.feature_dim * 2 + self.traj_encoder.output_dim
        self.mlp = nn.Sequential(
            nn.Linear(fusion_in, 1024),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, 128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, 3)  # Only x_pred and y_pred
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
        f_traj = self.traj_encoder(positions)                     # [B, 96]

        # Fusion
        fused = torch.cat([f_past, f_cur, f_traj], dim=1)         # [B, 1280+1280+96]
        logits = self.mlp(fused)
        xy_logit = logits[:, :2]  # [B, 2]
        vis_logit = logits[:, 2]  # [B]
        xy = torch.sigmoid(xy_logit)  # [B, 2] in [0,1]

        # Return concatenated: [x, y, vis_logit]
        return torch.cat([xy, vis_logit.unsqueeze(-1)], dim=1)

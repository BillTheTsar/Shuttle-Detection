import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models


# ---------- Attention Pooling ----------
class AttentionPooling(nn.Module):
    """
    Computes weighted average of past frame features using learned attention scores.
    Input: features [B, N, feature_dim]
    Output: [B, feature_dim]
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

# ---------- Stage 1 Model ----------
class Stage1Model(nn.Module):
    def __init__(self):
        super().__init__()
        # Load EfficientNet-B3 backbone (remove classifier)
        backbone = models.efficientnet_b3(weights="IMAGENET1K_V1")
        self.feature_extractor = nn.Sequential(*list(backbone.children())[:-1])
        self.feature_dim = 1536

        # Attention for past frames
        self.attention_pool = AttentionPooling(self.feature_dim)

        # Trajectory encoder
        self.traj_encoder = TrajectoryEncoder()

        # Fusion MLP
        fusion_input_dim = self.feature_dim * 2 + self.traj_encoder.output_dim
        self.mlp = nn.Sequential(
            nn.Linear(fusion_input_dim, 1024),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, 128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, 6)  # Three predictions of (x_pred, y_pred)
        )

    def extract_features(self, img):
        x = self.feature_extractor(img)  # [B, 1536, 1, 1]
        return x.flatten(1)  # [B, 1536]

    def forward(self, current_img, past_imgs, positions):
        # Current frame features
        f_current = self.extract_features(current_img)

        # Past frames
        B, N, C, H, W = past_imgs.shape
        past_imgs = past_imgs.view(B * N, C, H, W)
        f_past = self.extract_features(past_imgs)
        f_past = f_past.view(B, N, -1)
        f_past = self.attention_pool(f_past)

        # Trajectory features
        f_traj = self.traj_encoder(positions)

        # Fusion
        fused = torch.cat([f_past, f_current, f_traj], dim=1)
        return self.mlp(fused)
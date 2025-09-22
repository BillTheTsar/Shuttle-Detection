import json
import random
import torch
from pathlib import Path
from PIL import Image
import matplotlib.pyplot as plt
import pandas as pd
import torchvision.transforms as T
from stage1_model_v2 import Stage1Model
import numpy as np

# ------------------------
# Configurations
# ------------------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
DATA_ROOT = PROJECT_ROOT / "sequences"  # Adjust if necessary
INDEX_FILE = PROJECT_ROOT / "configs" / "dataset_index.json"
CHECKPOINT_PATH = PROJECT_ROOT / "models" / "stage1_v2_best.pth"

device = "cuda" if torch.cuda.is_available() else "cpu"
IMG_WIDTH, IMG_HEIGHT = 1920, 1080
# IMAGENET_MEAN = torch.tensor([0.485, 0.456, 0.406])[:,None,None]
# IMAGENET_STD  = torch.tensor([0.229, 0.224, 0.225])[:,None,None]

def normalize_img(t: torch.Tensor) -> torch.Tensor:
    # t: [3,H,W]
    return (t - IMAGENET_MEAN) / IMAGENET_STD

# Load transforms (match training)
transform = T.Compose([
    T.Resize((300, 300)),
    T.ToTensor()
])

# ------------------------
# Load model
# ------------------------
model = Stage1Model().to(device)
model.load_state_dict(torch.load(CHECKPOINT_PATH, map_location=device))
model.eval()

# ------------------------
# Load dataset index
# ------------------------
with open(INDEX_FILE, "r") as f:
    index_data = json.load(f)

test_samples = index_data.get("test", [])
if len(test_samples) == 0:
    raise ValueError("No samples found in 'test' split of index file.")

# Pick 4 random samples
samples = random.sample(test_samples, min(4, len(test_samples)))

# ------------------------
# Visualization Loop
# ------------------------
plt.figure(figsize=(16, 12))

for idx, sample_rel in enumerate(samples, 1):
    sample_path = DATA_ROOT / sample_rel.replace("\\", "/")

    # Load images
    current_img = Image.open(sample_path / "frame.jpg").convert("RGB")
    past_imgs = [Image.open(sample_path / f"frame-{i}.jpg").convert("RGB") for i in range(3, 0, -1)]

    # Transform images
    current_tensor = transform(current_img).unsqueeze(0).to(device)
    # current_tensor = normalize_img(current_tensor)
    past_tensor = torch.stack([transform(img) for img in past_imgs]).unsqueeze(0).to(device)
    # past_tensor = normalize_img(past_tensor)

    # Load positions
    positions = torch.tensor(
        pd.read_csv(sample_path / "positions.csv").values[::-1].copy(),
        dtype=torch.float32
    ).unsqueeze(0).to(device)

    # Load target
    target = pd.read_csv(sample_path / "target.csv").iloc[0]
    target_x, target_y, target_vis = target.shuttle_x, target.shuttle_y, target.shuttle_visibility

    # Inference
    with torch.no_grad():
        output = model(current_tensor, past_tensor, positions)  # [1, 2]
        pred_xy = output.cpu().numpy()[0]

    pred_x_norm, pred_y_norm = pred_xy
    pred_x_abs, pred_y_abs = pred_x_norm * IMG_WIDTH, pred_y_norm * IMG_HEIGHT
    target_x_abs, target_y_abs = target_x * IMG_WIDTH, target_y * IMG_HEIGHT

    # Compute triangle points (normalized)
    radius = 0.06  # normalized
    angles_deg = [0, 120, 240]
    triangle_points = []
    for angle in angles_deg:
        theta = np.radians(angle)
        dx = radius * np.cos(theta)
        dy = radius * np.sin(theta)
        px = pred_x_norm + dx
        py = pred_y_norm + dy
        triangle_points.append((px, py))

    # Visualization
    plt.subplot(2, 2, idx)
    plt.imshow(current_img)
    plt.scatter(target_x_abs, target_y_abs, color="green", label="Target", s=30)
    plt.scatter(pred_x_abs, pred_y_abs, color="blue", label="Prediction", s=30)

    # Draw triangle points and squares
    for pt in triangle_points:
        x_abs = pt[0] * IMG_WIDTH
        y_abs = pt[1] * IMG_HEIGHT

        # Light blue dot
        plt.scatter(x_abs, y_abs, color="skyblue", s=20, label="Stage2 Point" if pt == triangle_points[0] else None)

        # Shaded rectangle (crop box)
        box_size = 224
        rect_x = x_abs - box_size / 2
        rect_y = y_abs - box_size / 2
        rect = plt.Rectangle((rect_x, rect_y), box_size, box_size,
                             linewidth=1.5, edgecolor="skyblue", facecolor="skyblue", alpha=0.2)
        plt.gca().add_patch(rect)

    plt.legend()
    distance = np.sqrt((pred_x_norm - target_x) ** 2 + (pred_y_norm - target_y) ** 2)
    plt.title(f"Distance: {distance:.4f}")

plt.tight_layout()
plt.show()
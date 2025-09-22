import json
import random
import torch
from pathlib import Path
from PIL import Image
import matplotlib.pyplot as plt
import pandas as pd
import torchvision.transforms as T
from stage1_model_v3 import Stage1Model  # updated model
import numpy as np

# ------------------------
# Configurations
# ------------------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
DATA_ROOT = PROJECT_ROOT / "sequences"
INDEX_FILE = PROJECT_ROOT / "configs" / "dataset_index.json"
CHECKPOINT_PATH = PROJECT_ROOT / "models" / "stage1_threePred_best.pth"

device = "cuda" if torch.cuda.is_available() else "cpu"
IMG_WIDTH, IMG_HEIGHT = 1920, 1080

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
    past_tensor = torch.stack([transform(img) for img in past_imgs]).unsqueeze(0).to(device)

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
        output = model(current_tensor, past_tensor, positions)  # [1, 6]
        pred_xy = output.view(-1, 3, 2).squeeze(0).cpu().numpy()  # [3, 2]

    # Convert predictions and target to pixel coords
    pred_coords = [(x * IMG_WIDTH, y * IMG_HEIGHT) for x, y in pred_xy]
    target_x_abs, target_y_abs = target_x * IMG_WIDTH, target_y * IMG_HEIGHT

    # Visualization
    plt.subplot(2, 2, idx)
    plt.imshow(current_img)
    plt.scatter(target_x_abs, target_y_abs, color="yellow", label="Ground Truth", s=100, marker="*")

    for j, (px, py) in enumerate(pred_coords):
        plt.scatter(px, py, color="blue", s=80, label=f"Prediction {j+1}" if j == 0 else None)

    # Optional: Euclidean distances
    distances = [np.sqrt((px / IMG_WIDTH - target_x) ** 2 + (py / IMG_HEIGHT - target_y) ** 2) for px, py in pred_coords]
    title_text = " / ".join([f"P{j+1}: {d:.4f}" for j, d in enumerate(distances)])

    plt.title(f"Distances → {title_text}")
    plt.legend()

plt.tight_layout()
plt.show()

import torch
import matplotlib.pyplot as plt
import json
from pathlib import Path
from dataset import Stage1Dataset
from stage1_model_v2 import Stage1Model
from train_stage1_v2 import relaxed_l2_loss
from torchvision import transforms
from PIL import Image
import numpy as np

# ------------------------
# Config
# ------------------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent
# MODEL_PATH = "outputs/checkpoints/stage1_v2_best.pth"  # Change if needed
MODEL_PATH = PROJECT_ROOT / "models" / "stage1_v2_best.pth"
# ROOT_DIR = Path("sequences")                        # Root dataset folder
ROOT_DIR = PROJECT_ROOT / "sequences"
# INDEX_FILE = Path("configs/dataset_index.json")     # Dataset index file
INDEX_FILE = PROJECT_ROOT / "configs" / "dataset_index.json"
# HARD_INDEX_FILE = Path("configs/hard_index.json")   # File for hard samples
HARD_INDEX_FILE = PROJECT_ROOT / "configs" / "hard_index.json"
SPLIT = "train"                                      # Which split to evaluate
NUM_PLOTS = 9                                     # Number of hardest samples to visualize
HARD_SAMPLE_LIMIT = 200                             # Number of samples to save (0 = skip)
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print(DEVICE)

# ------------------------
# Load Model
# ------------------------
model = Stage1Model()
model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
model = model.to(DEVICE).eval()

# ------------------------
# Load Dataset
# ------------------------
dataset = Stage1Dataset(ROOT_DIR, INDEX_FILE, split=SPLIT, train_mode=False)
print(f"Loaded {len(dataset)} samples from {SPLIT} split.")

# ------------------------
# Compute per-sample loss
# ------------------------
sample_losses = []
for idx in range(len(dataset)):
    sample = dataset[idx]
    current_img = sample["current_img"].unsqueeze(0).to(DEVICE)
    past_imgs = sample["past_imgs"].unsqueeze(0).to(DEVICE)
    positions = sample["positions"].unsqueeze(0).to(DEVICE)
    target = sample["target"].to(DEVICE)

    with torch.no_grad():
        pred_xy = model(current_img, past_imgs, positions)  # [1, 2]
        loss_xy = relaxed_l2_loss(pred_xy, target[:2].unsqueeze(0), margin=0.03)
        weighted_loss = (target[2] * loss_xy).item()

    sample_path = dataset.sample_dirs[idx]  # e.g., "010\\00541"
    sample_losses.append((sample_path, weighted_loss, pred_xy.squeeze().cpu(), target.cpu()))

# Sort by loss (descending)
sample_losses.sort(key=lambda x: x[1], reverse=True)


# ------------------------
# Save hard samples to JSON
# ------------------------
if HARD_SAMPLE_LIMIT > 0:
    print(f"\nSaving top {HARD_SAMPLE_LIMIT} hard samples to {HARD_INDEX_FILE}...")
    top_for_json = sample_losses[:HARD_SAMPLE_LIMIT]
    formatted_paths = [p for p, *_ in top_for_json]

    # Print them with losses
    print("\nPaths and losses:")
    for p, l, *_ in top_for_json:
        print(f"{p} -> Loss: {l:.4f}")

    if HARD_INDEX_FILE.exists():
        with open(HARD_INDEX_FILE, "r") as f:
            hard_index = json.load(f)
    else:
        hard_index = {"train": []}

    # Deduplicate
    existing = set(hard_index["train"])
    new_entries = [p for p in formatted_paths if p not in existing]
    hard_index["train"].extend(new_entries)

    HARD_INDEX_FILE.parent.mkdir(parents=True, exist_ok=True)
    with open(HARD_INDEX_FILE, "w") as f:
        json.dump(hard_index, f, indent=4)

    print(f"\n✅ Added {len(new_entries)} new entries. Total now: {len(hard_index['train'])}")
else:
    print("\n⚠ HARD_SAMPLE_LIMIT is 0, skipping JSON save.")


# ------------------------
# Plot hardest samples
# ------------------------
top_samples = sample_losses[:NUM_PLOTS]
print(f"\nTop {NUM_PLOTS} hardest samples (by weighted loss):")
for path, loss, pred, target in top_samples:
    print(f"{path} -> Loss: {loss:.4f}")

    # Convert normalized coords to image size
    img_path = ROOT_DIR / path.replace("\\", "/") / "frame.jpg"
    img = Image.open(img_path).convert("RGB")
    w, h = img.size
    pred_x, pred_y = pred[0].item() * w, pred[1].item() * h
    tgt_x, tgt_y = target[0].item() * w, target[1].item() * h
    dist = np.sqrt((pred_x - tgt_x) ** 2 + (pred_y - tgt_y) ** 2)

    plt.figure(figsize=(6, 6))
    plt.imshow(img)
    plt.scatter([tgt_x], [tgt_y], c='green', label="Target")
    plt.scatter([pred_x], [pred_y], c='red', label="Predicted")
    plt.legend()
    plt.title(f"{path}\nLoss: {loss:.4f}, Dist: {dist:.1f}px")
    plt.show()
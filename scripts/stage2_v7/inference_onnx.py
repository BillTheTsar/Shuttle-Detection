import onnxruntime as ort
import numpy as np
import torch
from pathlib import Path

# ---- 1. Load ONNX model ----
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
session = ort.InferenceSession(PROJECT_ROOT / "onnx_exports/stage2_v7.onnx")

# ---- 2. Prepare inputs (convert to NumPy) ----
B = 4
current_crop = torch.rand(B, 3, 224, 224).numpy().astype(np.float32)
past_crops = torch.rand(B, 3, 3, 224, 224).numpy().astype(np.float32)
positions = torch.rand(B, 30, 3).numpy().astype(np.float32)

# ---- 3. Run inference ----
inputs = {
    "current_crop": current_crop,
    "past_crops": past_crops,
    "positions": positions
}

outputs = session.run(["xy", "heatmap"], inputs)

# ---- 4. Print results ----
xy, heatmap = outputs
print("xy:", xy)
# print("vis_logit:", vis_logit)
print("heatmap shape:", heatmap.shape)
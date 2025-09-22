import torch
from pathlib import Path
from scripts.model import Stage1ModelBiFPN  # Adjust import path

# ---- 1. Initialize model ----
model = Stage1ModelBiFPN(tau=1.0)
model.eval()

# Optional: load weights
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
model.load_state_dict(torch.load(PROJECT_ROOT / "models/stage1_hm_best_epoch25.pth", map_location="cpu"))

# ---- 2. Prepare dummy input ----
B = 1  # Batch size 1
current = torch.rand(B, 3, 300, 300)
past = torch.rand(B, 3, 3, 300, 300)
positions = torch.rand(B, 30, 3)

# ---- 3. Export to ONNX ----
torch.onnx.export(
    model,
    (current, past, positions),
    PROJECT_ROOT / "onnx_exports/stage1_v4.onnx",
    input_names=["current", "past", "positions"],
    output_names=["xy", "heatmap"],
    dynamic_axes={
        "current": {0: "batch_size"},
        "past": {0: "batch_size"},
        "positions": {0: "batch_size"},
        "xy": {0: "batch_size"},
        "heatmap": {0: "batch_size"}
    },
    opset_version=19,  # Or 11–17; ONNX Runtime supports up to 17+
    do_constant_folding=True
)

print("✅ Exported stage1_v4.onnx")
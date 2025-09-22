import torch
from pathlib import Path
from scripts.model import Stage2ModelHeatmap  # Adjust import path

# ---- 1. Initialize model ----
model = Stage2ModelHeatmap(tau=1.0)
model.eval()

# Optional: load weights
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
model.load_state_dict(torch.load(PROJECT_ROOT / "models/stage2_hm_best_epoch260.pth", map_location="cpu"))

# ---- 2. Prepare dummy input ----
B = 1  # Batch size 1
current_crop = torch.rand(B, 3, 224, 224)
past_crops = torch.rand(B, 3, 3, 224, 224)
positions = torch.rand(B, 30, 3)

# ---- 3. Export to ONNX ----
torch.onnx.export(
    model,
    (current_crop, past_crops, positions),
    PROJECT_ROOT / "onnx_exports/stage2_v4.onnx",
    input_names=["current_crop", "past_crops", "positions"],
    output_names=["xy", "vis_logit", "heatmap"],
    dynamic_axes={
        "current_crop": {0: "batch_size"},
        "past_crops": {0: "batch_size"},
        "positions": {0: "batch_size"},
        "xy": {0: "batch_size"},
        "vis_logit": {0: "batch_size"},
        "heatmap": {0: "batch_size"}
    },
    opset_version=19,  # Or 11–17; ONNX Runtime supports up to 17+
    do_constant_folding=True
)

print("✅ Exported stage2_v4.onnx")
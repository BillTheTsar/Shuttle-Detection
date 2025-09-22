import os
import json
import time
import random
import argparse
from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

from stage2_model_v3 import Stage2Model
from dataloader import get_stage2_loader

# ------------------------
# Helpers
# ------------------------
@torch.no_grad()
def eval_split(model, loader, device, xy_threshold=0.05):
    """
    Returns a set of relative paths for samples where:
      - target visibility == 1
      - Euclidean distance(pred_xy, target_xy) > xy_threshold
    """
    model.eval()
    hard_paths = set()

    for batch in loader:
        current_img = batch["current_crop"].to(device)
        past_imgs   = batch["past_crops"].to(device)
        positions   = batch["positions"].to(device)
        target      = batch["target"].to(device)     # [B, 3]
        paths       = batch["path"]                  # list of strings

        out = model(current_img, past_imgs, positions)
        pred_xy = out["xy"]                          # [B, 2], already in [0,1] if you use sigmoid head; else raw

        # Euclidean distance (no margin) per sample
        diff = pred_xy - target[:, :2]
        dist = torch.sqrt((diff ** 2).sum(dim=1) + 1e-12)

        vis_mask = (target[:, 2] > 0.5)              # consider only visible samples
        bad_mask = (dist > xy_threshold) & vis_mask

        for is_bad, p in zip(bad_mask.tolist(), paths):
            if is_bad:
                hard_paths.add(p)

    return hard_paths

# ------------------------
# Main
# ------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=str, required=True)
    parser.add_argument("--index-file", type=str, required=True)
    parser.add_argument("--save-dir", type=str, default="outputs/hard_mining")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--num-workers", type=int, default=8)
    parser.add_argument("--trials", type=int, default=3)
    parser.add_argument("--xy-threshold", type=float, default=0.05)
    parser.add_argument("--splits", type=str, nargs="+", default=["train", "val"])  # which splits to mine
    parser.add_argument("--seed", type=int, default=1337)
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    root_dir = Path(args.data_dir)
    index_file = Path(args.index_file)
    save_dir = Path(args.save_dir)
    os.makedirs(save_dir, exist_ok=True)

    # Evaluate each split with each best checkpoint; union failing samples
    hard = {split: set() for split in args.splits}

    # fresh model per checkpoint
    model = Stage2Model().to(device)
    sd = torch.load(Path("storage/checkpoints/stage2_best.pth"), map_location=device)
    model.load_state_dict(sd)

    for trial in args.trials:
        for split in args.splits:
            loader = get_stage2_loader(
                root_dir, index_file, split=split,
                batch_size=args.batch_size, num_workers=args.num_workers,
                train_mode=True, apply_flip=False, jitter_enabled=False, noise_std=0, position_perturbation=0
            )
            bad_paths = eval_split(model, loader, device, xy_threshold=args.xy_threshold)
            hard[split].update(bad_paths)

    # Make JSON matching dataset_index.json structure
    # If `test` exists in your original index and you want to process it, add it to --splits.
    hard_json = {k: sorted(list(v)) for k, v in hard.items()}

    out_path = save_dir / "stage2_hard_examples_index.json"
    with open(out_path, "w") as f:
        json.dump(hard_json, f, indent=2)
    print(f"\n📝 Wrote hard-example index to: {out_path}")
    for k in hard_json:
        print(f"  {k}: {len(hard_json[k])} samples > {args.xy_threshold}")

if __name__ == "__main__":
    main()
import os
import time
import argparse
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
from torch.cuda import amp
from torch.utils.tensorboard import SummaryWriter

from stage2_model_v1 import Stage2Model
from dataloader import get_stage2_loader

import warnings
warnings.filterwarnings("ignore", message=".*CUDA is not available.*")


# ------------------------
# Utils
# ------------------------
def relaxed_l2_loss(pred_xy, target_xy, margin: float):
    """
    pred_xy, target_xy: [B, 2]
    returns per-sample loss (no .mean())
    """
    diff = pred_xy - target_xy
    dist = torch.sqrt((diff ** 2).sum(dim=1))
    penalty = torch.clamp(dist - margin, min=0.0)
    return penalty


def save_and_log_checkpoint(model, path: Path, writer, step):
    torch.save(model.state_dict(), path)
    if path.exists():
        size = path.stat().st_size
        print(f"✅ Saved checkpoint at {path} ({size / 1024:.2f} KB)")
        if writer:
            writer.add_scalar("Checkpoint/file_size_bytes", size, step)
    else:
        print(f"❌ Failed to save checkpoint at {path}")

IMAGENET_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(1,3,1,1)
IMAGENET_STD  = torch.tensor([0.229, 0.224, 0.225]).view(1,3,1,1)

def _maybe_denorm(img_t: torch.Tensor):
    """
    img_t: [3,H,W], either already in [0,1] or ImageNet-normalized.
    Returns np.uint8 HxWx3.
    """
    x = img_t.detach().cpu()
    # Heuristic: if many values are <0 or >1, assume ImageNet norm.
    if (x.min() < 0.0) or (x.max() > 1.0):
        x = (x.unsqueeze(0) * IMAGENET_STD + IMAGENET_MEAN).squeeze(0)
    x = x.clamp(0,1)
    x = (x.permute(1,2,0).numpy() * 255).astype(np.uint8)
    return x

def _draw_point(ax, xy, visible: float, color="lime"):
    x, y = float(xy[0]), float(xy[1])
    # coords assumed normalized to [0,1] in the crop
    # Simplify: compute from image array size instead
    img_h, img_w = ax.images[0].get_array().shape[:2]
    px, py = x * img_w, y * img_h
    if np.isnan(px) or np.isnan(py):
        return
    m = "o" if visible >= 0.5 else "x"
    ax.scatter([px], [py], s=60, marker=m, edgecolors=color, facecolors="none", linewidths=2)

def _save_panel(split, save_root: Path, epoch: int, step: int, sample_idx: int,
                current_img: torch.Tensor,
                past_imgs: torch.Tensor,   # [N,3,H,W]
                positions: torch.Tensor,   # [T,3] normalized in crop coords
                target: torch.Tensor,      # [3]
                past_to_show: int = 3):
    """
    Assumptions:
      - positions is time-ordered; we map the *last* k entries to the *last* k past crops.
      - target is [x,y,vis] for the current_img.
    """
    save_dir = save_root / (f"{split}_data_visualization")
    save_dir.mkdir(parents=True, exist_ok=True)

    # Build figure: 1 (current) + up to 3 past
    N = past_imgs.shape[0]
    k = min(N, past_to_show)
    cols = 1 + k
    fig, axes = plt.subplots(1, cols, figsize=(4*cols, 4))
    if cols == 1:
        axes = [axes]

    # Current
    cur_img_np = _maybe_denorm(current_img)
    axes[0].imshow(cur_img_np)
    axes[0].set_title("current")
    axes[0].axis("off")
    _draw_point(axes[0], target[:2].cpu().numpy(), target[2].item(), color="yellow")

    # Past: show the *last* k past crops, mapped to the *last* k positions
    # If positions has length T, we take positions[-k:] for these images.
    T = positions.shape[0]
    pos_block = positions[-k:] if T >= k else positions
    for j in range(k):
        img_np = _maybe_denorm(past_imgs[-k + j])  # last k, left-to-right oldest->newest within those k
        axes[1+j].imshow(img_np)
        axes[1+j].set_title(f"past[{N - k + j}]")
        axes[1+j].axis("off")
        xyv = pos_block[j]
        _draw_point(axes[1+j], xyv[:2].cpu().numpy(), xyv[2].item(), color="lime")

    fname = save_dir / f"ep{epoch:02d}_step{step:06d}_idx{sample_idx:02d}.png"
    fig.tight_layout()
    fig.savefig(fname, dpi=120)
    plt.close(fig)



# ------------------------
# Main
# ------------------------
def main():
    PROJECT_ROOT = Path(__file__).resolve().parent.parent

    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=str, default=str(PROJECT_ROOT / "storage" / "dataset"))
    parser.add_argument("--save-dir", type=str, default="/storage/checkpoints")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--accum-steps", type=int, default=2)
    parser.add_argument("--lr-backbone", type=float, default=1e-4)   # discriminative LR
    parser.add_argument("--lr-head", type=float, default=5e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--num-workers", type=int, default=8)
    parser.add_argument("--print-freq", type=int, default=40)
    parser.add_argument("--margin", type=float, default=0.01)        # tighter than stage1
    parser.add_argument("--freeze-backbone-epochs", type=int, default=0)  # 0 = no freeze
    parser.add_argument("--vis-freq", type=int, default=40,
                        help="How often (steps) to save a visualization during an epoch")
    parser.add_argument("--vis-max-per-epoch", type=int, default=5,
                        help="Cap number of visualizations per epoch per split")

    args = parser.parse_args()

    CONFIG = {
        "root_dir": Path(args.data_dir),
        "index_file": PROJECT_ROOT / "configs/dataset_index.json",
        "save_dir": Path(args.save_dir),
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "accum_steps": args.accum_steps,
        "lr_backbone": args.lr_backbone,
        "lr_head": args.lr_head,
        "weight_decay": args.weight_decay,
        "num_workers": args.num_workers,
        "print_freq": args.print_freq,
        "margin": args.margin,
        "freeze_backbone_epochs": args.freeze_backbone_epochs,
        "device": "cuda" if torch.cuda.is_available() else "cpu",
        "vis_freq": args.vis_freq,
        "vis_max_per_epoch": args.vis_max_per_epoch,
    }

    os.makedirs(CONFIG["save_dir"], exist_ok=True)

    # ------------------------
    # Data
    # ------------------------
    def get_dataloaders():
        train_loader = get_stage2_loader(
            CONFIG["root_dir"], CONFIG["index_file"], split="train",
            batch_size=CONFIG["batch_size"], num_workers=CONFIG["num_workers"],
            train_mode=True, position_perturbation=0.002
        )
        val_loader = get_stage2_loader(
            CONFIG["root_dir"], CONFIG["index_file"], split="val",
            batch_size=CONFIG["batch_size"], num_workers=CONFIG["num_workers"],
            train_mode=True, apply_flip=False, jitter_enabled=False,noise_std=0, position_perturbation=0
        )
        return train_loader, val_loader

    train_loader, val_loader = get_dataloaders()

    # ------------------------
    # Model
    # ------------------------
    device = CONFIG["device"]
    model = Stage2Model().to(device)

    # Optional freeze (epoch-based)
    if CONFIG["freeze_backbone_epochs"] > 0:
        for p in model.feature_extractor.parameters():
            p.requires_grad = False
        print(f"🔒 Freezing EfficientNet-B0 for {CONFIG['freeze_backbone_epochs']} epoch(s).")

    # Losses
    bce = nn.BCEWithLogitsLoss()

    # Optimizer with discriminative LR
    backbone_params = [p for p in model.feature_extractor.parameters() if p.requires_grad]
    head_params = [
        *model.attn_pool.parameters(),
        *model.traj_encoder.parameters(),
        *model.mlp.parameters(),
    ]
    optimizer = optim.AdamW(
        [
            {"params": backbone_params, "lr": CONFIG["lr_backbone"]},
            {"params": head_params,     "lr": CONFIG["lr_head"]},
        ],
        weight_decay=CONFIG["weight_decay"]
    )

    # Scheduler (simple step; swap for cosine if you prefer)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=3, gamma=0.8)

    scaler = amp.GradScaler()
    writer = SummaryWriter(log_dir=CONFIG["save_dir"] / "logs_stage2")
    best_val = float("inf")
    global_step = 0

    # ------------------------
    # Training
    # ------------------------
    for epoch in range(CONFIG["epochs"]):
        # Unfreeze after warm epochs (if any)
        if epoch == CONFIG["freeze_backbone_epochs"]:
            for p in model.feature_extractor.parameters():
                p.requires_grad = True
            print("🔓 Unfroze EfficientNet-B0 backbone.")

        model.train()
        running = 0.0
        vis_saved_train = 0
        start = time.time()

        # log lrs
        curr_lrs = [pg["lr"] for pg in optimizer.param_groups]
        writer.add_scalar("LR/backbone", curr_lrs[0], epoch)
        writer.add_scalar("LR/heads", curr_lrs[1], epoch)

        optimizer.zero_grad()

        for i, batch in enumerate(train_loader):
            current_img = batch["current_crop"].to(device)
            past_imgs   = batch["past_crops"].to(device)
            positions   = batch["positions"].to(device)
            target      = batch["target"].to(device)   # [B, 3]

            try:
                if (i % CONFIG["vis_freq"] == 0) and (vis_saved_train < CONFIG["vis_max_per_epoch"]):
                    b_index = 6  # 7th sample
                    if current_img.size(0) > b_index:
                        curr = current_img[b_index].cpu()
                        past = past_imgs[b_index].cpu()  # [N,3,H,W]
                        traj = positions[b_index].cpu()  # [T,3]
                        tgt = target[b_index].cpu()  # [3]
                        _save_panel(
                            split="training",
                            save_root=PROJECT_ROOT,
                            epoch=epoch + 1,
                            step=global_step,
                            sample_idx=b_index,
                            current_img=curr,
                            past_imgs=past,
                            positions=traj,
                            target=tgt,
                            past_to_show=3
                        )
                        vis_saved_train += 1
            except Exception as e:
                print(f"[vis][train] failed to save visualization at step {i}: {e}")

            # with amp.autocast():
            logits = model(current_img, past_imgs, positions)  # [B, 3]
            pred_xy = logits[:, :2]       # raw (no sigmoid)
            pred_vis_logit = logits[:, 2] # raw logit

            # Losses
            loss_xy_ps = relaxed_l2_loss(pred_xy, target[:, :2], margin=CONFIG["margin"])
            loss_vis   = bce(pred_vis_logit, target[:, 2])

            # Soft gate xy by predicted visibility (differentiable)
            xy_term = target[:, 2] * loss_xy_ps
            xy_term = torch.nan_to_num(xy_term, nan=0, posinf=0, neginf=0)
            loss = xy_term.mean() + loss_vis
                # loss = xy_term.mean()

            scaler.scale(loss).backward()

            if (i + 1) % CONFIG["accum_steps"] == 0:
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()

            running += loss.item()
            global_step += 1

            if (i + 1) % CONFIG["print_freq"] == 0:
                avg_loss = running / CONFIG["print_freq"]
                print(f"Ep [{epoch+1}/{CONFIG['epochs']}], "
                      f"Step [{i+1}/{len(train_loader)}], "
                      f"Loss: {avg_loss:.4f}")
                writer.add_scalar("Loss/train_total", avg_loss, global_step)
                writer.add_scalar("Loss/train_xy", loss_xy_ps.mean().item(), global_step)
                writer.add_scalar("Loss/train_vis_bce", loss_vis.item(), global_step)
                running = 0.0

        # ------------------------
        # Validation
        # ------------------------
        model.eval()
        vis_saved_val = 0
        val_total = 0.0
        val_xy = 0.0
        val_vis = 0.0
        with torch.no_grad():
            for i, batch in enumerate(val_loader):
                current_img = batch["current_crop"].to(device)
                past_imgs   = batch["past_crops"].to(device)
                positions   = batch["positions"].to(device)
                target      = batch["target"].to(device)

                try:
                    if (i % CONFIG["vis_freq"] == 0) and (vis_saved_val < CONFIG["vis_max_per_epoch"]):
                        b_index = 6
                        if current_img.size(0) > b_index:
                            curr = current_img[b_index].cpu()
                            past = past_imgs[b_index].cpu()
                            traj = positions[b_index].cpu()
                            tgt = target[b_index].cpu()
                            _save_panel(
                                split="validation",
                                save_root=PROJECT_ROOT,
                                epoch=epoch + 1,
                                step=i,  # per-epoch step for val
                                sample_idx=b_index,
                                current_img=curr,
                                past_imgs=past,
                                positions=traj,
                                target=tgt,
                                past_to_show=3
                            )
                            vis_saved_val += 1
                except Exception as e:
                    print(f"[vis][val] failed to save visualization at step {i}: {e}")

                # with amp.autocast():
                logits = model(current_img, past_imgs, positions)
                pred_xy = logits[:, :2]
                pred_vis_logit = logits[:, 2]

                loss_xy_ps = relaxed_l2_loss(pred_xy, target[:, :2], margin=CONFIG["margin"])
                loss_vis   = bce(pred_vis_logit, target[:, 2])

                xy_term = target[:, 2] * loss_xy_ps
                xy_term = torch.nan_to_num(xy_term, nan=0, posinf=0, neginf=0)
                loss = xy_term.mean() + loss_vis
                # loss = xy_term.mean()

                val_total += loss.item()
                val_xy    += loss_xy_ps.mean().item()
                val_vis   += loss_vis.item()

        val_total /= len(val_loader)
        val_xy    /= len(val_loader)
        val_vis   /= len(val_loader)

        duration = time.time() - start
        print(f"Epoch {epoch+1} in {duration:.1f}s  "
              f"- Val Total: {val_total:.4f} | Val XY: {val_xy:.4f} | Val Vis: {val_vis:.4f}")

        writer.add_scalar("Loss/val_total", val_total, epoch)
        writer.add_scalar("Loss/val_xy", val_xy, epoch)
        writer.add_scalar("Loss/val_vis_bce", val_vis, epoch)

        # Save best
        if val_total < best_val:
            best_val = val_total
            save_and_log_checkpoint(
                model, Path(CONFIG["save_dir"]) / "stage2_best.pth",
                writer, global_step
            )
            print("✅ Saved new best model.")

        scheduler.step()

    # Save last
    save_and_log_checkpoint(
        model, Path(CONFIG["save_dir"]) / "stage2_last.pth",
        writer, global_step
    )
    writer.close()
    print("Training completed!")


if __name__ == "__main__":
    main()

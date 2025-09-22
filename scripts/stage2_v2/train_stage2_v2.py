import os
import time
import argparse
from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim
from torch.cuda import amp
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter

from stage2_model_v2 import Stage2Model
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
    dist = torch.sqrt((diff ** 2).sum(dim=1) + 1e-12)
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


def split_weight_decay(module):
    """Return (decay, no_decay) parameter lists for a module.

    Rules:
      - no decay for biases
      - no decay for normalization params (BatchNorm/LayerNorm, etc.)
      - decay for everything else
    """
    decay, no_decay = [], []
    for full_name, p in module.named_parameters():
        if not p.requires_grad:
            continue
        if full_name.endswith("bias") or "norm" in full_name.lower() or "bn" in full_name.lower():
            no_decay.append(p)
        else:
            decay.append(p)
    return decay, no_decay


# ------------------------
# Main
# ------------------------
def main():
    PROJECT_ROOT = Path(__file__).resolve().parent.parent

    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=str, default=str(PROJECT_ROOT / "storage" / "dataset"))
    parser.add_argument("--save-dir", type=str, default="/storage/checkpoints")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--accum-steps", type=int, default=4)
    parser.add_argument("--lr-backbone", type=float, default=5e-5)   # discriminative LR
    parser.add_argument("--lr-head", type=float, default=2e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--num-workers", type=int, default=8)
    parser.add_argument("--print-freq", type=int, default=40)
    parser.add_argument("--margin", type=float, default=0.01)        # tighter than stage1
    parser.add_argument("--freeze-backbone-epochs", type=int, default=0)  # 0 = no freeze
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

    # Split backbone
    bb_decay, bb_no = split_weight_decay(model.feature_extractor)

    # Split heads (attn_pool, traj_encoder, mlp)
    head_decay, head_no = [], []
    for m in (model.attn_pool, model.traj_encoder, model.mlp):
        d, nd = split_weight_decay(m)
        head_decay += d
        head_no += nd

    optimizer = optim.AdamW(
        [
            # Backbone: same LR, different decay
            {"params": bb_decay, "lr": CONFIG["lr_backbone"], "weight_decay": CONFIG["weight_decay"]},
            {"params": bb_no, "lr": CONFIG["lr_backbone"], "weight_decay": 0.0},

            # Heads: same LR, different decay
            {"params": head_decay, "lr": CONFIG["lr_head"], "weight_decay": CONFIG["weight_decay"]},
            {"params": head_no, "lr": CONFIG["lr_head"], "weight_decay": 0.0},
        ],
        betas=(0.9, 0.999)  # optional; keep your defaults if you prefer
    )

    # Scheduler (simple step; swap for cosine if you prefer)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=3, gamma=0.85)

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
        start = time.time()

        # log lrs
        curr_lrs = [pg["lr"] for pg in optimizer.param_groups]
        writer.add_scalar("LR/backbone", curr_lrs[0], epoch)
        writer.add_scalar("LR/heads", curr_lrs[1], epoch)

        optimizer.zero_grad()

        # Slow warm-up of lambda_vis
        lambda_vis_max = 0.5  # e.g., vis contributes up to 1/3–1/2 of XY scale
        warmup_epochs = 5
        lambda_vis = min(1.0, max(0.0, (epoch + 1) / warmup_epochs)) * lambda_vis_max

        for i, batch in enumerate(train_loader):
            current_img = batch["current_crop"].to(device)
            past_imgs   = batch["past_crops"].to(device)
            positions   = batch["positions"].to(device)
            target      = batch["target"].to(device)   # [B, 3]

            with amp.autocast():
                logits = model(current_img, past_imgs, positions)  # [B, 3]
                pred_xy = logits[:, :2]       # raw (no sigmoid)
                pred_vis_logit = logits[:, 2] # raw logit

                # Losses
                loss_xy_ps = relaxed_l2_loss(pred_xy, target[:, :2], margin=CONFIG["margin"])
                # loss_vis   = bce(pred_vis_logit, target[:, 2])
                pred_vis = torch.sigmoid(pred_vis_logit) # An alternative
                loss_vis = F.mse_loss(pred_vis, target[:, 2])

                xy_term = target[:, 2] * loss_xy_ps
                xy_term = torch.nan_to_num(xy_term, nan=0, posinf=0, neginf=0)
                loss = xy_term.mean() + lambda_vis * loss_vis

            scaler.scale(loss).backward()

            if (i + 1) % CONFIG["accum_steps"] == 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1)
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()

            running += loss.item()
            global_step += 1

            if (i + 1) % CONFIG["print_freq"] == 0:
                avg_loss = running / CONFIG["print_freq"]
                print(f"Ep [{epoch+1}/{CONFIG['epochs']}], "
                      f"Step [{i+1}/{len(train_loader)}], "
                      f"Loss: {avg_loss:.4f}, "
                      f"Loss_xy: {xy_term.mean().item():.4f}, "
                      f"Loss_vis: {loss_vis.item():.4f}")
                writer.add_scalar("Loss/train_total", avg_loss, global_step)
                writer.add_scalar("Loss/train_xy", xy_term.mean().item(), global_step)
                writer.add_scalar("Loss/train_vis_bce", loss_vis.item(), global_step)
                running = 0.0

        # ------------------------
        # Validation
        # ------------------------
        model.eval()
        val_total = 0.0
        val_xy = 0.0
        val_vis = 0.0
        with torch.no_grad():
            for batch in val_loader:
                current_img = batch["current_crop"].to(device)
                past_imgs   = batch["past_crops"].to(device)
                positions   = batch["positions"].to(device)
                target      = batch["target"].to(device)

                with amp.autocast():
                    logits = model(current_img, past_imgs, positions)
                    pred_xy = logits[:, :2]
                    pred_vis_logit = logits[:, 2]

                    loss_xy_ps = relaxed_l2_loss(pred_xy, target[:, :2], margin=CONFIG["margin"])
                    # loss_vis   = bce(pred_vis_logit, target[:, 2])
                    pred_vis = torch.sigmoid(pred_vis_logit)
                    loss_vis = F.mse_loss(pred_vis, target[:, 2])

                    xy_term = target[:, 2] * loss_xy_ps
                    xy_term = torch.nan_to_num(xy_term, nan=0, posinf=0, neginf=0)
                    loss = xy_term.mean() + lambda_vis * loss_vis
                    # loss = xy_term.mean()

                val_total += loss.item()
                val_xy    += xy_term.mean().item()
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

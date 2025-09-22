# train_stage1_v4.py
import os, time, argparse
from pathlib import Path
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter

from stage1_model_v4 import Stage1ModelBiFPN
from scripts.dataloader import get_stage1_loader  # assumes you have this

# ---------- utils ----------
def relaxed_l2_loss(pred_xy, target_xy, margin: float):
    diff = pred_xy - target_xy
    dist = torch.sqrt((diff ** 2).sum(dim=1) + 1e-12)
    return torch.clamp(dist - margin, min=0.0)  # [B]

def split_weight_decay(module):
    decay, no_decay = [], []
    for n, p in module.named_parameters():
        if not p.requires_grad:
            continue
        if n.endswith("bias") or "norm" in n.lower() or "bn" in n.lower():
            no_decay.append(p)
        else:
            decay.append(p)
    return decay, no_decay

# ---------- main ----------
def main():
    PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", type=str, default=str(PROJECT_ROOT / "sequences"))
    ap.add_argument("--index-file", type=str, default=str(PROJECT_ROOT / "configs" / "dataset_index.json"))
    ap.add_argument("--save-dir", type=Path, default=Path(PROJECT_ROOT / "temp_save"))
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--accum-steps", type=int, default=2)
    ap.add_argument("--lr-backbone", type=float, default=2e-5)
    ap.add_argument("--lr-head", type=float, default=6e-5)
    ap.add_argument("--weight-decay", type=float, default=1e-4)
    ap.add_argument("--num-workers", type=int, default=16)
    ap.add_argument("--print-freq", type=int, default=40)
    ap.add_argument("--margin", type=float, default=0.015)   # <-- wider margin
    ap.add_argument("--tau", type=float, default=1.0)
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    args.save_dir.mkdir(parents=True, exist_ok=True)

    # data
    train_loader = get_stage1_loader(
        Path(args.data_dir), Path(args.index_file), split="train",
        batch_size=args.batch_size, num_workers=args.num_workers,
        train_mode=True, noise_std=0.02, position_perturbation=0.03
    )
    val_loader = get_stage1_loader(
        Path(args.data_dir), Path(args.index_file), split="val",
        batch_size=args.batch_size, num_workers=args.num_workers,
        train_mode=False, apply_flip=False, jitter_enabled=False, noise_std=0, position_perturbation=0
    )

    # model
    model = Stage1ModelBiFPN(tau=args.tau).to(device)
    sd = torch.load(Path(PROJECT_ROOT / "models/stage1_hm_best_epoch25.pth"))
    model.load_state_dict(sd)
    print(f"Loaded weights from {Path(PROJECT_ROOT / "models/stage1_hm_best_epoch25.pth")}")

    # optimizer param groups
    bb_decay, bb_no = split_weight_decay(model.backbone)
    head_modules = [
        model.bifpn_first, model.bifpn_iters, model.head
    ]
    head_decay, head_no = [], []
    for m in head_modules:
        d, nd = split_weight_decay(m)
        head_decay += d; head_no += nd
    # include the aux gate parameter (no weight decay)
    extra = [p for n,p in model.named_parameters() if n.startswith("aux_gate")]

    optimizer = optim.AdamW(
        [
            {"params": bb_decay, "lr": args.lr_backbone, "weight_decay": args.weight_decay},
            {"params": bb_no,    "lr": args.lr_backbone, "weight_decay": 0.0},
            {"params": head_decay, "lr": args.lr_head, "weight_decay": args.weight_decay},
            {"params": head_no + extra, "lr": args.lr_head, "weight_decay": 0.0},
        ],
        betas=(0.9, 0.999)
    )
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=3, gamma=0.90)

    scaler = torch.amp.GradScaler(device="cuda")
    writer = SummaryWriter(log_dir=str(args.save_dir / "logs_stage1_hm"))
    best_val = float("inf"); global_step = 0

    for epoch in range(args.epochs):
        model.train(); running = 0.0; t0 = time.time()
        optimizer.zero_grad(set_to_none=True)

        for i, batch in enumerate(train_loader):
            cur = batch["current_img"].to(device)  # [B,3,300,300]
            past = batch["past_imgs"].to(device)  # [B,3,3,300,300]

            # Concatenate along "time" → [B,4,3,300,300]
            positions = batch["positions"].to(device)  # [B,30,3]
            target = batch["target"].to(device)        # [B,3]

            # with torch.amp.autocast("cuda"):
            out = model(cur, past, positions)
            pred_xy = out["xy"]                        # [B,2]

            # loss
            loss_xy_ps = relaxed_l2_loss(pred_xy, target[:, :2], margin=args.margin)
            xy_term = target[:, 2] * loss_xy_ps
            loss = xy_term.mean()  # no vis term here

            scaler.scale(loss).backward()

            if (i + 1) % args.accum_steps == 0:
                scaler.step(optimizer); scaler.update()
                optimizer.zero_grad(set_to_none=True)

            running += loss.item(); global_step += 1

            if (i + 1) % args.print_freq == 0:
                avg = running / args.print_freq
                writer.add_scalar("Loss/train_total", avg, global_step)
                writer.add_scalar("Loss/train_xy", xy_term.mean().item(), global_step)
                if hasattr(model, "aux_gate"):
                    writer.add_scalar("Aux/alpha", F.softplus(model.aux_gate).item(), global_step)
                print(f"Ep [{epoch+1}/{args.epochs}] Step [{i+1}/{len(train_loader)}] "
                      f"Loss {avg:.4f} | XY {xy_term.mean().item():.4f}")
                running = 0.0

        # ---- validation ----
        model.eval(); val_total = val_xy = 0.0
        with torch.no_grad():
            for batch in val_loader:
                cur = batch["current_img"].to(device)  # [B,3,300,300]
                past = batch["past_imgs"].to(device)  # [B,3,3,300,300]

                # Concatenate along "time" → [B,4,3,300,300]
                positions = batch["positions"].to(device)
                target = batch["target"].to(device)

                # with torch.amp.autocast("cuda"):
                out = model(cur, past, positions)
                loss_xy_ps = relaxed_l2_loss(out["xy"], target[:, :2], margin=args.margin)
                xy_term = target[:, 2] * loss_xy_ps
                loss = xy_term.mean()

                val_total += loss.item(); val_xy += xy_term.mean().item()

        val_total /= len(val_loader); val_xy /= len(val_loader)
        dt = time.time() - t0
        print(f"Epoch {epoch+1} in {dt:.1f}s | Val Total {val_total:.4f} | Val XY {val_xy:.4f}")
        writer.add_scalar("Val/total", val_total, epoch)
        writer.add_scalar("Val/xy", val_xy, epoch)

        if val_total < best_val:
            best_val = val_total
            torch.save(model.state_dict(), args.save_dir / "stage1_hm_best_epoch25.pth")
            print("✅ Saved new best.")

        scheduler.step()

    torch.save(model.state_dict(), args.save_dir / "stage1_hm_last.pth")
    writer.close()
    print("Training completed!")

if __name__ == "__main__":
    main()
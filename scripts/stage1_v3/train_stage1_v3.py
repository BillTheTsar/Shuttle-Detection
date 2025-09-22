import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.cuda import amp
from pathlib import Path
import os
import time
import argparse
from torch.utils.tensorboard import SummaryWriter

from stage1_model_v3 import Stage1Model
from dataloader import get_stage1_loader

import warnings
warnings.filterwarnings("ignore", message=".*CUDA is not available.*")

# ------------------------
# Configurations
# ------------------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent  # parent is scripts, parent of that is Shuttle Detection Model
parser = argparse.ArgumentParser()
parser.add_argument("--data-dir", type=str, default=str(PROJECT_ROOT / "sequences"))
args = parser.parse_args()

CONFIG = {
    "root_dir": Path(args.data_dir),
    "index_file": PROJECT_ROOT / "configs/dataset_index.json",
    "batch_size": 16,
    "accumulation_steps": 4,  # effective batch size = batch_size * accumulation_steps
    "epochs": 30,
    "lr": 1e-3,
    "weight_decay": 1e-4,
    "num_workers": 8,
    "device": "cuda" if torch.cuda.is_available() else "cpu",
    "save_dir": Path("/storage/checkpoints"),
    "print_freq": 30
}

os.makedirs(CONFIG["save_dir"], exist_ok=True)

# ------------------------
# Dataset & DataLoader
# ------------------------
def get_dataloaders():
    train_loader = get_stage1_loader(CONFIG["root_dir"], CONFIG["index_file"], split="train",
                        batch_size=CONFIG["batch_size"], num_workers=CONFIG["num_workers"], train_mode=True)

    val_loader = get_stage1_loader(CONFIG["root_dir"], CONFIG["index_file"], split="val",
                        batch_size=CONFIG["batch_size"], num_workers=CONFIG["num_workers"], train_mode=False)

    return train_loader, val_loader

# ------------------------
# Saving checkpoints
# ------------------------
def save_and_log_checkpoint(model, path, writer, step):
    torch.save(model.state_dict(), path)
    if path.exists():
        size = path.stat().st_size
        print(f"✅ Saved checkpoint at {path} ({size / 1024:.2f} KB)")
        writer.add_scalar("Checkpoint/file_size_bytes", size, step)
    else:
        print(f"❌ Failed to save checkpoint at {path}")

# Loss functions
def relaxed_l2_loss(pred_xy, target_xy, margin):
    # pred_xy, target_xy: [B, 2]
    diff = pred_xy - target_xy
    dist = torch.sqrt((diff ** 2).sum(dim=1))  # Euclidean distance per sample
    penalty = torch.clamp(dist - margin, min=0.0)  # max(0, d - margin)
    return penalty ** 2

def diversity_loss(preds, margin):
    # preds: [B, 3, 2]
    d1 = torch.norm(preds[:, 0] - preds[:, 1], dim=1)
    d2 = torch.norm(preds[:, 0] - preds[:, 2], dim=1)
    d3 = torch.norm(preds[:, 1] - preds[:, 2], dim=1)
    min_dist = torch.min(torch.stack([d1, d2, d3], dim=1), dim=1)[0]
    return torch.clamp(margin - min_dist, min=0.0) ** 2  # Margin for diversity

# ------------------------
# Training Function
# ------------------------
def train_stage1():
    device = CONFIG["device"]
    train_loader, val_loader = get_dataloaders()

    # Model
    model = Stage1Model().to(device)

    # Optimizer & Scheduler
    optimizer = optim.Adam(model.parameters(), lr=CONFIG["lr"], weight_decay=CONFIG["weight_decay"])
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=3, gamma=0.8)  # decay every 3 epochs

    scaler = amp.GradScaler()  # For mixed precision
    best_val_loss = float("inf")

    # TensorBoard
    writer = SummaryWriter(log_dir=CONFIG["save_dir"] / "logs")
    global_step = 0

    # ------------------------
    # Training Loop
    # ------------------------
    for epoch in range(CONFIG["epochs"]):
        model.train()
        running_loss = 0.0
        start_time = time.time()

        optimizer.zero_grad()
        for i, batch in enumerate(train_loader):
            current_img = batch["current_img"].to(device)
            past_imgs = batch["past_imgs"].to(device)
            positions = batch["positions"].to(device)
            target = batch["target"].to(device)  # [B, 2] → (x, y)

            with amp.autocast():  # Mixed precision
                output = model(current_img, past_imgs, positions)  # [B, 6] -> 3 pairs
                pred_3 = output.view(-1, 3, 2)  # [B, 3, 2]
                loss_div = diversity_loss(pred_3, margin=0.05)
                target_xy = target[:, :2].unsqueeze(1).expand_as(pred_3)  # [B, 3, 2]

                losses_xy = relaxed_l2_loss(pred_3, target_xy, margin=0.03)  # [B, 3]
                gm_loss = (losses_xy.prod(dim=1)) ** (1/3)  # [B]
                loss = (target[:, 2] * (gm_loss + loss_div)).mean()

            scaler.scale(loss).backward()

            # Gradient accumulation
            if (i + 1) % CONFIG["accumulation_steps"] == 0:
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()

            running_loss += loss.item()
            global_step += 1

            if (i + 1) % CONFIG["print_freq"] == 0:
                avg_loss = running_loss / CONFIG["print_freq"]
                print(f"Epoch [{epoch+1}/{CONFIG['epochs']}], Step [{i+1}/{len(train_loader)}], Loss: {avg_loss:.4f}")
                writer.add_scalar("Loss/train_total", avg_loss, global_step)
                # writer.add_scalar("Loss/train_xy", loss_xy.mean().item(), global_step)
                # writer.add_scalar("Loss/train_vis", loss_vis.item(), global_step)
                running_loss = 0.0

        # ------------------------
        # Validation
        # ------------------------
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for batch in val_loader:
                current_img = batch["current_img"].to(device)
                past_imgs = batch["past_imgs"].to(device)
                positions = batch["positions"].to(device)
                target = batch["target"].to(device)

                with amp.autocast():
                    output = model(current_img, past_imgs, positions)  # [B, 6] -> 3 pairs
                    pred_3 = output.view(-1, 3, 2)  # [B, 3, 2]
                    loss_div = diversity_loss(pred_3, margin=0.05)
                    target_xy = target[:, :2].unsqueeze(1).expand_as(pred_3)  # [B, 3, 2]

                    losses_xy = relaxed_l2_loss(pred_3, target_xy, margin=0.03)  # [B, 3]
                    gm_loss = (losses_xy.prod(dim=1)) ** (1/3)  # [B]
                    loss = (target[:, 2] * (gm_loss + loss_div)).mean()
                    val_loss += loss.item()

        val_loss /= len(val_loader)
        print(f"Epoch {epoch+1} finished in {time.time()-start_time:.2f}s - Val Loss: {val_loss:.4f}")
        writer.add_scalar("Loss/val_total", val_loss, epoch)

        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            save_and_log_checkpoint(model, Path(CONFIG["save_dir"]) / "stage1_best.pth", writer, global_step)
            print("✅ Saved new best model.")

        scheduler.step()

    # Save last epoch
    save_and_log_checkpoint(model, Path(CONFIG["save_dir"]) / "stage1_last.pth", writer, global_step)
    writer.close()
    print("Training completed!")

if __name__ == "__main__":
    train_stage1()

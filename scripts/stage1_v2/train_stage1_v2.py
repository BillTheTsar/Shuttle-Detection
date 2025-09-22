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
from huggingface_hub import HfApi

from stage1_model_v2 import Stage1Model
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
    "print_freq": 40
}

HF_TOKEN = os.getenv("HF_TOKEN")  # set in Gradient Notebook environment
HF_REPO = "BillTheTsar/shuttle-model-checkpoints"
api = HfApi()

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
def save_and_log_checkpoint(model, path, writer, step, upload_to_hf=False):
    torch.save(model.state_dict(), path)
    if path.exists():
        size = path.stat().st_size
        print(f"✅ Saved checkpoint at {path} ({size / 1024:.2f} KB)")
        writer.add_scalar("Checkpoint/file_size_bytes", size, step)

        if upload_to_hf and HF_TOKEN:
            try:
                api.upload_file(
                    path_or_fileobj=str(path),
                    path_in_repo=path.name,
                    repo_id=HF_REPO,
                    repo_type="model",
                    token=HF_TOKEN
                )
                print(f"📤 Uploaded {path.name} to Hugging Face Hub.")
            except Exception as e:
                print(f"⚠️ Failed to upload to Hugging Face: {e}")
    else:
        print(f"❌ Failed to save checkpoint at {path}")

# Loss functions
def relaxed_l2_loss(pred_xy, target_xy, margin):
    # pred_xy, target_xy: [B, 2]
    diff = pred_xy - target_xy
    dist = torch.sqrt((diff ** 2).sum(dim=1))  # Euclidean distance per sample
    penalty = torch.clamp(dist - margin, min=0.0)  # max(0, d - margin)
    return penalty

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
                output = model(current_img, past_imgs, positions)  # [B, 2]
                pred_xy = output

                loss_xy = relaxed_l2_loss(pred_xy, target[:, :2], margin=0.03)
                target_vis = target[:, 2]

                loss = (target_vis * loss_xy).mean()

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
                writer.add_scalar("Loss/train_xy", loss_xy.mean().item(), global_step)
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
                    output = model(current_img, past_imgs, positions)  # [B, 3]
                    pred_xy = output

                    loss_xy = relaxed_l2_loss(pred_xy, target[:, :2], margin=0.03)
                    target_vis = target[:, 2]

                    loss = (target_vis * loss_xy).mean()
                    val_loss += loss.item()

        val_loss /= len(val_loader)
        print(f"Epoch {epoch+1} finished in {time.time()-start_time:.2f}s - Val Loss: {val_loss:.4f}")
        writer.add_scalar("Loss/val_total", val_loss, epoch)

        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            save_and_log_checkpoint(model, Path(CONFIG["save_dir"]) / "stage1_v2_best.pth", writer, global_step, upload_to_hf=True)
            print("✅ Saved new best model.")

        scheduler.step()

    # Save last epoch
    save_and_log_checkpoint(model, Path(CONFIG["save_dir"]) / "stage1_v2_last.pth", writer, global_step, upload_to_hf=True)
    writer.close()
    print("Training completed!")

if __name__ == "__main__":
    train_stage1()

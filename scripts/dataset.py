import torch
from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image
import random
import json
import os
import numpy as np
import pandas as pd
from pathlib import Path
import torchvision.transforms.functional as F


# =========================
# Helper Functions
# =========================

def apply_jitter(frames, color_jitter, jitter_enabled=True):
    """
    Apply SAME color jitter to all frames by sampling one set of params.
    """
    if jitter_enabled:
        brightness = random.uniform(*color_jitter.brightness) if color_jitter.brightness else 1
        contrast = random.uniform(*color_jitter.contrast) if color_jitter.contrast else 1
        saturation = random.uniform(*color_jitter.saturation) if color_jitter.saturation else 1
        hue = random.uniform(*color_jitter.hue) if color_jitter.hue else 0

        jittered = []
        for img in frames:
            img = F.adjust_brightness(img, brightness)
            img = F.adjust_contrast(img, contrast)
            img = F.adjust_saturation(img, saturation)
            img = F.adjust_hue(img, hue)
            jittered.append(img)
        return jittered
    else:
        return frames

def apply_noise(frames, noise_std=0.01):
    """
    Apply different Gaussian noise per frame.
    Takes in a list of images and returns a list of tensors.
    """
    noisy_frames = []
    for img in frames:
        img_tensor = transforms.ToTensor()(img)
        noise_std = np.random.uniform(low=0, high=noise_std)
        noise = torch.randn_like(img_tensor) * noise_std
        img_tensor = torch.clamp(img_tensor + noise, 0, 1)
        noisy_frames.append(img_tensor)
    return noisy_frames

def generate_crop_box(gt_pos, gt_vis, img_w, img_h, crop_size, train_mode=True):
    """
    Compute crop box for Stage 2, ensuring the crop stays fully inside the image.
    Returns (x_min, y_min, x_max, y_max) as ints.

    Rules:
      - If GT is visible: GT center + random offset, then clamp center to valid range.
      - If GT is not visible: sample a random center uniformly from the valid range.
      - If train_mode is False: use gt_pos directly, then clamp to valid range.
    """
    half = crop_size / 2

    # Handle the extremely rare case where image is smaller than crop_size
    # (not your case, but safe to guard)
    valid_x_min = half
    valid_x_max = max(half, img_w - half)
    valid_y_min = half
    valid_y_max = max(half, img_h - half)

    def clamp_center(cx, cy):
        cx = max(valid_x_min, min(cx, valid_x_max))
        cy = max(valid_y_min, min(cy, valid_y_max))
        return cx, cy

    if train_mode:
        if gt_vis == 1:
            # Base on GT center in absolute pixels + your random offset
            gx = gt_pos[0] * img_w
            gy = gt_pos[1] * img_h
            offset_x = random.uniform(-0.4 * crop_size, 0.4 * crop_size)
            offset_y = random.uniform(-0.4 * crop_size, 0.4 * crop_size)
            center_x, center_y = gx + offset_x, gy + offset_y
        else:
            # If not visible, random valid center
            center_x = random.uniform(valid_x_min, valid_x_max)
            center_y = random.uniform(valid_y_min, valid_y_max)
    else:
        # Inference/val mode: use provided target center (from gt_pos), then clamp
        center_x = gt_pos[0] * img_w
        center_y = gt_pos[1] * img_h

    # Clamp to keep crop fully inside the image
    center_x, center_y = clamp_center(center_x, center_y)

    x_min = int(round(center_x - half))
    y_min = int(round(center_y - half))
    x_max = x_min + crop_size
    y_max = y_min + crop_size
    return x_min, y_min, x_max, y_max


# def generate_crop_box(gt_pos, gt_vis, img_w, img_h, crop_size, train_mode=True):
#     """
#     Compute crop box for Stage 2.
#     train_mode=True: uses GT + randomness.
#     train_mode=False: uses inferred_target directly (passed as gt_pos here).
#     """
#     if train_mode:
#         if gt_vis == 1:
#             cx, cy = gt_pos
#             offset_x = random.uniform(-0.4 * crop_size, 0.4 * crop_size)
#             offset_y = random.uniform(-0.4 * crop_size, 0.4 * crop_size)
#             center_x = cx * img_w + offset_x
#             center_y = cy * img_h + offset_y
#         else:  # gt_vis == 0 → random crop
#             center_x = random.uniform(0, img_w)
#             center_y = random.uniform(0, img_h)
#     else:
#         center_x, center_y = gt_pos[0] * img_w, gt_pos[1] * img_h
#
#     x_min = int(center_x - crop_size / 2)
#     y_min = int(center_y - crop_size / 2)
#     x_max = x_min + crop_size
#     y_max = y_min + crop_size
#     return x_min, y_min, x_max, y_max

def crop_with_padding(img, x_min, y_min, x_max, y_max, crop_size):
    # Guaranteed in-bounds after clamping; do a straight crop
    return img.crop((x_min, y_min, x_max, y_max))

# def crop_with_padding(img, x_min, y_min, x_max, y_max, crop_size):
#     """
#     Crop with black padding if crop exceeds image bounds.
#     Returns a square crop of size crop_size.
#     """
#     img_w, img_h = img.size
#     crop = Image.new("RGB", (crop_size, crop_size), (0, 0, 0))
#
#     # Calculate overlap
#     overlap_x_min = max(0, x_min)
#     overlap_y_min = max(0, y_min)
#     overlap_x_max = min(img_w, x_max)
#     overlap_y_max = min(img_h, y_max)
#
#     if overlap_x_min < overlap_x_max and overlap_y_min < overlap_y_max:
#         img_crop = img.crop((overlap_x_min, overlap_y_min, overlap_x_max, overlap_y_max))
#         paste_x = overlap_x_min - x_min
#         paste_y = overlap_y_min - y_min
#         crop.paste(img_crop, (paste_x, paste_y))
#     return crop

def adjust_positions_for_crop(positions, x_min, y_min, crop_size, img_w, img_h):
    """
    Normalize positions relative to crop.
    If point is outside crop, visibility = 0.
    The convention is that we pass in positions in the order of -30, -29, ... , -2, -1.
    Thus, we do not need to reverse the order again
    """
    adjusted = []
    for (x, y, vis) in positions:
        if vis < 0.5: # If the shuttle is not visible
            adjusted.append((0, 0, 0))
            continue
        abs_x, abs_y = x * img_w, y * img_h
        if x_min <= abs_x < x_min + crop_size and y_min <= abs_y < y_min + crop_size: # Visible and in crop
            rel_x = (abs_x - x_min) / crop_size
            rel_y = (abs_y - y_min) / crop_size
            adjusted.append((rel_x, rel_y, vis))
        else:
            adjusted.append((0, 0, 0)) # Visible but not in crop
    return adjusted


def perturb_positions(positions, max_perturb=0.02):
    """
    Perturbs visible (x, y) positions in a list or NumPy array.

    Args:
        positions: list of [x, y, vis] or NumPy array of shape [T, 3]
        max_perturb: max change in normalized coords (±)

    Returns:
        np.ndarray of shape [T, 3] with perturbed positions
    """
    positions = np.array(positions, dtype=np.float32).copy()  # Make sure it's a NumPy array

    for i in range(len(positions)):
        x, y, vis = positions[i]
        if vis >= 0.5:
            dx = np.random.uniform(-max_perturb, max_perturb)
            dy = np.random.uniform(-max_perturb, max_perturb)
            new_x = np.clip(x + dx, 0.0, 1.0)
            new_y = np.clip(y + dy, 0.0, 1.0)
            positions[i][0] = new_x
            positions[i][1] = new_y
            # visibility stays unchanged

    return positions


# =========================
# Stage 1 Dataset
# =========================

class Stage1Dataset(Dataset):
    def __init__(self, root_dir, index_file, split="train", train_mode = True, apply_flip=True, jitter_enabled=True, noise_std=0.01, position_perturbation=0.02):
        self.root_dir = Path(root_dir)
        with open(index_file, 'r') as f:
            index_data = json.load(f)
        self.sample_dirs = index_data[split]
        self.train_mode = train_mode
        self.apply_flip = apply_flip if train_mode else False # Do not flip unless training
        self.color_jitter = transforms.ColorJitter(0.3, 0.2, 0.2, 0.05)
        self.noise_std = noise_std if train_mode else 0.0 # Disable noise in inference
        self.jitter_enabled = jitter_enabled if train_mode else False  # Disable jitter in inference
        self.position_perturbation = position_perturbation
        self.resize = transforms.Resize((300, 300))

    def __len__(self):
        return len(self.sample_dirs)

    def __getitem__(self, idx):
        sample_path = self.root_dir / self.sample_dirs[idx].replace("\\", "/")
        current_img = Image.open(sample_path / "frame.jpg").convert("RGB")
        past_imgs = [Image.open(sample_path / f"frame-{i}.jpg").convert("RGB") for i in range(3, 0, -1)]

        jittered_imgs = apply_jitter([current_img] + past_imgs, self.color_jitter, self.jitter_enabled)

        resized_current = self.resize(jittered_imgs[0])
        resized_past = [self.resize(img) for img in jittered_imgs[1:]]
        current_tensor = apply_noise([resized_current], self.noise_std)[0]
        past_tensor = torch.stack(apply_noise(resized_past, self.noise_std))

        positions = pd.read_csv(sample_path / "positions.csv").values[::-1].copy()
        if self.train_mode:
            positions = perturb_positions(positions, self.position_perturbation)
        positions_tensor = torch.tensor(positions, dtype=torch.float32)

        target = pd.read_csv(sample_path / "target.csv").iloc[0]
        target_tensor = torch.tensor([target.shuttle_x, target.shuttle_y, target.shuttle_visibility], dtype=torch.float32)

        # Horizontal flip
        if self.apply_flip and random.random() < 0.5:
            current_tensor = torch.flip(current_tensor, dims=[2])
            past_tensor = torch.flip(past_tensor, dims=[3])
            positions_tensor[:, 0] = 1.0 - positions_tensor[:, 0]
            target_tensor[0] = 1.0 - target_tensor[0]

        return {
            "current_img": current_tensor,
            "past_imgs": past_tensor,
            "positions": positions_tensor,
            "target": target_tensor,
            # "path": sample_path,
        }


# ==============================
# Stage 2 Dataset
# ==============================

class Stage2Dataset(Dataset):
    def __init__(self, root_dir, index_file, split="train", train_mode=True, inferred_target=None,
                 apply_flip=True, jitter_enabled=True, noise_std=0.01, position_perturbation=0.02):
        self.root_dir = Path(root_dir)
        with open(index_file, 'r') as f:
            index_data = json.load(f)
        self.sample_dirs = index_data[split]
        self.train_mode = train_mode
        self.inferred_target = inferred_target
        self.apply_flip = apply_flip if train_mode else False
        self.color_jitter = transforms.ColorJitter(0.3, 0.2, 0.2, 0.05)
        self.noise_std = noise_std if train_mode else 0.0 # Disable noise in inference
        self.jitter_enabled = jitter_enabled if train_mode else False  # Disable jitter in inference
        self.position_perturbation = position_perturbation
        self.final_resize = transforms.Resize((224, 224))

    def __len__(self):
        return len(self.sample_dirs)

    def __getitem__(self, idx):
        sample_path = self.root_dir / self.sample_dirs[idx].replace("\\", "/")
        current_img = Image.open(sample_path / "frame.jpg").convert("RGB")
        past_imgs = [Image.open(sample_path / f"frame-{i}.jpg").convert("RGB") for i in range(3, 0, -1)]

        positions = pd.read_csv(sample_path / "positions.csv").values[::-1].copy()
        if self.train_mode:
            target = pd.read_csv(sample_path / "target.csv").iloc[0]
            gt_x, gt_y, gt_vis = target.shuttle_x, target.shuttle_y, target.shuttle_visibility
            positions = perturb_positions(positions, self.position_perturbation)
        else:
            if self.inferred_target is None:
                target = pd.read_csv(sample_path / "target.csv").iloc[0]
                gt_x, gt_y, gt_vis = target.shuttle_x, target.shuttle_y, target.shuttle_visibility
                # raise ValueError("inferred_target must be provided for inference mode")
            else:
                gt_x, gt_y, gt_vis = self.inferred_target


        jittered_imgs = apply_jitter([current_img] + past_imgs, self.color_jitter, self.jitter_enabled)
        current_img, past_imgs = jittered_imgs[0], jittered_imgs[1:]

        img_w, img_h = current_img.size
        crop_size = 224

        x_min, y_min, x_max, y_max = generate_crop_box((gt_x, gt_y), gt_vis, img_w, img_h, crop_size, train_mode=self.train_mode)
        cropped_current = crop_with_padding(current_img, x_min, y_min, x_max, y_max, crop_size)
        cropped_past = [crop_with_padding(img, x_min, y_min, x_max, y_max, crop_size) for img in past_imgs]

        resized_current = self.final_resize(cropped_current)
        resized_past = [self.final_resize(img) for img in cropped_past]

        current_tensor = apply_noise([resized_current], self.noise_std)[0]
        past_tensor = torch.stack(apply_noise(resized_past, self.noise_std))

        adjusted_positions = adjust_positions_for_crop(positions, x_min, y_min, crop_size, img_w, img_h)
        positions_tensor = torch.tensor(adjusted_positions, dtype=torch.float32)
        adjusted_target = adjust_positions_for_crop([[gt_x, gt_y, gt_vis]], x_min, y_min, crop_size, img_w, img_h)[0]
        target_tensor = torch.tensor(adjusted_target, dtype=torch.float32)

        # Horizontal flip
        if self.apply_flip and random.random() < 0.5:
            current_tensor = torch.flip(current_tensor, dims=[2])
            past_tensor = torch.flip(past_tensor, dims=[3])
            positions_tensor[:, 0] = 1.0 - positions_tensor[:, 0]
            target_tensor[0] = 1.0 - target_tensor[0]

        return {
            "current_crop": current_tensor,
            "past_crops": past_tensor,
            "positions": positions_tensor,
            "target": target_tensor,
            "path": sample_path
        }
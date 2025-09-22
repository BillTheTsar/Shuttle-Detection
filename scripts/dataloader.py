from torch.utils.data import DataLoader
from .dataset import Stage1Dataset, Stage2Dataset
import torch

# ---------- Collate Function ----------
def collate_fn(batch):
    out = {}
    for k in batch[0].keys():
        v0 = batch[0][k]
        if torch.is_tensor(v0):
            out[k] = torch.stack([b[k] for b in batch])
        else:
            out[k] = [b[k] for b in batch]
    return out
    # keys = batch[0].keys()
    # return {k: torch.stack([item[k] for item in batch]) for k in keys}

# ---------- Example Stage 1 Loader ----------
def get_stage1_loader(root_dir, index_file, split="train", batch_size=4, num_workers=0, train_mode=True, apply_flip=True,
                      jitter_enabled=True, noise_std=0.01, position_perturbation=0.02):
    dataset = Stage1Dataset(
        root_dir=root_dir,
        index_file=index_file,
        split=split,
        train_mode=train_mode,
        apply_flip=apply_flip,
        jitter_enabled=jitter_enabled,
        noise_std=noise_std,
        position_perturbation=position_perturbation,
    )

    loader = DataLoader(
        dataset, batch_size=batch_size, shuffle=True,
        num_workers=num_workers, pin_memory=True,
        collate_fn=collate_fn
    )
    return loader

# ---------- Example Stage 2 Loader ----------
def get_stage2_loader(root_dir, index_file, split="train", batch_size=4, num_workers=0, train_mode=True, inferred_target=None,
                      apply_flip=True, jitter_enabled=True, noise_std=0.01, position_perturbation=0.02):
    dataset = Stage2Dataset(
        root_dir=root_dir,
        index_file=index_file,
        split=split,
        train_mode=train_mode,
        inferred_target=inferred_target,
        apply_flip=apply_flip,
        jitter_enabled=jitter_enabled,
        noise_std=noise_std,
        position_perturbation=position_perturbation
    )

    loader = DataLoader(
        dataset, batch_size=batch_size, shuffle=True,
        num_workers=num_workers, pin_memory=True,
        collate_fn=collate_fn
    )
    return loader

# ---------- Quick Test ----------
# if __name__ == "__main__":
#     root = "F:/GitHub/Shuttle Detection Model/sequences"
#     index = "F:/GitHub/Shuttle Detection Model/configs/dataset_index.json"
#
#     # loader = get_stage1_loader(root, index, batch_size=4, train_mode=True)
#     loader = get_stage2_loader(root, index, batch_size=4, train_mode=True)
#     batch = next(iter(loader))
#
#     # print("Stage 1 Batch Shapes:")
#     # print(f"Current Frame: {batch['current_img'].shape}")    # [B, 3, 300, 300]
#     # print(f"Past Frames: {batch['past_imgs'].shape}")       # [B, 3, 3, 300, 300]
#     # print(f"Positions: {batch['positions'].shape}")         # [B, 30, 3]
#     # print(f"Target: {batch['target'].shape}")               # [B, 3]
#
#     print("Stage 2 Batch Shapes:")
#     print(f"Current Frame: {batch['current_crop'].shape}")    # [B, 3, 224, 224]
#     print(f"Past Frames: {batch['past_crops'].shape}")       # [B, 3, 3, 224, 224]
#     print(f"Positions: {batch['positions'].shape}")         # [B, 30, 3]
#     print(f"Target: {batch['target'].shape}")               # [B, 3]

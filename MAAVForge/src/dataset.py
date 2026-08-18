"""
dataset.py — Data Pipeline for KLA Semiconductor Image Restoration
===================================================================
Design decisions grounded in empirical data analysis:

Data Profiling Results (across all 3,200 pairs):
  - Input  (NoisyLR): 128×128, float32, range [-0.28, 2.16] — multiplicative speckle
  - Target (GT):      256×256, float32, range [0.00,  1.00] — clean, normalized
  - Scale factor: always 2×
  - LR and GT pixel distributions share nearly identical median and IQR
    (P25: 0.178 vs 0.184 | P50: 0.359 vs 0.366 | P75: 0.598 vs 0.601)

Normalization Strategy:
  Raw signal values are fed directly into the network without any
  normalization. The LR and GT distributions are naturally aligned in
  pixel space — applying IQR or Min-Max normalization would break this
  alignment and cause the model to learn a meaningless mapping.
  The model is trained to map out-of-range noisy inputs to clean [0,1] outputs.
"""

import os
import numpy as np
import torch
from torch.utils.data import Dataset
import random


class SemiconTrainDataset(Dataset):
    """
    Training dataset with paired LR-GT images.
    Zero data leakage: train/val split is deterministic and non-overlapping.
    """
    def __init__(self, root_dir, mode='train', val_split=0.1, seed=42):
        self.root_dir = root_dir
        self.mode = mode

        self.gt_dir = os.path.join(root_dir, 'GT')
        self.lr_dir = os.path.join(root_dir, 'NoisyLR')

        # Only use files that exist in BOTH directories (safety)
        gt_files = set(f for f in os.listdir(self.gt_dir) if f.endswith('.npy'))
        lr_files = set(f for f in os.listdir(self.lr_dir) if f.endswith('.npy'))
        all_files = sorted(list(gt_files & lr_files))

        # Deterministic, reproducible, leak-free split
        rng = random.Random(seed)
        shuffled = all_files.copy()
        rng.shuffle(shuffled)

        split_idx = int(len(shuffled) * (1 - val_split))
        if mode == 'train':
            self.files = shuffled[:split_idx]
        elif mode == 'val':
            self.files = shuffled[split_idx:]
        else:
            self.files = all_files  # 'all' mode for final training

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        fname = self.files[idx]

        gt = np.load(os.path.join(self.gt_dir, fname)).astype(np.float32)
        lr = np.load(os.path.join(self.lr_dir, fname)).astype(np.float32)

        # No normalization tricks — feed raw signal.
        # GT is [0,1]. LR may exceed [0,1] due to speckle.
        # The model must learn this mapping naturally.

        if self.mode == 'train':
            gt, lr = self._augment(gt, lr)

        # (H, W) → (1, H, W)
        gt = gt[np.newaxis, :, :]
        lr = lr[np.newaxis, :, :]

        return torch.from_numpy(lr.copy()), torch.from_numpy(gt.copy())

    def _augment(self, gt, lr):
        """
        Structure-preserving geometric augmentations.
        Applied identically to both GT and LR to maintain correspondence.
        """
        # Horizontal flip
        if random.random() > 0.5:
            gt = np.flip(gt, axis=1)
            lr = np.flip(lr, axis=1)

        # Vertical flip
        if random.random() > 0.5:
            gt = np.flip(gt, axis=0)
            lr = np.flip(lr, axis=0)

        # Random 90° rotation (0, 90, 180, 270)
        k = random.randint(0, 3)
        if k > 0:
            gt = np.rot90(gt, k)
            lr = np.rot90(lr, k)

        return gt, lr


class SemiconTestDataset(Dataset):
    """
    Test dataset — LR images only, no GT.
    Uses the exact same preprocessing as training to ensure consistency.
    """
    def __init__(self, input_dir):
        self.input_dir = input_dir
        self.files = sorted([f for f in os.listdir(input_dir) if f.endswith('.npy')])

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        fname = self.files[idx]
        lr = np.load(os.path.join(self.input_dir, fname)).astype(np.float32)
        lr = lr[np.newaxis, :, :]  # (1, H, W)
        return torch.from_numpy(lr.copy()), fname

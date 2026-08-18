"""
loss.py — Composite Loss Function for Semiconductor Image Restoration
======================================================================
Three components, each targeting a different failure mode:

1. Charbonnier Loss (spatial domain):
   - Robust L1-like loss, smooth at zero (unlike raw L1)
   - Handles outlier pixels from speckle without exploding gradients
   - Primary driver of pixel-level accuracy

2. SSIM Loss (structural domain):
   - Directly optimizes the SSIM metric we are evaluated on
   - Measures luminance, contrast, and structural similarity
   - Prevents the model from producing pixel-accurate but structurally wrong images

3. Focal Frequency Loss (frequency domain):
   - Operates in 2D Fourier space
   - Speckle noise has a chaotic high-frequency signature
   - This loss teaches the model to suppress noise frequencies while
     preserving legitimate high-frequency detail (edges, fine structures)
   - The "focal" part: dynamically weights hard-to-reconstruct frequencies
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class CharbonnierLoss(nn.Module):
    """Smooth approximation of L1 loss: sqrt((x-y)^2 + eps^2)"""
    def __init__(self, eps=1e-3):
        super().__init__()
        self.eps2 = eps * eps

    def forward(self, pred, target):
        diff = pred - target
        return torch.mean(torch.sqrt(diff * diff + self.eps2))


class SSIMLoss(nn.Module):
    """
    Differentiable SSIM loss.
    Returns 1 - SSIM so we can minimize it.
    """
    def __init__(self, window_size=11):
        super().__init__()
        self.window_size = window_size
        self.register_buffer('window', self._create_window(window_size, 1))

    @staticmethod
    def _gaussian(window_size, sigma=1.5):
        coords = torch.arange(window_size, dtype=torch.float32) - window_size // 2
        g = torch.exp(-coords ** 2 / (2 * sigma ** 2))
        return g / g.sum()

    def _create_window(self, window_size, channel):
        g = self._gaussian(window_size)
        window = g.unsqueeze(1) * g.unsqueeze(0)  # outer product → 2D
        return window.unsqueeze(0).unsqueeze(0).expand(channel, 1, -1, -1).contiguous()

    def forward(self, pred, target):
        C1, C2 = 0.01 ** 2, 0.03 ** 2
        pad = self.window_size // 2
        ch = pred.size(1)

        # Ensure window is on correct device and dtype
        window = self.window.to(pred.device, pred.dtype)
        if ch != window.size(0):
            window = window.expand(ch, -1, -1, -1)

        mu1 = F.conv2d(pred, window, padding=pad, groups=ch)
        mu2 = F.conv2d(target, window, padding=pad, groups=ch)

        mu1_sq, mu2_sq, mu12 = mu1 ** 2, mu2 ** 2, mu1 * mu2

        sigma1_sq = F.conv2d(pred * pred, window, padding=pad, groups=ch) - mu1_sq
        sigma2_sq = F.conv2d(target * target, window, padding=pad, groups=ch) - mu2_sq
        sigma12 = F.conv2d(pred * target, window, padding=pad, groups=ch) - mu12

        ssim_map = ((2 * mu12 + C1) * (2 * sigma12 + C2)) / \
                   ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))

        return 1.0 - ssim_map.mean()


class FocalFrequencyLoss(nn.Module):
    """
    Compares predicted and target images in the Fourier frequency domain.
    Dynamically focuses on frequencies where the model is struggling most.
    """
    def __init__(self, alpha=1.0):
        super().__init__()
        self.alpha = alpha

    def forward(self, pred, target):
        # 2D FFT
        pred_fft = torch.fft.rfft2(pred, norm='ortho')
        target_fft = torch.fft.rfft2(target, norm='ortho')

        # Complex difference → magnitude
        diff = torch.abs(pred_fft - target_fft)

        # Focal weighting: focus on hard frequencies (where error is large)
        weight = diff.detach() ** self.alpha
        loss = (weight * diff).mean()

        return loss


class CombinedLoss(nn.Module):
    """
    Unified loss combining all three components.
    Weights chosen based on loss magnitude balancing.
    """
    def __init__(self, char_w=1.0, ssim_w=0.1, freq_w=0.05):
        super().__init__()
        self.charbonnier = CharbonnierLoss()
        self.ssim = SSIMLoss()
        self.ffl = FocalFrequencyLoss()

        self.char_w = char_w
        self.ssim_w = ssim_w
        self.freq_w = freq_w

    def forward(self, pred, target):
        l_char = self.charbonnier(pred, target)
        l_ssim = self.ssim(pred, target)
        l_ffl = self.ffl(pred, target)

        total = self.char_w * l_char + self.ssim_w * l_ssim + self.freq_w * l_ffl

        return total, {
            'char': l_char.item(),
            'ssim': l_ssim.item(),
            'ffl': l_ffl.item(),
            'total': total.item(),
        }

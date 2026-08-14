"""
model.py — NAFNet-SR Architecture for Semiconductor Image Restoration
======================================================================
Architecture: NAFNet (Nonlinear Activation Free Network) + PixelShuffle(2x)

Design rationale:
- All computation happens at input resolution (128x128)
- PixelShuffle at the end does 128→256 (2x upscale, matching our data)
- NAFBlocks use SimpleGate (element-wise multiply) instead of GELU/ReLU
  → eliminates activation function overhead → fastest possible inference
- Channel Attention via global avg pooling → learns which features are
  signal vs noise without spatial attention overhead
- Global residual via bicubic upscale → model only learns the high-freq
  residual (denoising + detail recovery), not the entire reconstruction

Parameter budget: ~2M params → fast training, fast inference, no overfitting
on 3200 samples.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class LayerNorm2d(nn.Module):
    """Channel-wise LayerNorm for conv features."""
    def __init__(self, channels, eps=1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(channels))
        self.bias = nn.Parameter(torch.zeros(channels))
        self.eps = eps

    def forward(self, x):
        # x: (B, C, H, W)
        u = x.mean(1, keepdim=True)
        s = (x - u).pow(2).mean(1, keepdim=True)
        x = (x - u) / torch.sqrt(s + self.eps)
        return self.weight[None, :, None, None] * x + self.bias[None, :, None, None]


class SimpleGate(nn.Module):
    """
    Replaces activation functions (GELU, ReLU, etc.).
    Splits channels into two halves and multiplies them element-wise.
    This is non-linear (product of two learned quantities) but uses
    zero special math operations → maximum hardware throughput.
    """
    def forward(self, x):
        x1, x2 = x.chunk(2, dim=1)
        return x1 * x2


class NAFBlock(nn.Module):
    """
    Core building block of NAFNet.
    Flow: Norm → 1x1 Conv (expand) → 3x3 DWConv → SimpleGate → 
          Channel Attention → 1x1 Conv (compress) → Residual
    Then: Norm → 1x1 Conv (expand) → SimpleGate → 1x1 Conv → Residual
    """
    def __init__(self, channels, dw_expand=2, ffn_expand=2):
        super().__init__()
        dw_ch = channels * dw_expand

        # Spatial mixing branch
        self.norm1 = LayerNorm2d(channels)
        self.conv1 = nn.Conv2d(channels, dw_ch, 1)          # expand
        self.conv2 = nn.Conv2d(dw_ch, dw_ch, 3, 1, 1, groups=dw_ch)  # depthwise
        self.sg1 = SimpleGate()                               # dw_ch → dw_ch//2
        self.sca = nn.Sequential(                             # channel attention
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(dw_ch // 2, dw_ch // 2, 1),
        )
        self.conv3 = nn.Conv2d(dw_ch // 2, channels, 1)     # compress

        # FFN branch
        ffn_ch = channels * ffn_expand
        self.norm2 = LayerNorm2d(channels)
        self.conv4 = nn.Conv2d(channels, ffn_ch, 1)
        self.sg2 = SimpleGate()                               # ffn_ch → ffn_ch//2
        self.conv5 = nn.Conv2d(ffn_ch // 2, channels, 1)

        # Learnable residual scaling (initialized at 0 for stable training)
        self.beta = nn.Parameter(torch.zeros(1, channels, 1, 1))
        self.gamma = nn.Parameter(torch.zeros(1, channels, 1, 1))

    def forward(self, x):
        # Spatial mixing
        y = self.norm1(x)
        y = self.conv1(y)
        y = self.conv2(y)
        y = self.sg1(y)
        y = y * self.sca(y)
        y = self.conv3(y)
        x = x + y * self.beta

        # FFN
        y = self.norm2(x)
        y = self.conv4(y)
        y = self.sg2(y)
        y = self.conv5(y)
        x = x + y * self.gamma

        return x


class NAFNetSR(nn.Module):
    """
    Full restoration network.

    Args:
        img_channel: Input channels (1 for grayscale)
        width: Base channel width
        enc_blk_nums: Number of NAFBlocks per encoder stage
        middle_blk_num: Number of NAFBlocks in bottleneck
        dec_blk_nums: Number of NAFBlocks per decoder stage
        upscale: Super-resolution factor (2 for our data)
    """
    def __init__(self, img_channel=1, width=32, enc_blk_nums=None,
                 middle_blk_num=6, dec_blk_nums=None, upscale=2):
        super().__init__()
        if enc_blk_nums is None:
            enc_blk_nums = [2, 2, 4]
        if dec_blk_nums is None:
            dec_blk_nums = [4, 2, 2]

        self.upscale = upscale

        # Input projection
        self.intro = nn.Conv2d(img_channel, width, 3, 1, 1)

        # Encoder
        self.encoders = nn.ModuleList()
        self.downs = nn.ModuleList()
        ch = width
        for num_blks in enc_blk_nums:
            self.encoders.append(nn.Sequential(*[NAFBlock(ch) for _ in range(num_blks)]))
            self.downs.append(nn.Conv2d(ch, ch * 2, 2, 2))  # stride-2 downsample
            ch *= 2

        # Bottleneck
        self.middle = nn.Sequential(*[NAFBlock(ch) for _ in range(middle_blk_num)])

        # Decoder
        self.decoders = nn.ModuleList()
        self.ups = nn.ModuleList()
        for num_blks in dec_blk_nums:
            self.ups.append(nn.Sequential(
                nn.Conv2d(ch, ch * 2, 1),  # expand channels for PixelShuffle
                nn.PixelShuffle(2),         # spatial upsample 2x, channels /4
            ))
            ch //= 2
            self.decoders.append(nn.Sequential(*[NAFBlock(ch) for _ in range(num_blks)]))

        # Output: project to upscale^2 channels, then PixelShuffle for SR
        self.ending = nn.Sequential(
            nn.Conv2d(ch, img_channel * (upscale ** 2), 3, 1, 1),
            nn.PixelShuffle(upscale),
        )

        # Padder for encoder compatibility
        self.padder_size = 2 ** len(enc_blk_nums)

    def forward(self, x):
        B, C, H, W = x.shape
        x = self._check_and_pad(x)

        # Global residual: bicubic upscale of input
        # Model only needs to learn the HIGH-FREQUENCY RESIDUAL
        base = F.interpolate(x, scale_factor=self.upscale, mode='bicubic', align_corners=False)

        x = self.intro(x)

        # Encoder with skip connections
        skips = []
        for enc, down in zip(self.encoders, self.downs):
            x = enc(x)
            skips.append(x)
            x = down(x)

        # Bottleneck
        x = self.middle(x)

        # Decoder with skip connections
        for dec, up, skip in zip(self.decoders, self.ups, reversed(skips)):
            x = up(x)
            x = x + skip  # additive skip connection
            x = dec(x)

        # Output projection + PixelShuffle upscale
        x = self.ending(x)

        # Add global residual
        x = x + base

        # Remove padding
        return x[:, :, :H * self.upscale, :W * self.upscale]

    def _check_and_pad(self, x):
        """Pad input to be divisible by padder_size."""
        _, _, h, w = x.shape
        pad_h = (self.padder_size - h % self.padder_size) % self.padder_size
        pad_w = (self.padder_size - w % self.padder_size) % self.padder_size
        if pad_h > 0 or pad_w > 0:
            x = F.pad(x, (0, pad_w, 0, pad_h), mode='reflect')
        return x

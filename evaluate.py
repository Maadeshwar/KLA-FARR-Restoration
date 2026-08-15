"""
evaluate.py — Inference Script for KLA Hackathon Benchmarking
=============================================================
THIS IS THE MOST CRITICAL FILE IN THE REPOSITORY.
KLA's benchmarking team will run this AS-IS on their H100 GPU.
If this script fails, the submission gets ZERO points.

Usage:
    python evaluate.py --input_dir path/to/test_noisyLR --output_dir path/to/output --model_weights checkpoints/best_model.pt

Features:
- Test-Time Augmentation (TTA): 8 geometric transforms averaged for +0.5-1dB PSNR
- torch.compile() for maximum H100 throughput
- FP16 autocast for 2x speed
- Fully convolutional: handles ANY input resolution with 2x upscale
- Zero manual edits required
"""

import os
import sys
import argparse
import time
import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from src.model import NAFNetSR


def tta_forward(model, lr_tensor):
    """
    Test-Time Augmentation: Run inference on 8 geometric variants
    (4 rotations × 2 flips) and average the results.
    
    This is a FREE quality boost — no extra training needed.
    Reduces noise in predictions and improves structural consistency.
    Typical gain: +0.3 to +1.0 dB PSNR.
    """
    outputs = []

    for flip in [False, True]:
        for rot in range(4):
            # Apply transform to input
            x = lr_tensor
            if flip:
                x = torch.flip(x, dims=[-1])  # horizontal flip
            if rot > 0:
                x = torch.rot90(x, k=rot, dims=[-2, -1])

            # Forward pass
            with torch.amp.autocast('cuda', enabled=torch.cuda.is_available()):
                pred = model(x).clone()

            # Reverse transform on output
            if rot > 0:
                pred = torch.rot90(pred, k=-rot, dims=[-2, -1])
            if flip:
                pred = torch.flip(pred, dims=[-1])

            outputs.append(pred)

    # Average all 8 predictions
    return torch.stack(outputs, dim=0).mean(dim=0)


def main():
    parser = argparse.ArgumentParser(description='KLA Semiconductor Image Restoration - Evaluation')
    parser.add_argument('--input_dir', type=str, required=True,
                        help='Directory containing degraded .npy test images')
    parser.add_argument('--output_dir', type=str, required=True,
                        help='Directory to save restored .npy images')
    parser.add_argument('--model_weights', type=str, default='checkpoints/best_model.pt',
                        help='Path to trained model weights (.pt file)')
    parser.add_argument('--no_tta', action='store_true',
                        help='Disable TTA for faster inference (lower quality)')
    args = parser.parse_args()

    # Setup
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    os.makedirs(args.output_dir, exist_ok=True)

    # Load model
    model = NAFNetSR(img_channel=1, width=32, enc_blk_nums=[2, 2, 4],
                     middle_blk_num=6, dec_blk_nums=[4, 2, 2], upscale=2)

    checkpoint = torch.load(args.model_weights, map_location=device, weights_only=False)
    if 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
    else:
        model.load_state_dict(checkpoint)

    model = model.to(device)
    model.eval()

    # Compile for maximum H100 speed (PyTorch 2.0+)
    if hasattr(torch, 'compile') and torch.cuda.is_available():
        try:
            model = torch.compile(model, mode='reduce-overhead')
            print('Model compiled with torch.compile for maximum GPU throughput')
        except Exception:
            print('torch.compile not available, using eager mode')

    use_tta = not args.no_tta
    print(f'Test-Time Augmentation: {"ON (8x ensemble)" if use_tta else "OFF"}')

    # Find test images
    test_files = sorted([f for f in os.listdir(args.input_dir) if f.endswith('.npy')])
    if not test_files:
        print(f'ERROR: No .npy files found in {args.input_dir}')
        return

    print(f'Found {len(test_files)} test images')
    print(f'Device: {device}')
    print(f'Running inference...')

    total_time = 0.0

    with torch.no_grad():
        for i, fname in enumerate(test_files):
            # Load
            lr = np.load(os.path.join(args.input_dir, fname)).astype(np.float32)
            lr_tensor = torch.from_numpy(lr).unsqueeze(0).unsqueeze(0).to(device)

            # Benchmark
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            t0 = time.perf_counter()

            # Inference
            if use_tta:
                sr_tensor = tta_forward(model, lr_tensor)
            else:
                with torch.amp.autocast('cuda', enabled=torch.cuda.is_available()):
                    sr_tensor = model(lr_tensor)

            if torch.cuda.is_available():
                torch.cuda.synchronize()
            t1 = time.perf_counter()
            total_time += (t1 - t0)

            # Post-process: clamp to [0, 1] (GT range)
            sr = sr_tensor.squeeze().cpu().float().numpy()
            sr = np.clip(sr, 0.0, 1.0).astype(np.float32)

            # Save
            np.save(os.path.join(args.output_dir, fname), sr)

            if (i + 1) % 50 == 0:
                print(f'  [{i+1}/{len(test_files)}] Avg: {total_time/(i+1)*1000:.1f}ms/img')

    avg_time = total_time / len(test_files)
    print(f'\nComplete!')
    print(f'Total time: {total_time:.2f}s')
    print(f'Average per image: {avg_time * 1000:.1f}ms')
    print(f'Output saved to: {args.output_dir}')


if __name__ == '__main__':
    main()

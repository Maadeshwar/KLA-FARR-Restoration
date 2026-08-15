import os
import argparse
import numpy as np
from PIL import Image

def main():
    parser = argparse.ArgumentParser(description="Generate Before vs After images for Test Set")
    parser.add_argument('--noisy_dir', type=str, required=True, help='Path to 128x128 NoisyLR .npy files')
    parser.add_argument('--pred_dir', type=str, required=True, help='Path to 256x256 Predicted .npy files')
    parser.add_argument('--output_dir', type=str, required=True, help='Path to save side-by-side PNGs')
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    files = sorted([f for f in os.listdir(args.pred_dir) if f.endswith('.npy')])
    print(f"Found {len(files)} files to visualize.")

    for i, fname in enumerate(files):
        # 1. Load the Noisy Input (128x128)
        noisy_path = os.path.join(args.noisy_dir, fname)
        if not os.path.exists(noisy_path):
            print(f"Warning: Could not find matching noisy input for {fname}")
            continue
        
        noisy_np = np.load(noisy_path).squeeze()
        noisy_np = np.clip(noisy_np * 255.0, 0, 255).astype(np.uint8)
        noisy_img = Image.fromarray(noisy_np, mode='L')
        
        # Upscale the Noisy image to 256x256 using Nearest Neighbor so the pixels look chunky 
        # (This makes the contrast against our smooth AI image look even more impressive!)
        noisy_img = noisy_img.resize((256, 256), Image.Resampling.NEAREST)

        # 2. Load the AI Prediction (256x256)
        pred_path = os.path.join(args.pred_dir, fname)
        pred_np = np.load(pred_path).squeeze()
        pred_np = np.clip(pred_np * 255.0, 0, 255).astype(np.uint8)
        pred_img = Image.fromarray(pred_np, mode='L')

        # 3. Create a side-by-side image (512 width, 256 height)
        # We will also add a 4-pixel white border between them
        canvas = Image.new('L', (512 + 4, 256), color=255)
        canvas.paste(noisy_img, (0, 0))
        canvas.paste(pred_img, (256 + 4, 0))

        # 4. Save
        png_name = fname.replace('.npy', '.png')
        canvas.save(os.path.join(args.output_dir, png_name))

        if (i + 1) % 50 == 0:
            print(f"Processed {i + 1}/{len(files)} images...")

    print(f"Done! All PNGs saved to {args.output_dir}")

if __name__ == '__main__':
    main()

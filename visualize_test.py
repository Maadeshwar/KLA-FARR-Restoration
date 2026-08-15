import os
import argparse
import numpy as np
from PIL import Image

def main():
    parser = argparse.ArgumentParser(description="Generate Before and After PNGs for Test Set")
    parser.add_argument('--noisy_dir', type=str, required=True, help='Path to 128x128 NoisyLR .npy files')
    parser.add_argument('--pred_dir', type=str, required=True, help='Path to 256x256 Predicted .npy files')
    parser.add_argument('--output_dir', type=str, required=True, help='Base path to save the test images')
    args = parser.parse_args()

    # Create the before and after folders inside the output directory (e.g., results/test)
    before_dir = os.path.join(args.output_dir, 'before')
    after_dir = os.path.join(args.output_dir, 'after')
    os.makedirs(before_dir, exist_ok=True)
    os.makedirs(after_dir, exist_ok=True)

    files = sorted([f for f in os.listdir(args.pred_dir) if f.endswith('.npy')])
    print(f"Found {len(files)} files to visualize.")

    for i, fname in enumerate(files):
        png_name = fname.replace('.npy', '.png')

        # 1. Process and save the "Before" Image (NoisyLR)
        noisy_path = os.path.join(args.noisy_dir, fname)
        if os.path.exists(noisy_path):
            noisy_np = np.load(noisy_path).squeeze()
            noisy_np = np.clip(noisy_np * 255.0, 0, 255).astype(np.uint8)
            noisy_img = Image.fromarray(noisy_np, mode='L')
            
            # Upscale it to 256x256 so it matches the physical size of the AI output
            noisy_img = noisy_img.resize((256, 256), Image.Resampling.NEAREST)
            noisy_img.save(os.path.join(before_dir, png_name))
        else:
            print(f"Warning: Could not find matching noisy input for {fname}")

        # 2. Process and save the "After" Image (Prediction)
        pred_path = os.path.join(args.pred_dir, fname)
        pred_np = np.load(pred_path).squeeze()
        pred_np = np.clip(pred_np * 255.0, 0, 255).astype(np.uint8)
        pred_img = Image.fromarray(pred_np, mode='L')
        pred_img.save(os.path.join(after_dir, png_name))

        if (i + 1) % 50 == 0:
            print(f"Processed {i + 1}/{len(files)} images...")

    print(f"Done!")
    print(f"Before images saved to: {before_dir}")
    print(f"After images saved to:  {after_dir}")

if __name__ == '__main__':
    main()

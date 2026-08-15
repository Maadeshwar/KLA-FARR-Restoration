import os
import torch
from PIL import Image
import numpy as np
import sys
import glob

sys.path.insert(0, 'D:/Semicon')
from src.model import NAFNetSR

def main():
    input_dir = 'D:/Semicon/Chip_Test/Input'
    output_dir = 'D:/Semicon/Chip_Test/Output'
    os.makedirs(output_dir, exist_ok=True)

    # 1. Load the model
    print("Loading AI Model...")
    device = torch.device('cpu')
    model = NAFNetSR(img_channel=1, width=32, enc_blk_nums=[2, 2, 4], middle_blk_num=6, dec_blk_nums=[4, 2, 2], upscale=2)
    checkpoint = torch.load('D:/Semicon/checkpoints/best_model.pt', map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'] if 'model_state_dict' in checkpoint else checkpoint)
    model.eval()

    # 2. Find images
    image_paths = glob.glob(os.path.join(input_dir, '*.png'))
    if not image_paths:
        print(f"No PNG images found in {input_dir}")
        return

    print(f"Found {len(image_paths)} images. Processing...")

    # 3. Process each image
    for img_path in image_paths:
        fname = os.path.basename(img_path)
        print(f"Restoring {fname}...")
        
        # Load Input Image
        img = Image.open(img_path).convert('L')
        # Ensure it's 128x128 exactly as the model expects
        if img.size != (128, 128):
            img = img.resize((128, 128), Image.Resampling.LANCZOS)
            
        img_np = np.array(img).astype(np.float32) / 255.0
        
        # Prepare for model
        tensor = torch.from_numpy(img_np).unsqueeze(0).unsqueeze(0)
        
        # Run AI Inference
        with torch.no_grad():
            out = model(tensor)
            
        # Post-process Output
        out_np = out.squeeze().numpy()
        out_np = np.clip(out_np * 255.0, 0, 255).astype(np.uint8)
        pred_img = Image.fromarray(out_np)

        # Create Side-by-Side Canvas
        # Upscale noisy input to 256x256 using Nearest Neighbor to match AI output height
        noisy_upscaled = img.resize((256, 256), Image.Resampling.NEAREST)
        
        # Canvas: 512 + 4px border width, 256 height
        canvas = Image.new('L', (512 + 4, 256), color=255)
        canvas.paste(noisy_upscaled, (0, 0))
        canvas.paste(pred_img, (256 + 4, 0))

        # Save
        save_path = os.path.join(output_dir, fname)
        canvas.save(save_path)

    print(f"\nDone! All {len(image_paths)} side-by-side images saved to {output_dir}")

if __name__ == '__main__':
    main()

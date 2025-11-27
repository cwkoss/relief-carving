"""
Run depth estimation only - for GUI caching
"""
import sys
import numpy as np
from PIL import Image
import torch

def main():
    input_path = sys.argv[1] if len(sys.argv) > 1 else "gui_temp/input.png"
    output_path = sys.argv[2] if len(sys.argv) > 2 else "gui_temp/depth.npy"
    alpha_path = sys.argv[3] if len(sys.argv) > 3 else "gui_temp/alpha_mask.npy"

    print(f"Running depth estimation on: {input_path}")

    # Load image
    image_orig = Image.open(input_path)
    has_alpha = image_orig.mode in ('RGBA', 'LA') or (image_orig.mode == 'P' and 'transparency' in image_orig.info)

    # Extract alpha channel if present, otherwise create full opacity mask
    if has_alpha:
        print("Image has transparency - extracting alpha mask")
        image_rgba = image_orig.convert("RGBA")
        alpha_mask = np.array(image_rgba)[:, :, 3].astype(float) / 255.0
        print(f"Alpha mask: {(alpha_mask == 0).sum()} fully transparent pixels")
    else:
        print("No transparency detected - creating full opacity mask")
        alpha_mask = np.ones((image_orig.height, image_orig.width), dtype=np.float32)

    # Always save alpha mask for post-processing
    np.save(alpha_path, alpha_mask)
    print(f"Alpha mask saved to {alpha_path} (shape: {alpha_mask.shape})")

    image = image_orig.convert("RGB")
    print(f"Image size: {image.size}")

    # Load model
    print("Loading ZoeDepth model...")
    model = torch.hub.load(".", "ZoeD_N", source="local", pretrained=True, trust_repo=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = model.to(device)
    model.eval()
    print(f"Model loaded (device: {device})")

    # Run depth estimation
    print("Running depth estimation...")
    with torch.no_grad():
        depth = model.infer_pil(image)
    print(f"Depth range: {depth.min():.2f} to {depth.max():.2f}")

    # Save depth map
    print(f"Saving depth map to {output_path}...")
    np.save(output_path, depth)
    print("Depth estimation complete!")

if __name__ == "__main__":
    main()

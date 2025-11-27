import warnings
warnings.filterwarnings("ignore")

import torch
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import os
from pathlib import Path

print("="*60)
print("ZoeDepth Batch Processing")
print("="*60)

# Load model once
print("\nLoading ZoeDepth model...")
model = torch.hub.load(".", "ZoeD_N", source="local", pretrained=True, trust_repo=True)

# Set device
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")
model = model.to(device)
model.eval()

# Find all images in the images folder
images_dir = Path("images")
output_dir = Path("output")
output_dir.mkdir(exist_ok=True)

# Supported image formats
image_extensions = ['.jpg', '.jpeg', '.png', '.webp', '.bmp']
image_files = []
for ext in image_extensions:
    image_files.extend(images_dir.glob(f"*{ext}"))
    image_files.extend(images_dir.glob(f"*{ext.upper()}"))

print(f"\nFound {len(image_files)} images to process")
print("-"*60)

# Process each image
for idx, image_path in enumerate(image_files, 1):
    print(f"\n[{idx}/{len(image_files)}] Processing: {image_path.name}")

    try:
        # Load image
        image = Image.open(image_path).convert("RGB")
        print(f"  Image size: {image.size}")

        # Run inference
        print("  Running depth estimation...")
        with torch.no_grad():
            depth = model.infer_pil(image)

        print(f"  Depth range: {depth.min():.2f} to {depth.max():.2f}")

        # Create output filename (remove extension and add suffix)
        base_name = image_path.stem

        # Save depth map as numpy array
        np_path = output_dir / f"{base_name}_depth.npy"
        np.save(np_path, depth)
        print(f"  Saved: {np_path.name}")

        # Create side-by-side visualization
        fig, axes = plt.subplots(1, 2, figsize=(15, 5))

        # Original image
        axes[0].imshow(image)
        axes[0].set_title(f"Original: {image_path.name}")
        axes[0].axis("off")

        # Depth map
        im = axes[1].imshow(depth, cmap="inferno")
        axes[1].set_title("Depth Map")
        axes[1].axis("off")
        plt.colorbar(im, ax=axes[1], fraction=0.046, pad=0.04)

        plt.tight_layout()
        viz_path = output_dir / f"{base_name}_visualization.png"
        plt.savefig(viz_path, dpi=150, bbox_inches="tight")
        plt.close()
        print(f"  Saved: {viz_path.name}")

        # Save standalone depth map
        plt.figure(figsize=(10, 10))
        plt.imshow(depth, cmap="inferno")
        plt.axis("off")
        depth_path = output_dir / f"{base_name}_depth_map.png"
        plt.savefig(depth_path, dpi=150, bbox_inches="tight", pad_inches=0)
        plt.close()
        print(f"  Saved: {depth_path.name}")

        print(f"  ✓ Complete!")

    except Exception as e:
        print(f"  ✗ Error processing {image_path.name}: {e}")
        continue

print("\n" + "="*60)
print(f"Batch processing complete! Processed {len(image_files)} images.")
print(f"Output saved to: {output_dir.absolute()}")
print("="*60)

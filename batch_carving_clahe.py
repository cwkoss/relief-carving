import warnings
warnings.filterwarnings("ignore")

import numpy as np
from PIL import Image
import cv2
from pathlib import Path
from process_for_carving import normalize_array, enhance_with_high_freq_detail
import torch
from tqdm import tqdm

def apply_clahe_refined(depth_map, clip_limit=2.5, tile_size=16):
    """Apply CLAHE with optimized parameters"""
    depth_8bit = (depth_map * 255).astype(np.uint8)
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=(tile_size, tile_size))
    enhanced = clahe.apply(depth_8bit)
    return enhanced.astype(float) / 255.0

def apply_bilateral_filter(depth_map, d=9, sigma_color=75, sigma_space=75):
    """Bilateral filter - reduces noise while preserving edges"""
    depth_8bit = (depth_map * 255).astype(np.uint8)
    filtered = cv2.bilateralFilter(depth_8bit, d, sigma_color, sigma_space)
    return filtered.astype(float) / 255.0

def process_image_for_carving(image_path, model, device, output_dir):
    """Process single image with CLAHE + Bilateral pipeline"""

    base_name = Path(image_path).stem
    output_dir = Path(output_dir)

    print(f"\n  Processing: {base_name}")

    # Load image
    image = Image.open(image_path).convert("RGB")
    print(f"    Size: {image.size}")

    # Run depth estimation
    print(f"    Running depth estimation...")
    with torch.no_grad():
        depth = model.infer_pil(image)

    print(f"    Depth range: {depth.min():.2f} to {depth.max():.2f}")

    # Enhance with high-frequency detail (inverted depth)
    print(f"    Enhancing with high-frequency detail...")
    enhanced_base = enhance_with_high_freq_detail(image, depth, detail_strength=0.2, invert_depth=True)

    # Apply CLAHE
    print(f"    Applying CLAHE...")
    clahe_enhanced = apply_clahe_refined(enhanced_base, clip_limit=2.5, tile_size=16)

    # Apply bilateral filter
    print(f"    Applying bilateral filter...")
    final_enhanced = apply_bilateral_filter(clahe_enhanced, d=9, sigma_color=75, sigma_space=75)

    # Save outputs
    # 16-bit PNG for CNC
    enhanced_16bit = (final_enhanced * 65535).astype(np.uint16)
    carving_path = output_dir / f"{base_name}_carving_clahe.png"
    cv2.imwrite(str(carving_path), enhanced_16bit)
    print(f"    ✓ Saved 16-bit: {carving_path.name}")

    # NPY file for further processing
    npy_path = output_dir / f"{base_name}_carving_clahe.npy"
    np.save(npy_path, final_enhanced)
    print(f"    ✓ Saved NPY: {npy_path.name}")

    # Preview PNG
    preview_path = output_dir / f"{base_name}_carving_clahe_preview.png"
    preview_8bit = (final_enhanced * 255).astype(np.uint8)
    cv2.imwrite(str(preview_path), preview_8bit)
    print(f"    ✓ Saved preview: {preview_path.name}")

    return final_enhanced

def batch_process_images(image_dir, output_dir):
    """Process all images in directory with CLAHE + Bilateral pipeline"""

    image_dir = Path(image_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True)

    # Find all image files
    image_extensions = ['*.jpg', '*.jpeg', '*.png', '*.webp', '*.JPG', '*.JPEG', '*.PNG']
    image_files = []
    for ext in image_extensions:
        image_files.extend(image_dir.glob(ext))

    image_files = sorted(set(image_files))

    if not image_files:
        print(f"No images found in {image_dir}")
        return

    print(f"\nFound {len(image_files)} images to process")

    # Load ZoeDepth model once
    print("\nLoading ZoeDepth model...")
    model = torch.hub.load(".", "ZoeD_N", source="local", pretrained=True, trust_repo=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = model.to(device)
    model.eval()
    print(f"Using device: {device}")

    # Process each image
    print("\n" + "="*70)
    print("Processing images with CLAHE + Bilateral Filter pipeline")
    print("="*70)

    for image_path in image_files:
        try:
            process_image_for_carving(image_path, model, device, output_dir)
        except Exception as e:
            print(f"    ✗ Error processing {image_path.name}: {e}")
            continue

    print("\n" + "="*70)
    print(f"Done! Processed {len(image_files)} images")
    print(f"Output saved to: {output_dir}")
    print("\nPipeline used:")
    print("  1. ZoeDepth estimation")
    print("  2. High-frequency detail transfer")
    print("  3. Depth inversion (closer = raised)")
    print("  4. CLAHE (clip=2.5, tile=16)")
    print("  5. Bilateral filter (smoothing + edge preservation)")
    print("="*70)

if __name__ == "__main__":
    print("="*70)
    print("Batch Relief Carving Processor")
    print("CLAHE + Bilateral Filter (Winner Configuration)")
    print("="*70)

    image_dir = "images"
    output_dir = "output"

    batch_process_images(image_dir, output_dir)

import warnings
warnings.filterwarnings("ignore")

import torch
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import cv2
import os
from pathlib import Path

def normalize_array(arr, min_val=0, max_val=1):
    """Normalize array to given range"""
    arr_min, arr_max = arr.min(), arr.max()
    if arr_max - arr_min == 0:
        return np.full_like(arr, min_val)
    return (arr - arr_min) / (arr_max - arr_min) * (max_val - min_val) + min_val

def enhance_with_high_freq_detail(image, depth, detail_strength=0.2, invert_depth=True):
    """
    Transfer high-frequency details from image to depth map.
    This is the standard method for relief carving.

    Args:
        image: PIL Image in RGB
        depth: numpy array of depth values
        detail_strength: How much image detail to add (0.1-0.3 recommended)
        invert_depth: If True, inverts depth (closer = deeper carve)

    Returns:
        enhanced_depth: numpy array with enhanced depth
    """
    # Convert to grayscale
    gray = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2GRAY)
    gray_norm = gray.astype(float) / 255.0

    # Resize gray to match depth if needed
    if gray.shape != depth.shape:
        gray_norm = cv2.resize(gray_norm, (depth.shape[1], depth.shape[0]))

    # Extract low-frequency (overall shape) from grayscale using Gaussian blur
    gray_blur = cv2.GaussianBlur(gray_norm, (21, 21), 0)

    # Extract high-frequency (fine detail) - the texture/features we want
    high_freq = gray_norm - gray_blur

    # Normalize depth to 0-1 range
    depth_norm = normalize_array(depth)

    # Invert depth if requested (for different carving styles)
    if invert_depth:
        depth_norm = 1.0 - depth_norm

    # Combine: use depth for overall shape, add high-freq for surface detail
    enhanced = depth_norm + (high_freq * detail_strength)

    # Normalize final result
    return normalize_array(enhanced)

def process_image_for_carving(image_path, output_dir, detail_strength=0.2, invert_depth=True):
    """
    Complete pipeline: Load image -> Generate depth -> Enhance for carving

    Args:
        image_path: Path to input image
        output_dir: Directory for output files
        detail_strength: Detail enhancement strength (default 0.2)
        invert_depth: Invert depth map (default True - closer objects carve deeper)
    """
    base_name = Path(image_path).stem

    print(f"\nProcessing: {base_name}")

    # Load image
    image = Image.open(image_path).convert("RGB")
    print(f"  Image size: {image.size}")

    # Load ZoeDepth model (if not already loaded globally)
    if not hasattr(process_image_for_carving, 'model'):
        print("  Loading ZoeDepth model...")
        model = torch.hub.load(".", "ZoeD_N", source="local", pretrained=True, trust_repo=True)
        device = "cuda" if torch.cuda.is_available() else "cpu"
        model = model.to(device)
        model.eval()
        process_image_for_carving.model = model
        print(f"  Using device: {device}")

    # Run depth estimation
    print("  Running depth estimation...")
    with torch.no_grad():
        depth = process_image_for_carving.model.infer_pil(image)

    print(f"  Depth range: {depth.min():.2f} to {depth.max():.2f}")

    # Enhance with high-frequency detail
    invert_str = "inverted" if invert_depth else "normal"
    print(f"  Enhancing with high-frequency detail ({invert_str} depth)...")
    enhanced_depth = enhance_with_high_freq_detail(image, depth, detail_strength, invert_depth)

    # Save outputs
    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True)

    # 1. Save raw depth map as numpy array
    depth_npy_path = output_dir / f"{base_name}_depth.npy"
    np.save(depth_npy_path, depth)

    # 2. Save enhanced depth as numpy array
    enhanced_npy_path = output_dir / f"{base_name}_carving.npy"
    np.save(enhanced_npy_path, enhanced_depth)

    # 3. Save enhanced depth as 16-bit PNG for CNC software
    enhanced_16bit = (enhanced_depth * 65535).astype(np.uint16)
    enhanced_png_path = output_dir / f"{base_name}_carving.png"
    cv2.imwrite(str(enhanced_png_path), enhanced_16bit)
    print(f"  Saved: {enhanced_png_path.name} (16-bit PNG for CNC)")

    # 4. Create comparison visualization
    fig, axes = plt.subplots(2, 2, figsize=(12, 12))

    # Original image
    axes[0, 0].imshow(image)
    axes[0, 0].set_title("Original Image", fontsize=14, fontweight='bold')
    axes[0, 0].axis("off")

    # Raw depth map
    axes[0, 1].imshow(normalize_array(depth), cmap="gray")
    axes[0, 1].set_title("Raw Depth Map", fontsize=14, fontweight='bold')
    axes[0, 1].axis("off")

    # Enhanced depth for carving
    axes[1, 0].imshow(enhanced_depth, cmap="gray")
    axes[1, 0].set_title("Enhanced for Carving\n(High-Freq Detail)", fontsize=14, fontweight='bold')
    axes[1, 0].axis("off")

    # Depth map in color for reference
    axes[1, 1].imshow(enhanced_depth, cmap="inferno")
    axes[1, 1].set_title("Enhanced (Color Map)", fontsize=14, fontweight='bold')
    axes[1, 1].axis("off")

    plt.suptitle(f"Relief Carving Output: {base_name}", fontsize=16, fontweight='bold')
    plt.tight_layout()

    viz_path = output_dir / f"{base_name}_carving_preview.png"
    plt.savefig(viz_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {viz_path.name}")

    return enhanced_depth

def batch_process_for_carving(images_dir="images", output_dir="output", detail_strength=0.2, invert_depth=True):
    """
    Process all images in a directory for relief carving
    """
    print("="*70)
    print("ZoeDepth Relief Carving Pipeline")
    print("Standard Method: High-Frequency Detail Transfer")
    print("="*70)

    images_dir = Path(images_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True)

    # Find all images
    image_extensions = ['.jpg', '.jpeg', '.png', '.webp', '.bmp']
    image_files = []
    for ext in image_extensions:
        image_files.extend(images_dir.glob(f"*{ext}"))
        image_files.extend(images_dir.glob(f"*{ext.upper()}"))

    print(f"\nFound {len(image_files)} images to process")
    print(f"Detail strength: {detail_strength}")
    print(f"Invert depth: {invert_depth} ({'closer = deeper' if invert_depth else 'closer = shallower'})")
    print("-"*70)

    processed = 0
    for image_path in image_files:
        try:
            process_image_for_carving(image_path, output_dir, detail_strength, invert_depth)
            processed += 1
            print(f"  ✓ Complete!")
        except Exception as e:
            print(f"  ✗ Error: {e}")

    print("\n" + "="*70)
    print(f"Processing complete! {processed}/{len(image_files)} images processed.")
    print(f"\nOutput files (per image):")
    print(f"  • *_carving.png - 16-bit depth map for CNC software")
    print(f"  • *_carving.npy - Numpy array for further processing")
    print(f"  • *_carving_preview.png - Visual preview")
    print(f"  • *_depth.npy - Original depth map")
    print(f"\nAll files saved to: {output_dir.absolute()}")
    print("="*70)

if __name__ == "__main__":
    # Process all images in the images folder
    batch_process_for_carving(
        images_dir="images",
        output_dir="output",
        detail_strength=0.2,  # Adjust this value (0.1-0.3) for more/less detail
        invert_depth=True     # True = closer objects carve deeper, False = closer objects carve shallower
    )

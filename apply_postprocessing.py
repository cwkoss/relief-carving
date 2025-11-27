"""
Apply post-processing to cached depth map - for GUI fast updates
"""
import sys
import numpy as np
from PIL import Image
import cv2
from pathlib import Path

def normalize_array(arr):
    """Normalize array to 0-1 range"""
    arr_min = arr.min()
    arr_max = arr.max()
    if arr_max - arr_min == 0:
        return np.zeros_like(arr)
    return (arr - arr_min) / (arr_max - arr_min)

def enhance_with_high_freq_detail(image_path, depth, detail_strength=0.2, invert_depth=True):
    """Add high-frequency detail from image to depth map"""
    # Load and convert image to grayscale
    image = Image.open(image_path).convert("RGB")
    gray = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2GRAY)
    gray_norm = gray.astype(float) / 255.0

    # Extract high-frequency detail
    gray_blur = cv2.GaussianBlur(gray_norm, (21, 21), 0)
    high_freq = gray_norm - gray_blur

    # Normalize depth
    depth_norm = normalize_array(depth)

    # Invert if requested (for relief carving)
    if invert_depth:
        depth_norm = 1.0 - depth_norm

    # Add high-frequency detail
    enhanced = depth_norm + (high_freq * detail_strength)

    return normalize_array(enhanced)

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

def main():
    image_path = sys.argv[1] if len(sys.argv) > 1 else "gui_temp/input.png"
    depth_path = sys.argv[2] if len(sys.argv) > 2 else "gui_temp/depth.npy"
    output_path = sys.argv[3] if len(sys.argv) > 3 else "gui_temp/output.png"

    # Parse parameters
    detail_strength = float(sys.argv[4]) if len(sys.argv) > 4 else 0.2
    clahe_clip = float(sys.argv[5]) if len(sys.argv) > 5 else 2.5
    clahe_tile = int(sys.argv[6]) if len(sys.argv) > 6 else 16
    bilateral_d = int(sys.argv[7]) if len(sys.argv) > 7 else 9
    bilateral_color = float(sys.argv[8]) if len(sys.argv) > 8 else 75.0
    bilateral_space = float(sys.argv[9]) if len(sys.argv) > 9 else 75.0

    print(f"Applying post-processing...")
    print(f"Parameters: detail={detail_strength}, clahe_clip={clahe_clip}, clahe_tile={clahe_tile}")
    print(f"            bilateral_d={bilateral_d}, color={bilateral_color}, space={bilateral_space}")

    # Load cached depth map
    print(f"Loading depth map from {depth_path}...")
    depth = np.load(depth_path)
    print(f"Depth range: {depth.min():.2f} to {depth.max():.2f}")

    # Save debug images
    debug_dir = Path("gui_temp/debug")
    debug_dir.mkdir(exist_ok=True)

    # Check for alpha mask
    alpha_mask_path = Path("gui_temp") / "alpha_mask.npy"
    alpha_mask = None
    if alpha_mask_path.exists():
        alpha_mask = np.load(alpha_mask_path)
        fully_transparent = (alpha_mask == 0).sum()
        partially_transparent = ((alpha_mask > 0) & (alpha_mask < 1.0)).sum()
        fully_opaque = (alpha_mask == 1.0).sum()
        print(f"Alpha mask loaded: shape={alpha_mask.shape}, range=[{alpha_mask.min():.3f}, {alpha_mask.max():.3f}]")
        print(f"  Fully transparent pixels: {fully_transparent}")
        print(f"  Partially transparent: {partially_transparent}")
        print(f"  Fully opaque: {fully_opaque}")
        # Save alpha mask visualization
        cv2.imwrite(str(debug_dir / "00_alpha_mask.png"), (alpha_mask * 255).astype(np.uint8))
    else:
        print(f"WARNING: Alpha mask not found at {alpha_mask_path}")

    # Save raw ZoeDepth output (normalized but NOT inverted)
    depth_norm = normalize_array(depth)
    cv2.imwrite(str(debug_dir / "01_depth_raw.png"), (depth_norm * 255).astype(np.uint8))

    # Apply alpha mask to RAW depth (before inversion)
    if alpha_mask is not None:
        print(f"Applying alpha mask to raw depth...")
        print(f"  Before masking: depth range=[{depth_norm.min():.3f}, {depth_norm.max():.3f}]")
        depth_masked_raw = depth_norm * alpha_mask
        print(f"  After masking: depth range=[{depth_masked_raw.min():.3f}, {depth_masked_raw.max():.3f}]")
        pixels_zeroed = (depth_masked_raw == 0).sum() - (depth_norm == 0).sum()
        print(f"  Pixels set to zero by mask: {pixels_zeroed}")
        cv2.imwrite(str(debug_dir / "02_depth_raw_masked.png"), (depth_masked_raw * 255).astype(np.uint8))
    else:
        print(f"No alpha mask - skipping masking step")
        depth_masked_raw = depth_norm
        cv2.imwrite(str(debug_dir / "02_depth_raw_masked.png"), (depth_norm * 255).astype(np.uint8))

    # NOW invert the masked depth
    depth_inverted = 1.0 - depth_masked_raw
    cv2.imwrite(str(debug_dir / "03_depth_inverted.png"), (depth_inverted * 255).astype(np.uint8))

    # Apply post-processing pipeline on inverted masked depth
    print(f"Enhancing with high-frequency detail (strength={detail_strength})...")
    gray = cv2.cvtColor(np.array(Image.open(image_path).convert("RGB")), cv2.COLOR_RGB2GRAY)
    gray_norm = gray.astype(float) / 255.0
    gray_blur = cv2.GaussianBlur(gray_norm, (21, 21), 0)
    high_freq = gray_norm - gray_blur
    enhanced_base = depth_inverted + (high_freq * detail_strength)
    enhanced_base = normalize_array(enhanced_base)
    cv2.imwrite(str(debug_dir / "04_detail_added.png"), (enhanced_base * 255).astype(np.uint8))

    print(f"Applying CLAHE (clip={clahe_clip}, tile={clahe_tile})...")
    clahe_enhanced = apply_clahe_refined(enhanced_base, clip_limit=clahe_clip, tile_size=clahe_tile)
    cv2.imwrite(str(debug_dir / "05_clahe.png"), (clahe_enhanced * 255).astype(np.uint8))

    print(f"Applying bilateral filter (d={bilateral_d}, color={bilateral_color}, space={bilateral_space})...")
    final_enhanced = apply_bilateral_filter(clahe_enhanced, d=bilateral_d, sigma_color=bilateral_color, sigma_space=bilateral_space)
    cv2.imwrite(str(debug_dir / "06_bilateral.png"), (final_enhanced * 255).astype(np.uint8))

    # Re-apply alpha mask at the end to ensure clean edges
    if alpha_mask is not None:
        print(f"Re-applying alpha mask to final output for clean edges...")
        final_enhanced = final_enhanced * alpha_mask
        cv2.imwrite(str(debug_dir / "07_final_masked.png"), (final_enhanced * 255).astype(np.uint8))
    else:
        cv2.imwrite(str(debug_dir / "07_final_masked.png"), (final_enhanced * 255).astype(np.uint8))

    # Save outputs
    print(f"Saving to {output_path}...")
    output_16bit = (final_enhanced * 65535).astype(np.uint16)
    cv2.imwrite(output_path, output_16bit)

    # Also save 8-bit preview
    preview_path = output_path.replace(".png", "_preview.png")
    output_8bit = (final_enhanced * 255).astype(np.uint8)
    cv2.imwrite(preview_path, output_8bit)

    print("Post-processing complete!")

if __name__ == "__main__":
    main()

"""
Processing script for GUI - runs inside Docker container
"""
import sys
import numpy as np
from PIL import Image
import torch
from pathlib import Path
from process_for_carving import enhance_with_high_freq_detail
import cv2

def normalize_array(arr):
    """Normalize array to 0-1 range"""
    arr_min = arr.min()
    arr_max = arr.max()
    if arr_max - arr_min == 0:
        return np.zeros_like(arr)
    return (arr - arr_min) / (arr_max - arr_min)

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
    input_path = sys.argv[1] if len(sys.argv) > 1 else "gui_temp/input.jpg"
    output_path = sys.argv[2] if len(sys.argv) > 2 else "gui_temp/output.png"

    # Parse optional parameters
    detail_strength = float(sys.argv[3]) if len(sys.argv) > 3 else 0.2
    clahe_clip = float(sys.argv[4]) if len(sys.argv) > 4 else 2.5
    clahe_tile = int(sys.argv[5]) if len(sys.argv) > 5 else 16
    bilateral_d = int(sys.argv[6]) if len(sys.argv) > 6 else 9
    bilateral_color = float(sys.argv[7]) if len(sys.argv) > 7 else 75.0
    bilateral_space = float(sys.argv[8]) if len(sys.argv) > 8 else 75.0

    print(f"Processing: {input_path}")
    print(f"Parameters: detail={detail_strength}, clahe_clip={clahe_clip}, clahe_tile={clahe_tile}")
    print(f"            bilateral_d={bilateral_d}, color={bilateral_color}, space={bilateral_space}")

    # Load image
    image = Image.open(input_path).convert("RGB")

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

    # Enhance with high-frequency detail
    print(f"Enhancing with high-frequency detail (strength={detail_strength})...")
    enhanced_base = enhance_with_high_freq_detail(image, depth, detail_strength=detail_strength, invert_depth=True)

    # Apply CLAHE
    print(f"Applying CLAHE (clip={clahe_clip}, tile={clahe_tile})...")
    clahe_enhanced = apply_clahe_refined(enhanced_base, clip_limit=clahe_clip, tile_size=clahe_tile)

    # Apply bilateral filter
    print(f"Applying bilateral filter (d={bilateral_d}, color={bilateral_color}, space={bilateral_space})...")
    final_enhanced = apply_bilateral_filter(clahe_enhanced, d=bilateral_d, sigma_color=bilateral_color, sigma_space=bilateral_space)

    # Save 16-bit output
    print(f"Saving to {output_path}...")
    output_16bit = (final_enhanced * 65535).astype(np.uint16)
    cv2.imwrite(output_path, output_16bit)

    # Also save 8-bit preview
    preview_path = output_path.replace(".png", "_preview.png")
    output_8bit = (final_enhanced * 255).astype(np.uint8)
    cv2.imwrite(preview_path, output_8bit)

    print("Processing complete!")

if __name__ == "__main__":
    main()

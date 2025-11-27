import warnings
warnings.filterwarnings("ignore")

import sys
import time
import numpy as np
from PIL import Image
import cv2
from pathlib import Path
from process_for_carving import normalize_array, enhance_with_high_freq_detail
import torch

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

if __name__ == "__main__":
    image_path = "images/otto.jpg"
    output_dir = "output"

    print("="*70)
    print("Processing otto.jpg with CLAHE + Bilateral (TIMED)")
    print("="*70)

    total_start = time.time()

    base_name = Path(image_path).stem
    output_dir = Path(output_dir)

    # Load image
    load_start = time.time()
    print(f"\n1. Loading image...")
    image = Image.open(image_path).convert("RGB")
    load_time = time.time() - load_start
    print(f"   ✓ Loaded ({image.size}) in {load_time:.2f}s")

    # Load ZoeDepth model
    model_start = time.time()
    print(f"\n2. Loading ZoeDepth model...")
    model = torch.hub.load(".", "ZoeD_N", source="local", pretrained=True, trust_repo=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = model.to(device)
    model.eval()
    model_time = time.time() - model_start
    print(f"   ✓ Model loaded in {model_time:.2f}s (device: {device})")

    # Run depth estimation
    depth_start = time.time()
    print(f"\n3. Running depth estimation...")
    with torch.no_grad():
        depth = model.infer_pil(image)
    depth_time = time.time() - depth_start
    print(f"   ✓ Depth estimation in {depth_time:.2f}s")
    print(f"   Depth range: {depth.min():.2f} to {depth.max():.2f}")

    # Enhance with high-frequency detail
    enhance_start = time.time()
    print(f"\n4. Enhancing with high-frequency detail...")
    enhanced_base = enhance_with_high_freq_detail(image, depth, detail_strength=0.2, invert_depth=True)
    enhance_time = time.time() - enhance_start
    print(f"   ✓ Enhancement in {enhance_time:.2f}s")

    # Apply CLAHE
    clahe_start = time.time()
    print(f"\n5. Applying CLAHE (clip=2.5, tile=16)...")
    clahe_enhanced = apply_clahe_refined(enhanced_base, clip_limit=2.5, tile_size=16)
    clahe_time = time.time() - clahe_start
    print(f"   ✓ CLAHE in {clahe_time:.2f}s")

    # Apply bilateral filter
    bilateral_start = time.time()
    print(f"\n6. Applying bilateral filter...")
    final_enhanced = apply_bilateral_filter(clahe_enhanced, d=9, sigma_color=75, sigma_space=75)
    bilateral_time = time.time() - bilateral_start
    print(f"   ✓ Bilateral filter in {bilateral_time:.2f}s")

    # Save outputs
    save_start = time.time()
    print(f"\n7. Saving files...")
    enhanced_16bit = (final_enhanced * 65535).astype(np.uint16)
    carving_path = output_dir / f"{base_name}_carving_clahe.png"
    cv2.imwrite(str(carving_path), enhanced_16bit)

    npy_path = output_dir / f"{base_name}_carving_clahe.npy"
    np.save(npy_path, final_enhanced)

    preview_path = output_dir / f"{base_name}_carving_clahe_preview.png"
    preview_8bit = (final_enhanced * 255).astype(np.uint8)
    cv2.imwrite(str(preview_path), preview_8bit)
    save_time = time.time() - save_start
    print(f"   ✓ Files saved in {save_time:.2f}s")

    total_time = time.time() - total_start

    print(f"\n{'='*70}")
    print(f"TIMING BREAKDOWN:")
    print(f"{'='*70}")
    print(f"  1. Load image:           {load_time:6.2f}s")
    print(f"  2. Load model:           {model_time:6.2f}s")
    print(f"  3. Depth estimation:     {depth_time:6.2f}s")
    print(f"  4. High-freq detail:     {enhance_time:6.2f}s")
    print(f"  5. CLAHE:                {clahe_time:6.2f}s")
    print(f"  6. Bilateral filter:     {bilateral_time:6.2f}s")
    print(f"  7. Save files:           {save_time:6.2f}s")
    print(f"{'='*70}")
    print(f"  TOTAL TIME:              {total_time:6.2f}s")
    print(f"{'='*70}")

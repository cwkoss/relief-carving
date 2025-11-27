import warnings
warnings.filterwarnings("ignore")

import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import cv2
from pathlib import Path
from process_for_carving import normalize_array, enhance_with_high_freq_detail
import torch

def apply_clahe_refined(depth_map, clip_limit=2.0, tile_size=16):
    """Apply CLAHE with specified parameters"""
    depth_8bit = (depth_map * 255).astype(np.uint8)
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=(tile_size, tile_size))
    enhanced = clahe.apply(depth_8bit)
    return enhanced.astype(float) / 255.0

def apply_clahe_with_smoothing(depth_map, clip_limit=2.0, tile_size=16, smooth_background=True):
    """Apply CLAHE and optionally smooth the background to remove artifacts"""
    # Apply CLAHE
    enhanced = apply_clahe_refined(depth_map, clip_limit, tile_size)

    if smooth_background:
        # Create mask: highlights vs background
        # Threshold at 0.5 - above is highlights (keep detail), below is background (smooth)
        mask = depth_map > 0.5

        # Blur the background
        background_smoothed = cv2.GaussianBlur(enhanced, (5, 5), 0)

        # Blend: use enhanced where highlights, smoothed where background
        result = np.where(mask, enhanced, background_smoothed)

        # Smooth the transition between masked regions
        mask_blur = cv2.GaussianBlur(mask.astype(float), (11, 11), 0)
        result = enhanced * mask_blur + background_smoothed * (1 - mask_blur)

        return result

    return enhanced

def apply_bilateral_filter(depth_map, d=9, sigma_color=75, sigma_space=75):
    """
    Bilateral filter - reduces noise while preserving edges.
    Great for removing texture artifacts while keeping sharp features.
    """
    depth_8bit = (depth_map * 255).astype(np.uint8)
    filtered = cv2.bilateralFilter(depth_8bit, d, sigma_color, sigma_space)
    return filtered.astype(float) / 255.0

def process_clahe_variations(image_path, output_dir):
    """Process image with different CLAHE refinements"""

    base_name = Path(image_path).stem
    output_dir = Path(output_dir)

    print(f"\nProcessing: {base_name}")

    # Load image
    image = Image.open(image_path).convert("RGB")
    print(f"  Image size: {image.size}")

    # Load ZoeDepth model
    print("  Loading ZoeDepth model...")
    model = torch.hub.load(".", "ZoeD_N", source="local", pretrained=True, trust_repo=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = model.to(device)
    model.eval()
    print(f"  Using device: {device}")

    # Run depth estimation
    print("  Running depth estimation...")
    with torch.no_grad():
        depth = model.infer_pil(image)

    print(f"  Depth range: {depth.min():.2f} to {depth.max():.2f}")

    # Enhance with high-frequency detail (base version)
    print("  Enhancing with high-frequency detail...")
    enhanced_base = enhance_with_high_freq_detail(image, depth, detail_strength=0.2, invert_depth=True)

    print("\n  Applying CLAHE variations...")

    # Different CLAHE configurations
    variations = {
        "original_clahe": ("Original CLAHE\nclip=3.0, tile=8",
                           apply_clahe_refined(enhanced_base, clip_limit=3.0, tile_size=8)),

        "clahe_gentle": ("Gentle CLAHE\nclip=2.0, tile=16",
                        apply_clahe_refined(enhanced_base, clip_limit=2.0, tile_size=16)),

        "clahe_large_tiles": ("Large Tiles\nclip=2.5, tile=32",
                             apply_clahe_refined(enhanced_base, clip_limit=2.5, tile_size=32)),

        "clahe_smoothed_bg": ("Smoothed Background\nclip=2.5, tile=16",
                             apply_clahe_with_smoothing(enhanced_base, clip_limit=2.5, tile_size=16, smooth_background=True)),

        "clahe_bilateral": ("CLAHE + Bilateral Filter\n(removes spiky texture)",
                           apply_bilateral_filter(
                               apply_clahe_refined(enhanced_base, clip_limit=2.5, tile_size=16),
                               d=9, sigma_color=75, sigma_space=75)),

        "bilateral_only": ("Bilateral Filter Only\n(smooth but detailed)",
                          apply_bilateral_filter(enhanced_base, d=9, sigma_color=50, sigma_space=50)),
    }

    # Save all versions
    for name, (desc, enhanced) in variations.items():
        enhanced_16bit = (enhanced * 65535).astype(np.uint16)
        save_path = output_dir / f"{base_name}_clahe_{name}.png"
        cv2.imwrite(str(save_path), enhanced_16bit)
        print(f"    Saved: {save_path.name}")

    # Create comparison visualization
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    axes = axes.flatten()

    plot_order = [
        ("original_clahe", variations["original_clahe"]),
        ("clahe_gentle", variations["clahe_gentle"]),
        ("clahe_large_tiles", variations["clahe_large_tiles"]),
        ("clahe_smoothed_bg", variations["clahe_smoothed_bg"]),
        ("clahe_bilateral", variations["clahe_bilateral"]),
        ("bilateral_only", variations["bilateral_only"]),
    ]

    for idx, (name, (title, enhanced)) in enumerate(plot_order):
        axes[idx].imshow(enhanced, cmap="gray", vmin=0, vmax=1)
        axes[idx].set_title(title, fontsize=12, fontweight='bold')
        axes[idx].axis("off")

    plt.suptitle(f"CLAHE Refinements: {base_name}\nReducing artifacts & spikiness",
                 fontsize=16, fontweight='bold')
    plt.tight_layout()

    comparison_path = output_dir / f"{base_name}_clahe_refinements.png"
    plt.savefig(comparison_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"\n  Saved comparison: {comparison_path.name}")

    # Create recommended comparison
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))

    recommendations = [
        ("Original CLAHE\n(artifacts visible)", variations["original_clahe"][1]),
        ("CLAHE + Bilateral\n(Recommended)", variations["clahe_bilateral"][1]),
        ("Bilateral Only\n(Smoothest)", variations["bilateral_only"][1])
    ]

    for idx, (title, enhanced) in enumerate(recommendations):
        axes[idx].imshow(enhanced, cmap="gray", vmin=0, vmax=1)
        axes[idx].set_title(title, fontsize=14, fontweight='bold')
        axes[idx].axis("off")

    plt.suptitle(f"Recommended: {base_name}", fontsize=16, fontweight='bold')
    plt.tight_layout()

    rec_path = output_dir / f"{base_name}_clahe_recommended.png"
    plt.savefig(rec_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved recommended: {rec_path.name}")

if __name__ == "__main__":
    print("="*70)
    print("CLAHE Refinement for Relief Carving")
    print("Reducing artifacts and spiky texture")
    print("="*70)

    image_path = "images/otto.jpg"
    output_dir = "output"

    process_clahe_variations(image_path, output_dir)

    print("\n" + "="*70)
    print("Done! Generated refined CLAHE variations:")
    print("  • Gentle CLAHE - Reduced clip limit (less aggressive)")
    print("  • Large Tiles - Bigger tiles = fewer artifacts")
    print("  • Smoothed Background - Blur artifacts in background only")
    print("  • CLAHE + Bilateral - Best of both: detail + smoothness")
    print("  • Bilateral Only - Smoothest result, still detailed")
    print("\nRecommendation: Try 'clahe_bilateral' - it removes spikiness")
    print("while preserving edge detail for carving!")
    print("="*70)

import warnings
warnings.filterwarnings("ignore")

import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import cv2
from pathlib import Path
from process_for_carving import process_image_for_carving, normalize_array, enhance_with_high_freq_detail
import torch

def normalize_array(arr, min_val=0, max_val=1):
    """Normalize array to given range"""
    arr_min, arr_max = arr.min(), arr.max()
    if arr_max - arr_min == 0:
        return np.full_like(arr, min_val)
    return (arr - arr_min) / (arr_max - arr_min) * (max_val - min_val) + min_val

def method_1_gamma_adjustment(depth_map, gamma=0.5):
    """
    Gamma adjustment - expands highlights, compresses shadows.
    gamma < 1.0 brightens and expands highlight detail
    """
    return np.power(depth_map, gamma)

def method_2_power_curve_highlights(depth_map, power=2.0):
    """
    Power curve that expands the upper range more than lower.
    Inverts, applies power, inverts back.
    """
    # Invert so highlights are at top
    inverted = 1.0 - depth_map
    # Apply power (compresses darks, expands lights)
    adjusted = np.power(inverted, 1.0 / power)
    # Invert back
    return 1.0 - adjusted

def method_3_selective_contrast(depth_map, threshold=0.5, contrast=1.5):
    """
    Apply contrast only to values above threshold (highlights).
    Leaves darks alone, stretches highlights.
    """
    result = depth_map.copy()

    # Find highlight region
    mask = depth_map > threshold

    # Extract highlights
    highlights = depth_map[mask]

    if len(highlights) > 0:
        # Normalize highlights to 0-1
        h_min, h_max = highlights.min(), highlights.max()
        if h_max > h_min:
            normalized = (highlights - h_min) / (h_max - h_min)

            # Apply contrast
            centered = normalized - 0.5
            contrasted = centered * contrast
            contrasted = np.clip(contrasted + 0.5, 0, 1)

            # Scale back to original range
            adjusted = contrasted * (h_max - h_min) + h_min

            # Put back
            result[mask] = adjusted

    return result

def method_4_clahe(depth_map, clip_limit=2.0, tile_size=8):
    """
    CLAHE - Contrast Limited Adaptive Histogram Equalization.
    Great for bringing out local detail without washing out.
    """
    # Convert to 8-bit for CLAHE
    depth_8bit = (depth_map * 255).astype(np.uint8)

    # Apply CLAHE
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=(tile_size, tile_size))
    enhanced = clahe.apply(depth_8bit)

    # Convert back to 0-1
    return enhanced.astype(float) / 255.0

def method_5_histogram_stretching(depth_map, lower_percentile=1, upper_percentile=99):
    """
    Stretch histogram based on percentiles to preserve highlights.
    """
    lower = np.percentile(depth_map, lower_percentile)
    upper = np.percentile(depth_map, upper_percentile)

    if upper > lower:
        stretched = (depth_map - lower) / (upper - lower)
        return np.clip(stretched, 0, 1)
    return depth_map

def method_6_sigmoid_curve(depth_map, gain=10, cutoff=0.5):
    """
    S-curve (sigmoid) that preserves highlights better.
    Adjusts midpoint to favor highlights.
    """
    # Sigmoid function centered at cutoff
    x = depth_map
    adjusted = 1.0 / (1.0 + np.exp(-gain * (x - cutoff)))
    return normalize_array(adjusted)

def process_highlight_enhancements(image_path, output_dir):
    """Process image with different highlight enhancement methods"""

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

    # Try different enhancement methods
    print("\n  Applying enhancement methods...")

    enhanced_versions = {
        "original": enhanced_base,
        "gamma_0.5": method_1_gamma_adjustment(enhanced_base, gamma=0.5),
        "gamma_0.7": method_1_gamma_adjustment(enhanced_base, gamma=0.7),
        "power_curve": method_2_power_curve_highlights(enhanced_base, power=2.0),
        "selective_contrast": method_3_selective_contrast(enhanced_base, threshold=0.5, contrast=1.5),
        "clahe": method_4_clahe(enhanced_base, clip_limit=3.0, tile_size=8),
        "histogram_stretch": method_5_histogram_stretching(enhanced_base, lower_percentile=2, upper_percentile=98),
        "sigmoid": method_6_sigmoid_curve(enhanced_base, gain=8, cutoff=0.4)
    }

    # Save all versions
    for name, enhanced in enhanced_versions.items():
        enhanced_16bit = (enhanced * 65535).astype(np.uint16)
        save_path = output_dir / f"{base_name}_highlight_{name}.png"
        cv2.imwrite(str(save_path), enhanced_16bit)
        print(f"    Saved: {save_path.name}")

    # Create comparison visualization
    fig, axes = plt.subplots(3, 3, figsize=(18, 18))
    axes = axes.flatten()

    methods = [
        ("Original", "original"),
        ("Gamma 0.5\n(Expand highlights)", "gamma_0.5"),
        ("Gamma 0.7\n(Moderate expand)", "gamma_0.7"),
        ("Power Curve\n(Stretch upper range)", "power_curve"),
        ("Selective Contrast\n(Highlights only)", "selective_contrast"),
        ("CLAHE\n(Adaptive equalization)", "clahe"),
        ("Histogram Stretch\n(Percentile based)", "histogram_stretch"),
        ("Sigmoid Curve\n(S-curve)", "sigmoid"),
    ]

    for idx, (title, key) in enumerate(methods):
        if idx < len(axes):
            axes[idx].imshow(enhanced_versions[key], cmap="gray", vmin=0, vmax=1)
            axes[idx].set_title(title, fontsize=12, fontweight='bold')
            axes[idx].axis("off")

            # Add histogram in corner
            hist, bins = np.histogram(enhanced_versions[key].flatten(), bins=50, range=(0, 1))
            axes[idx].text(0.02, 0.98, f"Highlights: {np.sum(enhanced_versions[key] > 0.7)/enhanced_versions[key].size*100:.1f}%",
                          transform=axes[idx].transAxes, verticalalignment='top',
                          bbox=dict(boxstyle='round', facecolor='white', alpha=0.7), fontsize=8)

    # Hide unused subplot
    if len(methods) < len(axes):
        axes[-1].axis("off")

    plt.suptitle(f"Highlight Enhancement Comparison: {base_name}", fontsize=16, fontweight='bold')
    plt.tight_layout()

    comparison_path = output_dir / f"{base_name}_highlight_comparison.png"
    plt.savefig(comparison_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"\n  Saved comparison: {comparison_path.name}")

    # Create detailed comparison of top 3 methods
    fig, axes = plt.subplots(1, 4, figsize=(20, 5))

    top_methods = [
        ("Original", "original"),
        ("Gamma 0.5\n(Recommended)", "gamma_0.5"),
        ("CLAHE\n(Most Detail)", "clahe"),
        ("Selective Contrast", "selective_contrast")
    ]

    for idx, (title, key) in enumerate(top_methods):
        axes[idx].imshow(enhanced_versions[key], cmap="gray", vmin=0, vmax=1)
        axes[idx].set_title(title, fontsize=14, fontweight='bold')
        axes[idx].axis("off")

    plt.suptitle(f"Top Methods for {base_name}", fontsize=16, fontweight='bold')
    plt.tight_layout()

    top_path = output_dir / f"{base_name}_top_methods.png"
    plt.savefig(top_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved top methods: {top_path.name}")

if __name__ == "__main__":
    print("="*70)
    print("ZoeDepth Highlight Enhancement for Relief Carving")
    print("Focus: Preserve and enhance detail in raised areas (whites)")
    print("="*70)

    image_path = "images/otto.jpg"
    output_dir = "output"

    process_highlight_enhancements(image_path, output_dir)

    print("\n" + "="*70)
    print("Done! Generated multiple highlight-focused enhancements:")
    print("  • Gamma 0.5 - Expands highlights, good detail preservation")
    print("  • Gamma 0.7 - Moderate highlight expansion")
    print("  • Power Curve - Stretches upper range more than lower")
    print("  • Selective Contrast - Only adjusts highlights, leaves darks")
    print("  • CLAHE - Adaptive local contrast, excellent detail")
    print("  • Histogram Stretch - Percentile-based stretching")
    print("  • Sigmoid - S-curve for smooth transitions")
    print("\nRecommendations:")
    print("  • Best overall: Gamma 0.5 or CLAHE")
    print("  • Most detail in highlights: CLAHE")
    print("  • Smoothest: Gamma 0.5 or 0.7")
    print("="*70)

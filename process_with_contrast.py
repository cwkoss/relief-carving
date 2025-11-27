import warnings
warnings.filterwarnings("ignore")

import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import cv2
from pathlib import Path
from process_for_carving import process_image_for_carving, normalize_array, enhance_with_high_freq_detail
import torch

def adjust_contrast(depth_map, contrast=1.5):
    """
    Adjust contrast of depth map to increase variance.

    Args:
        depth_map: numpy array (0-1 normalized)
        contrast: contrast multiplier (1.0 = no change, >1.0 = more contrast)

    Returns:
        contrast-adjusted depth map
    """
    # Center around 0.5
    centered = depth_map - 0.5

    # Apply contrast
    adjusted = centered * contrast

    # Shift back and clip
    adjusted = adjusted + 0.5
    adjusted = np.clip(adjusted, 0, 1)

    return adjusted

def process_with_multiple_contrasts(image_path, output_dir):
    """Process image with different contrast levels for comparison"""

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

    # Try different contrast levels
    contrast_levels = [1.0, 1.5, 2.0, 2.5]
    enhanced_versions = []

    for contrast in contrast_levels:
        adjusted = adjust_contrast(enhanced_base, contrast=contrast)
        enhanced_versions.append(adjusted)

        # Save individual version
        enhanced_16bit = (adjusted * 65535).astype(np.uint16)
        save_path = output_dir / f"{base_name}_contrast_{contrast:.1f}.png"
        cv2.imwrite(str(save_path), enhanced_16bit)
        print(f"  Saved: {save_path.name}")

    # Create comparison visualization
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))

    # Original image
    axes[0, 0].imshow(image)
    axes[0, 0].set_title("Original Image", fontsize=14, fontweight='bold')
    axes[0, 0].axis("off")

    # Raw depth
    axes[0, 1].imshow(normalize_array(depth), cmap="gray")
    axes[0, 1].set_title("Raw Depth Map", fontsize=14, fontweight='bold')
    axes[0, 1].axis("off")

    # Base enhanced (contrast 1.0)
    axes[0, 2].imshow(enhanced_versions[0], cmap="gray")
    axes[0, 2].set_title(f"Contrast 1.0 (Original)", fontsize=14, fontweight='bold')
    axes[0, 2].axis("off")

    # Different contrast levels
    for idx, (enhanced, contrast) in enumerate(zip(enhanced_versions[1:], contrast_levels[1:]), 1):
        row = 1
        col = idx - 1
        axes[row, col].imshow(enhanced, cmap="gray")
        axes[row, col].set_title(f"Contrast {contrast:.1f}", fontsize=14, fontweight='bold')
        axes[row, col].axis("off")

    plt.suptitle(f"Contrast Comparison: {base_name}", fontsize=16, fontweight='bold')
    plt.tight_layout()

    comparison_path = output_dir / f"{base_name}_contrast_comparison.png"
    plt.savefig(comparison_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {comparison_path.name}")

    # Create color visualization for best contrast (2.0)
    fig, axes = plt.subplots(1, 2, figsize=(12, 6))

    axes[0].imshow(enhanced_versions[0], cmap="gray")
    axes[0].set_title("Original (Contrast 1.0)", fontsize=14)
    axes[0].axis("off")

    axes[1].imshow(enhanced_versions[2], cmap="gray")  # contrast 2.0
    axes[1].set_title("Enhanced (Contrast 2.0)", fontsize=14)
    axes[1].axis("off")

    plt.suptitle(f"Recommended: Contrast 2.0 for {base_name}", fontsize=16, fontweight='bold')
    plt.tight_layout()

    recommended_path = output_dir / f"{base_name}_recommended.png"
    plt.savefig(recommended_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {recommended_path.name}")

if __name__ == "__main__":
    print("="*70)
    print("ZoeDepth Contrast Adjustment for Relief Carving")
    print("="*70)

    image_path = "images/otto.jpg"
    output_dir = "output"

    process_with_multiple_contrasts(image_path, output_dir)

    print("\n" + "="*70)
    print("Done! Generated depth maps with different contrast levels:")
    print("  • Contrast 1.0 - Original (subtle relief)")
    print("  • Contrast 1.5 - Moderate (good balance)")
    print("  • Contrast 2.0 - Strong (recommended for carving)")
    print("  • Contrast 2.5 - Maximum (very dramatic)")
    print("\nRecommendation: Try Contrast 2.0 for best carving results!")
    print("="*70)

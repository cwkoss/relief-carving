import warnings
warnings.filterwarnings("ignore")

import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import cv2
from pathlib import Path

def normalize_array(arr, min_val=0, max_val=1):
    """Normalize array to given range"""
    arr_min, arr_max = arr.min(), arr.max()
    if arr_max - arr_min == 0:
        return np.full_like(arr, min_val)
    return (arr - arr_min) / (arr_max - arr_min) * (max_val - min_val) + min_val

def method_1_edge_overlay(image, depth):
    """Method 1: Edge detection + depth overlay"""
    # Convert to grayscale
    gray = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2GRAY)

    # Detect edges using Canny
    edges = cv2.Canny(gray, 50, 150)

    # Normalize depth to 0-1
    depth_norm = normalize_array(depth)

    # Add edges as relief detail (edges create small depressions/raises)
    edge_strength = 0.15  # How much edges affect the depth
    edges_norm = edges.astype(float) / 255.0

    enhanced = depth_norm - (edges_norm * edge_strength)

    return enhanced, edges

def method_2_high_freq_detail(image, depth):
    """Method 2: Transfer high-frequency details from image to depth"""
    # Convert to grayscale
    gray = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2GRAY)
    gray_norm = gray.astype(float) / 255.0

    # Resize gray to match depth if needed
    if gray.shape != depth.shape:
        gray_norm = cv2.resize(gray_norm, (depth.shape[1], depth.shape[0]))

    # Extract low-frequency (overall shape) from grayscale
    gray_blur = cv2.GaussianBlur(gray_norm, (21, 21), 0)

    # Extract high-frequency (fine detail)
    high_freq = gray_norm - gray_blur

    # Normalize depth
    depth_norm = normalize_array(depth)

    # Combine: use depth for shape, add high-freq for detail
    detail_strength = 0.2
    enhanced = depth_norm + (high_freq * detail_strength)

    return normalize_array(enhanced), high_freq

def method_3_simple_overlay(image, depth):
    """Method 3: Simple semi-transparent grayscale overlay"""
    # Convert to grayscale
    gray = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2GRAY)
    gray_norm = gray.astype(float) / 255.0

    # Resize to match depth
    if gray.shape != depth.shape:
        gray_norm = cv2.resize(gray_norm, (depth.shape[1], depth.shape[0]))

    # Normalize depth
    depth_norm = normalize_array(depth)

    # Blend: 70% depth, 30% grayscale
    alpha = 0.7
    enhanced = alpha * depth_norm + (1 - alpha) * gray_norm

    return enhanced, gray_norm

def method_4_adaptive_detail(image, depth):
    """Method 4: Adaptive edge enhancement with depth-aware strength"""
    gray = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2GRAY)

    # Detect edges with multiple scales
    edges1 = cv2.Canny(gray, 30, 100)
    edges2 = cv2.Canny(gray, 50, 150)
    edges3 = cv2.Canny(gray, 100, 200)

    # Combine edges (fine to coarse)
    edges_combined = (edges1 * 0.5 + edges2 * 0.3 + edges3 * 0.2) / 255.0

    # Normalize depth
    depth_norm = normalize_array(depth)

    # Apply edge detail with varying strength
    # Stronger edges on flatter areas, weaker on steep areas
    depth_gradient = np.gradient(depth_norm)[0]
    edge_strength = 0.2 * (1 - normalize_array(np.abs(depth_gradient)))

    enhanced = depth_norm - (edges_combined * edge_strength)

    return normalize_array(enhanced), edges_combined

def enhance_depth_for_carving(image_path, depth_path, output_dir):
    """Process a single image with all enhancement methods"""
    # Load image and depth
    image = Image.open(image_path).convert("RGB")
    depth = np.load(depth_path)

    base_name = Path(image_path).stem

    print(f"\nProcessing: {base_name}")
    print(f"  Image size: {image.size}")
    print(f"  Depth shape: {depth.shape}")

    # Apply all methods
    enhanced1, edges = method_1_edge_overlay(image, depth)
    enhanced2, high_freq = method_2_high_freq_detail(image, depth)
    enhanced3, gray = method_3_simple_overlay(image, depth)
    enhanced4, multi_edges = method_4_adaptive_detail(image, depth)

    # Create comparison visualization
    fig, axes = plt.subplots(3, 3, figsize=(18, 18))

    # Row 1: Original, Depth, Grayscale
    axes[0, 0].imshow(image)
    axes[0, 0].set_title("Original Image", fontsize=14, fontweight='bold')
    axes[0, 0].axis("off")

    axes[0, 1].imshow(normalize_array(depth), cmap="gray")
    axes[0, 1].set_title("Original Depth Map", fontsize=14, fontweight='bold')
    axes[0, 1].axis("off")

    axes[0, 2].imshow(cv2.cvtColor(np.array(image), cv2.COLOR_RGB2GRAY), cmap="gray")
    axes[0, 2].set_title("Grayscale Reference", fontsize=14, fontweight='bold')
    axes[0, 2].axis("off")

    # Row 2: Enhanced methods 1 & 2
    axes[1, 0].imshow(enhanced1, cmap="gray")
    axes[1, 0].set_title("Method 1: Edge Overlay\n(Canny edges added to depth)", fontsize=12)
    axes[1, 0].axis("off")

    axes[1, 1].imshow(enhanced2, cmap="gray")
    axes[1, 1].set_title("Method 2: High-Freq Detail\n(Texture detail transfer)", fontsize=12)
    axes[1, 1].axis("off")

    axes[1, 2].imshow(enhanced3, cmap="gray")
    axes[1, 2].set_title("Method 3: Simple Overlay\n(70% depth + 30% grayscale)", fontsize=12)
    axes[1, 2].axis("off")

    # Row 3: Method 4 and extracted features
    axes[2, 0].imshow(enhanced4, cmap="gray")
    axes[2, 0].set_title("Method 4: Adaptive Detail\n(Multi-scale edges)", fontsize=12)
    axes[2, 0].axis("off")

    axes[2, 1].imshow(edges, cmap="gray")
    axes[2, 1].set_title("Extracted Edges\n(for Method 1)", fontsize=12)
    axes[2, 1].axis("off")

    axes[2, 2].imshow(high_freq + 0.5, cmap="gray")
    axes[2, 2].set_title("High-Freq Detail\n(for Method 2)", fontsize=12)
    axes[2, 2].axis("off")

    plt.suptitle(f"Relief Carving Enhancement Comparison: {base_name}",
                 fontsize=16, fontweight='bold', y=0.995)
    plt.tight_layout()

    comparison_path = output_dir / f"{base_name}_carving_comparison.png"
    plt.savefig(comparison_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved comparison: {comparison_path.name}")

    # Save individual enhanced depth maps
    for i, (enhanced, method_name) in enumerate([
        (enhanced1, "edge_overlay"),
        (enhanced2, "high_freq_detail"),
        (enhanced3, "simple_overlay"),
        (enhanced4, "adaptive_detail")
    ], 1):
        # Save as 16-bit grayscale for better precision
        enhanced_16bit = (enhanced * 65535).astype(np.uint16)
        save_path = output_dir / f"{base_name}_enhanced_{method_name}.png"
        cv2.imwrite(str(save_path), enhanced_16bit)

        # Also save as numpy array
        npy_path = output_dir / f"{base_name}_enhanced_{method_name}.npy"
        np.save(npy_path, enhanced)

    print(f"  Saved 4 enhanced depth maps")

    return enhanced1, enhanced2, enhanced3, enhanced4

if __name__ == "__main__":
    print("="*70)
    print("ZoeDepth Enhancement for Relief Carving")
    print("="*70)

    images_dir = Path("images")
    output_dir = Path("output")

    # Process all images that have depth maps
    image_files = list(images_dir.glob("*.jpg")) + list(images_dir.glob("*.webp")) + list(images_dir.glob("*.png"))

    processed = 0
    for image_path in image_files:
        base_name = image_path.stem
        depth_path = output_dir / f"{base_name}_depth.npy"

        if depth_path.exists():
            enhance_depth_for_carving(image_path, depth_path, output_dir)
            processed += 1
        else:
            print(f"\nSkipping {base_name} - no depth map found")

    print("\n" + "="*70)
    print(f"Processing complete! Enhanced {processed} images.")
    print(f"Output saved to: {output_dir.absolute()}")
    print("="*70)
    print("\nRecommendations:")
    print("  • Method 1 (Edge Overlay): Best for sharp features, good for line art")
    print("  • Method 2 (High-Freq Detail): Best for textured surfaces")
    print("  • Method 3 (Simple Overlay): Quick/simple, moderate detail")
    print("  • Method 4 (Adaptive Detail): Best balance for varied scenes")
    print("\nAll methods saved as 16-bit PNG for CNC software import.")

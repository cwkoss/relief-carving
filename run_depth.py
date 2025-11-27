import warnings
warnings.filterwarnings("ignore")

import torch
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

print("Loading ZoeDepth model...")
model = torch.hub.load(".", "ZoeD_N", source="local", pretrained=True, trust_repo=True)

# Set device
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")
model = model.to(device)
model.eval()

# Load image
print("Loading image: otto.jpg")
image_path = "otto.jpg"
image = Image.open(image_path).convert("RGB")
print(f"Image size: {image.size}")

# Run inference
print("Running depth estimation...")
with torch.no_grad():
    depth = model.infer_pil(image)

print(f"Depth map shape: {depth.shape}")
print(f"Depth range: {depth.min():.2f} to {depth.max():.2f}")

# Save depth map as numpy array
np.save("output/otto_depth.npy", depth)
print("Saved depth array to: output/otto_depth.npy")

# Create visualization
fig, axes = plt.subplots(1, 2, figsize=(15, 5))

# Original image
axes[0].imshow(image)
axes[0].set_title("Original Image")
axes[0].axis("off")

# Depth map
im = axes[1].imshow(depth, cmap="inferno")
axes[1].set_title("Depth Map")
axes[1].axis("off")
plt.colorbar(im, ax=axes[1], fraction=0.046, pad=0.04)

plt.tight_layout()
plt.savefig("output/otto_depth_visualization.png", dpi=150, bbox_inches="tight")
print("Saved visualization to: output/otto_depth_visualization.png")

# Also save just the depth map
plt.figure(figsize=(10, 10))
plt.imshow(depth, cmap="inferno")
plt.axis("off")
plt.savefig("output/otto_depth_map.png", dpi=150, bbox_inches="tight", pad_inches=0)
print("Saved depth map to: output/otto_depth_map.png")

print("\nDone! Check the output folder for results.")

import os
import warnings
# Suppress warnings to focus on functionality
warnings.filterwarnings("ignore")

import torch
import numpy as np
from PIL import Image

print("Testing ZoeDepth installation...")
print(f"PyTorch version: {torch.__version__}")

try:
    # Test if we can import the basic modules
    from zoedepth.models.builder import build_model
    from zoedepth.utils.config import get_config
    from zoedepth.utils.misc import colorize
    print("[OK] ZoeDepth modules imported successfully")

    # Create a dummy RGB image for testing
    dummy_image = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
    pil_image = Image.fromarray(dummy_image)
    print("[OK] Created test image")

    # Try to load via torch hub (most reliable method)
    print("Loading ZoeDepth model via torch hub...")
    model = torch.hub.load("isl-org/ZoeDepth", "ZoeD_N", pretrained=True, trust_repo=True)
    print("[OK] Model loaded successfully!")

    # Test inference
    print("Testing inference...")
    depth = model.infer_pil(pil_image)
    print(f"[OK] Inference successful! Depth shape: {depth.shape}")

    # Test colorization
    colored_depth = colorize(depth)
    print(f"[OK] Colorization successful! Output shape: {colored_depth.shape}")

    print("\nSUCCESS: ZoeDepth installation is working correctly!")
    print("\nYou can now use ZoeDepth for depth estimation. Example usage:")
    print("from PIL import Image")
    print("import torch")
    print("model = torch.hub.load('isl-org/ZoeDepth', 'ZoeD_N', pretrained=True)")
    print("image = Image.open('your_image.jpg').convert('RGB')")
    print("depth = model.infer_pil(image)")

except Exception as e:
    print(f"[ERROR] {e}")
    print("\nThere might be version compatibility issues.")
    print("The repository is no longer actively maintained, which can cause issues with newer PyTorch versions.")
    print("You might want to consider using a different depth estimation model or creating a virtual environment with older package versions.")
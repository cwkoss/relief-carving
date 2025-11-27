"""
Test alpha channel detection on transparent images
"""
import numpy as np
from PIL import Image
import sys

def test_alpha_detection(image_path):
    print(f"\n{'='*60}")
    print(f"Testing: {image_path}")
    print(f"{'='*60}")

    try:
        # Load image
        image_orig = Image.open(image_path)
        print(f"\nOriginal image mode: {image_orig.mode}")
        print(f"Image size: {image_orig.size}")
        print(f"Image info: {image_orig.info}")

        # Check for alpha using current method
        has_alpha = image_orig.mode in ('RGBA', 'LA') or (image_orig.mode == 'P' and 'transparency' in image_orig.info)
        print(f"\nCurrent detection method says has_alpha: {has_alpha}")

        # Try different methods to extract alpha
        if image_orig.mode == 'RGBA':
            print("\nMethod 1: Direct RGBA extraction")
            rgba_array = np.array(image_orig)
            alpha_channel = rgba_array[:, :, 3]
            print(f"  Alpha channel shape: {alpha_channel.shape}")
            print(f"  Alpha range: [{alpha_channel.min()}, {alpha_channel.max()}]")
            print(f"  Unique alpha values: {np.unique(alpha_channel)[:10]}...")  # First 10
            print(f"  Fully transparent (0): {(alpha_channel == 0).sum()} pixels")
            print(f"  Fully opaque (255): {(alpha_channel == 255).sum()} pixels")
            print(f"  Partial transparency: {((alpha_channel > 0) & (alpha_channel < 255)).sum()} pixels")

        if image_orig.mode == 'LA':
            print("\nMethod 2: LA (Luminance + Alpha)")
            la_array = np.array(image_orig)
            alpha_channel = la_array[:, :, 1]
            print(f"  Alpha channel shape: {alpha_channel.shape}")
            print(f"  Alpha range: [{alpha_channel.min()}, {alpha_channel.max()}]")
            print(f"  Fully transparent (0): {(alpha_channel == 0).sum()} pixels")

        if image_orig.mode == 'P' and 'transparency' in image_orig.info:
            print("\nMethod 3: Palette mode with transparency")
            print(f"  Transparency info: {image_orig.info['transparency']}")
            # Convert to RGBA to get alpha
            rgba = image_orig.convert('RGBA')
            rgba_array = np.array(rgba)
            alpha_channel = rgba_array[:, :, 3]
            print(f"  Alpha range after RGBA conversion: [{alpha_channel.min()}, {alpha_channel.max()}]")
            print(f"  Fully transparent (0): {(alpha_channel == 0).sum()} pixels")

        # Try converting to RGBA regardless
        print("\nMethod 4: Force convert to RGBA and check")
        image_rgba = image_orig.convert("RGBA")
        rgba_array = np.array(image_rgba)
        alpha_channel = rgba_array[:, :, 3]
        print(f"  Alpha channel shape: {alpha_channel.shape}")
        print(f"  Alpha range: [{alpha_channel.min()}, {alpha_channel.max()}]")
        print(f"  Fully transparent (0): {(alpha_channel == 0).sum()} pixels")
        print(f"  Fully opaque (255): {(alpha_channel == 255).sum()} pixels")
        print(f"  Partial transparency: {((alpha_channel > 0) & (alpha_channel < 255)).sum()} pixels")

        # Check what getchannel returns
        if hasattr(image_orig, 'getchannel'):
            try:
                alpha_from_getchannel = image_orig.getchannel('A')
                if alpha_from_getchannel is not None:
                    print("\nMethod 5: Using getchannel('A')")
                    alpha_array = np.array(alpha_from_getchannel)
                    print(f"  Alpha range: [{alpha_array.min()}, {alpha_array.max()}]")
                    print(f"  Fully transparent (0): {(alpha_array == 0).sum()} pixels")
            except ValueError as e:
                print(f"\ngetchannel('A') failed: {e}")

    except Exception as e:
        print(f"\nERROR: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    # Test the problematic image
    test_paths = [
        "gui_temp/input.png",  # The most recently uploaded image
        "Image-removebg-preview_29.png"  # If in current directory
    ]

    if len(sys.argv) > 1:
        test_paths = [sys.argv[1]]

    for path in test_paths:
        try:
            test_alpha_detection(path)
        except FileNotFoundError:
            print(f"\nFile not found: {path}")

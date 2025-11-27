"""
Test the full alpha channel flow: load -> Gradio simulation -> save -> reload
"""
import numpy as np
from PIL import Image
import sys
from pathlib import Path

def simulate_gradio_image_loading(image_path, image_mode="RGBA"):
    """Simulate what Gradio does when loading an image with image_mode parameter"""
    print(f"\nSimulating Gradio loading: {image_path}")

    # Load the original image
    original = Image.open(image_path)
    print(f"  Original mode: {original.mode}, size: {original.size}")

    # Gradio with image_mode="RGBA" converts to that mode
    if image_mode:
        converted = original.convert(image_mode)
        print(f"  After Gradio image_mode='{image_mode}': {converted.mode}")
    else:
        converted = original

    return converted

def test_save_logic(pil_image, output_path):
    """Test our save logic from gradio_gui.py"""
    print(f"\nTesting save logic...")
    print(f"  Input PIL image mode: {pil_image.mode}")

    # This is the exact logic from gradio_gui.py
    if isinstance(pil_image, np.ndarray):
        if len(pil_image.shape) == 3 and pil_image.shape[-1] == 4:
            saved_image = Image.fromarray(pil_image, mode='RGBA')
        elif len(pil_image.shape) == 3 and pil_image.shape[-1] == 3:
            saved_image = Image.fromarray(pil_image, mode='RGB')
        else:
            saved_image = Image.fromarray(pil_image)
        saved_image.save(output_path, format='PNG')
    else:
        # PIL image - just save it (PNG format preserves alpha automatically)
        pil_image.save(output_path, format='PNG')

    print(f"  Saved to: {output_path}")

    # Reload and check
    reloaded = Image.open(output_path)
    print(f"  Reloaded mode: {reloaded.mode}")

    if reloaded.mode in ('RGBA', 'LA'):
        alpha_array = np.array(reloaded.getchannel('A'))
        print(f"  Alpha channel stats:")
        print(f"    Range: [{alpha_array.min()}, {alpha_array.max()}]")
        print(f"    Fully transparent (0): {(alpha_array == 0).sum()} pixels")
        print(f"    Fully opaque (255): {(alpha_array == 255).sum()} pixels")
        print(f"    Partial transparency: {((alpha_array > 0) & (alpha_array < 255)).sum()} pixels")

        return (alpha_array == 0).sum() > 0  # Return True if has transparency
    else:
        print(f"  WARNING: No alpha channel in reloaded image!")
        return False

def create_test_image_with_alpha(output_path):
    """Create a test image with known alpha channel"""
    print(f"\nCreating test image with transparency...")

    # Create 100x100 RGBA image
    width, height = 100, 100
    img_array = np.zeros((height, width, 4), dtype=np.uint8)

    # Red square in top-left (fully opaque)
    img_array[0:50, 0:50] = [255, 0, 0, 255]

    # Green square in top-right (50% transparent)
    img_array[0:50, 50:100] = [0, 255, 0, 128]

    # Blue square in bottom-left (fully transparent)
    img_array[50:100, 0:50] = [0, 0, 255, 0]

    # White square in bottom-right (fully opaque)
    img_array[50:100, 50:100] = [255, 255, 255, 255]

    test_img = Image.fromarray(img_array, mode='RGBA')
    test_img.save(output_path, format='PNG')

    print(f"  Created test image: {output_path}")
    print(f"  Expected: 2500 fully transparent pixels (bottom-left square)")

    return output_path

def main():
    print("="*70)
    print("Alpha Channel Preservation Test")
    print("="*70)

    # Test 1: Create our own test image
    test_img_path = Path("test_transparent.png")
    create_test_image_with_alpha(test_img_path)

    # Simulate Gradio loading it
    gradio_image = simulate_gradio_image_loading(test_img_path, image_mode="RGBA")

    # Test our save logic
    output_path = Path("test_output.png")
    has_transparency = test_save_logic(gradio_image, output_path)

    if has_transparency:
        print("\n✓ SUCCESS: Alpha channel preserved correctly!")
    else:
        print("\n✗ FAILURE: Alpha channel lost!")

    # Test 2: Try with the actual uploaded image if it exists
    print("\n" + "="*70)
    uploaded_img = Path("gui_temp/input.png")
    if uploaded_img.exists():
        print("Testing actual uploaded image from GUI...")
        gradio_image = simulate_gradio_image_loading(uploaded_img, image_mode="RGBA")
        output_path2 = Path("test_uploaded_reprocess.png")
        test_save_logic(gradio_image, output_path2)
    else:
        print("No uploaded image found in gui_temp/input.png")

    # Test 3: Try with user's actual file if provided
    if len(sys.argv) > 1:
        user_file = Path(sys.argv[1])
        if user_file.exists():
            print("\n" + "="*70)
            print(f"Testing user-provided file: {user_file}")
            gradio_image = simulate_gradio_image_loading(user_file, image_mode="RGBA")
            output_path3 = Path("test_user_file.png")
            has_transparency = test_save_logic(gradio_image, output_path3)

            if has_transparency:
                print("\n✓ User's image has transparency and it was preserved!")
            else:
                print("\n⚠ User's image either has no transparency or it was lost")

                # Double check the original
                original = Image.open(user_file)
                print(f"\nDouble-checking original file:")
                print(f"  Mode: {original.mode}")
                if original.mode == 'RGBA':
                    alpha_orig = np.array(original.getchannel('A'))
                    print(f"  Transparent pixels in original: {(alpha_orig == 0).sum()}")

if __name__ == "__main__":
    main()

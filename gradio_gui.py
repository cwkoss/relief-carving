"""
Relief Carving GUI - Gradio Interface
Processes images for CNC relief carving using ZoeDepth + CLAHE + Bilateral filtering
"""

import gradio as gr
import numpy as np
from PIL import Image
import cv2
from pathlib import Path
import subprocess
import time
import atexit
import sys

def cleanup_docker():
    """Cleanup Docker containers on exit"""
    print("\nCleaning up Docker containers...")
    subprocess.run(["docker-compose", "down"], capture_output=True)

# Register cleanup function
atexit.register(cleanup_docker)

def run_depth_estimation(input_image, progress=gr.Progress()):
    """Run ZoeDepth on uploaded image - this is the slow step"""
    if input_image is None:
        return "Please upload an image", False, ""

    try:
        progress(0, desc="Saving image...")

        # Create gui_temp directory
        temp_dir = Path("gui_temp")
        temp_dir.mkdir(exist_ok=True)

        # Save input image (use PNG to preserve alpha channel if present)
        input_path = temp_dir / "input.png"
        if isinstance(input_image, np.ndarray):
            # Check if array has alpha channel (4 channels)
            if len(input_image.shape) == 3 and input_image.shape[-1] == 4:
                pil_image = Image.fromarray(input_image, mode='RGBA')
            elif len(input_image.shape) == 3 and input_image.shape[-1] == 3:
                pil_image = Image.fromarray(input_image, mode='RGB')
            else:
                pil_image = Image.fromarray(input_image)
            pil_image.save(input_path, format='PNG')
        else:
            # PIL image - just save it (PNG format preserves alpha automatically)
            input_image.save(input_path, format='PNG')

        progress(0.1, desc="Running ZoeDepth (this takes ~30s)...")

        # Use forward slashes for Docker (Linux paths)
        docker_input = "gui_temp/input.png"
        docker_depth = "gui_temp/depth.npy"
        docker_alpha = "gui_temp/alpha_mask.npy"

        # Run depth estimation in Docker
        result = subprocess.run(
            ["docker-compose", "run", "--rm", "zoedepth",
             "python", "run_depth_only.py",
             docker_input, docker_depth, docker_alpha],
            capture_output=True,
            text=True,
            timeout=180
        )

        console_output = f"=== DEPTH ESTIMATION LOG ===\n{result.stdout}\n"
        if result.stderr:
            console_output += f"\nSTDERR:\n{result.stderr}"

        if result.returncode != 0:
            return f"❌ Error during depth estimation:\n{result.stderr}", False, console_output

        progress(1.0, desc="Depth map cached!")

        return "✓ Depth map cached! Now adjust sliders and click 'Apply Processing' for fast results.", True, console_output

    except subprocess.TimeoutExpired:
        return "❌ Error: Depth estimation timed out (>3 minutes)", False, ""
    except Exception as e:
        return f"❌ Error: {str(e)}", False, ""

def apply_processing(detail_strength, clahe_clip, clahe_tile, bilateral_d, bilateral_color, bilateral_space, shadow_crush, depth_cached, progress=gr.Progress()):
    """Apply post-processing to cached depth map - this is FAST!"""

    if not depth_cached:
        return None, "❌ Please upload an image first to run depth estimation", None, None, None, None, None, None, None, None, ""

    # Check if depth map exists
    depth_path = Path("gui_temp") / "depth.npy"
    if not depth_path.exists():
        return None, "❌ Depth map not found. Please upload an image first.", None, None, None, None, None, None, None, None, ""

    try:
        progress(0, desc="Applying post-processing...")

        temp_dir = Path("gui_temp")
        output_path = temp_dir / "output.png"
        preview_path = temp_dir / "output_preview.png"

        # Use forward slashes for Docker (Linux paths)
        docker_input = "gui_temp/input.png"
        docker_depth = "gui_temp/depth.npy"
        docker_output = "gui_temp/output.png"

        progress(0.3, desc="Running CLAHE and bilateral filter...")

        # Run post-processing in Docker (fast - no depth estimation!)
        result = subprocess.run(
            ["docker-compose", "run", "--rm", "zoedepth",
             "python", "apply_postprocessing.py",
             docker_input, docker_depth, docker_output,
             str(detail_strength), str(clahe_clip), str(clahe_tile),
             str(bilateral_d), str(bilateral_color), str(bilateral_space),
             str(shadow_crush)],
            capture_output=True,
            text=True,
            timeout=30  # Much shorter timeout since no depth estimation
        )

        console_output = f"=== POST-PROCESSING LOG ===\n{result.stdout}\n"
        if result.stderr:
            console_output += f"\nSTDERR:\n{result.stderr}"

        if result.returncode != 0:
            return None, f"❌ Error during post-processing:\n{result.stderr}", None, None, None, None, None, None, None, console_output

        progress(0.8, desc="Loading result...")

        # Load the preview image
        if not preview_path.exists():
            return None, f"❌ Error: Output file not generated\n{result.stdout}\n{result.stderr}", None, None, None, None, None, None, None, console_output

        output_display = np.array(Image.open(preview_path))

        # Copy 16-bit version to output folder
        final_output = Path("output") / "latest_carving.png"
        final_output.parent.mkdir(exist_ok=True)

        import shutil
        shutil.copy(output_path, final_output)

        # Load debug images
        debug_dir = temp_dir / "debug"
        debug_images = {}
        debug_files = [
            ("01_depth_raw.png", "1. Raw ZoeDepth"),
            ("02_depth_raw_masked.png", "2. ZoeDepth + Alpha Mask"),
            ("03_depth_inverted.png", "3. Inverted (for relief)"),
            ("04_detail_added.png", "4. High-Freq Detail Added"),
            ("05_clahe.png", "5. CLAHE Applied"),
            ("06_bilateral.png", "6. Bilateral Filter"),
            ("07_shadow_crush.png", "7. Shadow Crush"),
            ("08_final_masked.png", "8. Final (Re-masked)")
        ]

        for filename, label in debug_files:
            debug_path = debug_dir / filename
            if debug_path.exists():
                debug_images[label] = np.array(Image.open(debug_path))
            else:
                debug_images[label] = None

        progress(1.0, desc="Complete!")

        return (
            output_display,
            f"✓ Processing complete!\n16-bit depth map saved to: {final_output}",
            debug_images.get("1. Raw ZoeDepth"),
            debug_images.get("2. ZoeDepth + Alpha Mask"),
            debug_images.get("3. Inverted (for relief)"),
            debug_images.get("4. High-Freq Detail Added"),
            debug_images.get("5. CLAHE Applied"),
            debug_images.get("6. Bilateral Filter"),
            debug_images.get("7. Shadow Crush"),
            debug_images.get("8. Final (Re-masked)"),
            console_output
        )

    except subprocess.TimeoutExpired:
        return None, "❌ Error: Post-processing timed out (>30 seconds)", None, None, None, None, None, None, None, None, ""
    except Exception as e:
        return None, f"❌ Error: {str(e)}", None, None, None, None, None, None, None, None, ""

def initialize_system():
    """Initialize Docker on startup"""
    return "Ready! Upload an image to begin..."

# Build Gradio Interface
with gr.Blocks(title="Relief Carving Depth Map Generator") as demo:
    gr.Markdown("""
    # Relief Carving Depth Map Generator

    Upload an image to generate a CNC-ready depth map for relief carving.

    **Pipeline:** ZoeDepth → High-Freq Detail → Inverted Depth → CLAHE → Bilateral Filter → 16-bit PNG
    """)

    # State to track if depth is cached
    depth_cached = gr.State(False)

    with gr.Row():
        with gr.Column():
            input_image = gr.Image(type="pil", label="Input Image (ZoeDepth runs automatically)", format="png", image_mode="RGBA")
            depth_status = gr.Textbox(label="Depth Status", lines=2, interactive=False)

            gr.Markdown("### Processing Parameters")
            gr.Markdown("*Adjust these and click 'Apply Processing' for instant results*")

            with gr.Accordion("High-Frequency Detail", open=True):
                detail_strength = gr.Slider(
                    minimum=0.0, maximum=1.0, value=0.2, step=0.05,
                    label="Detail Strength",
                    info="How much surface texture to add from the original image"
                )

            with gr.Accordion("CLAHE (Contrast Enhancement)", open=True):
                clahe_clip = gr.Slider(
                    minimum=1.0, maximum=5.0, value=2.5, step=0.1,
                    label="Clip Limit",
                    info="Higher = more contrast (but more artifacts)"
                )
                clahe_tile = gr.Slider(
                    minimum=8, maximum=64, value=16, step=8,
                    label="Tile Size",
                    info="Larger = smoother transitions (less grid artifacts)"
                )

            with gr.Accordion("Bilateral Filter (Smoothing)", open=True):
                bilateral_d = gr.Slider(
                    minimum=5, maximum=15, value=9, step=2,
                    label="Filter Diameter",
                    info="Size of smoothing filter"
                )
                bilateral_color = gr.Slider(
                    minimum=10, maximum=150, value=75, step=5,
                    label="Color Sigma",
                    info="Higher = more aggressive smoothing"
                )
                bilateral_space = gr.Slider(
                    minimum=10, maximum=150, value=75, step=5,
                    label="Space Sigma",
                    info="Higher = larger smoothing area"
                )

            with gr.Accordion("Shadow Crush (Highlight Detail)", open=True):
                shadow_crush = gr.Slider(
                    minimum=0.0, maximum=0.5, value=0.0, step=0.05,
                    label="Crush Amount",
                    info="Wash out dark greys to increase detail in highlights (0=none, 0.3=heavy)"
                )

            process_btn = gr.Button("Apply Processing (Fast!)", variant="primary", size="lg")

        with gr.Column():
            output_image = gr.Image(type="numpy", label="Output Depth Map (Preview)", format="png")
            output_status = gr.Textbox(label="Output Status", lines=3)
            console_log = gr.Textbox(label="Docker Console Log", lines=10, interactive=False, max_lines=20)

    with gr.Accordion("Parameter Guide", open=False):
        gr.Markdown("""
        ### Parameter Recommendations

        **Detail Strength (0.2 recommended)**
        - 0.0 = No surface texture, smooth depth map only
        - 0.2 = Good balance (recommended for most images)
        - 0.5+ = Heavy texture, may look too busy

        **CLAHE Clip Limit (2.5 recommended)**
        - 1.0-2.0 = Subtle contrast, very smooth
        - 2.5 = Good balance (recommended)
        - 3.0+ = Strong contrast, more artifacts

        **CLAHE Tile Size (16 recommended)**
        - 8 = Fine detail but visible grid artifacts
        - 16 = Good balance (recommended)
        - 32+ = Smooth but less local contrast

        **Bilateral Filter (9, 75, 75 recommended)**
        - Lower values = preserve more detail but keep artifacts
        - Higher values = smoother but may lose fine detail
        - These defaults remove spiky texture while preserving edges

        **Shadow Crush (0.0 recommended, adjust as needed)**
        - 0.0 = No crushing (default)
        - 0.1-0.2 = Subtle - slightly more highlight detail
        - 0.3+ = Aggressive - washes out shadows, maximum highlight detail
        - Use this to increase dynamic range for fine details near white
        - Dark greys below the crush amount become pure black
        """)

    gr.Markdown("""
    ### How It Works:
    1. **Upload image** - ZoeDepth runs automatically (~30s, cached)
    2. **Adjust sliders** - Tune the parameters to your liking
    3. **Click 'Apply Processing'** - Near-instant results (<5s)
    4. **Iterate** - Adjust and reprocess as many times as you want!

    ### Notes:
    - First run downloads model (~1.3GB, cached)
    - Depth estimation is slow but only runs once per image
    - Post-processing is fast - adjust sliders freely!
    - Output saved as 16-bit PNG in `output/latest_carving.png`
    - Closer objects appear lighter (raised relief)
    - **Transparent backgrounds supported!** Transparent areas → 100% black (no carving)
    """)

    # Debug visualization
    with gr.Accordion("🔍 Debug: Processing Pipeline Visualization", open=False):
        gr.Markdown("*View intermediate steps to troubleshoot alpha masking and processing*")
        with gr.Row():
            debug_img1 = gr.Image(type="numpy", label="1. Raw ZoeDepth", format="png")
            debug_img2 = gr.Image(type="numpy", label="2. ZoeDepth + Alpha Mask", format="png")
            debug_img3 = gr.Image(type="numpy", label="3. Inverted (for relief)", format="png")
        with gr.Row():
            debug_img4 = gr.Image(type="numpy", label="4. High-Freq Detail Added", format="png")
            debug_img5 = gr.Image(type="numpy", label="5. CLAHE Applied", format="png")
            debug_img6 = gr.Image(type="numpy", label="6. Bilateral Filter", format="png")
        with gr.Row():
            debug_img7 = gr.Image(type="numpy", label="7. Shadow Crush", format="png")
            debug_img8 = gr.Image(type="numpy", label="8. Final (Re-masked)", format="png")

    # Event handlers

    # Run depth estimation when image is uploaded
    input_image.change(
        fn=run_depth_estimation,
        inputs=[input_image],
        outputs=[depth_status, depth_cached, console_log]
    )

    # Apply post-processing when button clicked
    process_btn.click(
        fn=apply_processing,
        inputs=[detail_strength, clahe_clip, clahe_tile, bilateral_d, bilateral_color, bilateral_space, shadow_crush, depth_cached],
        outputs=[output_image, output_status, debug_img1, debug_img2, debug_img3, debug_img4, debug_img5, debug_img6, debug_img7, debug_img8, console_log]
    )

    # Initialize on load
    demo.load(
        fn=initialize_system,
        outputs=[depth_status]
    )

if __name__ == "__main__":
    print("="*70)
    print("Relief Carving Depth Map Generator")
    print("="*70)
    print("\nInitializing system...")

    try:
        demo.launch(
            server_name="127.0.0.1",
            server_port=7860,
            share=False,
            inbrowser=True
        )
    except KeyboardInterrupt:
        print("\nShutting down...")
        cleanup_docker()
        sys.exit(0)

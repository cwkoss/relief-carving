"""
Relief Carving GUI - Stepwise Processing Interface
Process images step-by-step with full control over each stage
"""

import gradio as gr
import numpy as np
from PIL import Image
import cv2
from pathlib import Path
import subprocess
import atexit
import sys
import json

def cleanup_docker():
    """Cleanup Docker containers on exit"""
    print("\nCleaning up Docker containers...")
    subprocess.run(["docker-compose", "down"], capture_output=True)

# Register cleanup function
atexit.register(cleanup_docker)

# Processing functions for each step
def step_1_depth_estimation(input_image, progress=gr.Progress()):
    """Step 1: Run ZoeDepth depth estimation"""
    if input_image is None:
        return None, "Please upload an image first", {}, ""

    try:
        progress(0, desc="Saving image...")

        temp_dir = Path("gui_temp")
        temp_dir.mkdir(exist_ok=True)

        # Save input image
        input_path = temp_dir / "input.png"
        if isinstance(input_image, np.ndarray):
            if len(input_image.shape) == 3 and input_image.shape[-1] == 4:
                pil_image = Image.fromarray(input_image, mode='RGBA')
            elif len(input_image.shape) == 3 and input_image.shape[-1] == 3:
                pil_image = Image.fromarray(input_image, mode='RGB')
            else:
                pil_image = Image.fromarray(input_image)
            pil_image.save(input_path, format='PNG')
        else:
            input_image.save(input_path, format='PNG')

        progress(0.1, desc="Running ZoeDepth (~30s)...")

        # Run depth estimation
        result = subprocess.run(
            ["docker-compose", "run", "--rm", "zoedepth",
             "python", "run_depth_only.py",
             "gui_temp/input.png", "gui_temp/step1_depth.npy", "gui_temp/alpha_mask.npy"],
            capture_output=True,
            text=True,
            timeout=180
        )

        console = f"=== STEP 1: DEPTH ESTIMATION ===\n{result.stdout}"

        if result.returncode != 0:
            return None, f"Error: {result.stderr}", {}, console

        # Load and display depth
        depth = np.load(temp_dir / "step1_depth.npy")
        depth_norm = (depth - depth.min()) / (depth.max() - depth.min())
        depth_img = (depth_norm * 255).astype(np.uint8)

        progress(1.0, desc="Complete!")

        state = {"step": 1, "depth": depth.tolist()}
        return depth_img, "✓ Depth estimation complete", state, console

    except Exception as e:
        return None, f"Error: {str(e)}", {}, ""

def step_2_apply_alpha_mask(state, progress=gr.Progress()):
    """Step 2: Apply alpha mask (transparent areas → white)"""
    if not state or state.get("step", 0) < 1:
        return None, "Please run Step 1 first", state, ""

    try:
        temp_dir = Path("gui_temp")

        # Load depth
        depth = np.array(state["depth"])
        depth_norm = (depth - depth.min()) / (depth.max() - depth.min())

        # Load alpha mask
        alpha_path = temp_dir / "alpha_mask.npy"
        if alpha_path.exists():
            alpha_mask = np.load(alpha_path)
            # Set transparent areas to WHITE (1.0) before inversion
            depth_masked = depth_norm * alpha_mask + (1.0 - alpha_mask)
            console = f"Alpha mask applied: {((alpha_mask == 0).sum())} transparent pixels set to WHITE"
        else:
            depth_masked = depth_norm
            console = "No alpha mask found"

        # Save and display
        depth_img = (depth_masked * 255).astype(np.uint8)
        np.save(temp_dir / "step2_masked.npy", depth_masked)

        state["step"] = 2
        state["masked"] = depth_masked.tolist()

        return depth_img, "✓ Alpha mask applied", state, console

    except Exception as e:
        return None, f"Error: {str(e)}", state, ""

def step_3_invert(state, progress=gr.Progress()):
    """Step 3: Invert depth (for relief carving)"""
    if not state or state.get("step", 0) < 2:
        return None, "Please run Step 2 first", state, ""

    try:
        temp_dir = Path("gui_temp")

        # Load masked depth
        depth_masked = np.array(state["masked"])

        # Invert
        depth_inverted = 1.0 - depth_masked

        # Save and display
        depth_img = (depth_inverted * 255).astype(np.uint8)
        np.save(temp_dir / "step3_inverted.npy", depth_inverted)

        state["step"] = 3
        state["inverted"] = depth_inverted.tolist()

        return depth_img, "✓ Depth inverted", state, "Inverted: closer = darker (will be raised)"

    except Exception as e:
        return None, f"Error: {str(e)}", state, ""

def step_4_high_freq_detail(detail_strength, crush_amount, state, progress=gr.Progress()):
    """Step 4: Add high-frequency detail from original image"""
    if not state or state.get("step", 0) < 3:
        return None, "Please run Step 3 first", state, ""

    try:
        temp_dir = Path("gui_temp")

        # Load inverted depth
        depth_inverted = np.array(state["inverted"])

        # Load original image and extract high-freq detail
        image = Image.open(temp_dir / "input.png").convert("RGB")
        gray = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2GRAY)
        gray_norm = gray.astype(float) / 255.0
        gray_blur = cv2.GaussianBlur(gray_norm, (21, 21), 0)
        high_freq = gray_norm - gray_blur

        # Add detail
        enhanced = depth_inverted + (high_freq * detail_strength)
        enhanced = (enhanced - enhanced.min()) / (enhanced.max() - enhanced.min())

        # Re-mask with alpha
        alpha_path = temp_dir / "alpha_mask.npy"
        if alpha_path.exists():
            alpha_mask = np.load(alpha_path)
            enhanced = enhanced * alpha_mask

        # Apply crush if requested
        if crush_amount > 0:
            enhanced = np.clip((enhanced - crush_amount) / (1.0 - crush_amount), 0, 1)
            console = f"Detail added (strength={detail_strength}), crushed (amount={crush_amount})"
        else:
            console = f"Detail added (strength={detail_strength})"

        # Save and display
        depth_img = (enhanced * 255).astype(np.uint8)
        np.save(temp_dir / "step4_detail.npy", enhanced)

        state["step"] = 4
        state["detail"] = enhanced.tolist()

        return depth_img, f"✓ High-freq detail added", state, console

    except Exception as e:
        return None, f"Error: {str(e)}", state, ""

def step_5_clahe(clahe_clip, clahe_tile, crush_amount, state, progress=gr.Progress()):
    """Step 5: Apply CLAHE contrast enhancement"""
    if not state or state.get("step", 0) < 4:
        return None, "Please run Step 4 first", state, ""

    try:
        temp_dir = Path("gui_temp")

        # Load detail-enhanced depth
        depth = np.array(state["detail"])

        # Apply CLAHE
        depth_8bit = (depth * 255).astype(np.uint8)
        clahe = cv2.createCLAHE(clipLimit=clahe_clip, tileGridSize=(clahe_tile, clahe_tile))
        enhanced = clahe.apply(depth_8bit).astype(float) / 255.0

        # Re-mask with alpha
        alpha_path = temp_dir / "alpha_mask.npy"
        if alpha_path.exists():
            alpha_mask = np.load(alpha_path)
            enhanced = enhanced * alpha_mask

        # Apply crush if requested
        if crush_amount > 0:
            enhanced = np.clip((enhanced - crush_amount) / (1.0 - crush_amount), 0, 1)
            console = f"CLAHE applied (clip={clahe_clip}, tile={clahe_tile}), crushed (amount={crush_amount})"
        else:
            console = f"CLAHE applied (clip={clahe_clip}, tile={clahe_tile})"

        # Save and display
        depth_img = (enhanced * 255).astype(np.uint8)
        np.save(temp_dir / "step5_clahe.npy", enhanced)

        state["step"] = 5
        state["clahe"] = enhanced.tolist()

        return depth_img, f"✓ CLAHE applied", state, console

    except Exception as e:
        return None, f"Error: {str(e)}", state, ""

def step_6_bilateral(bilateral_d, bilateral_color, bilateral_space, crush_amount, state, progress=gr.Progress()):
    """Step 6: Apply bilateral filter"""
    if not state or state.get("step", 0) < 5:
        return None, "Please run Step 5 first", state, ""

    try:
        temp_dir = Path("gui_temp")

        # Load CLAHE-enhanced depth
        depth = np.array(state["clahe"])

        # Apply bilateral filter
        depth_8bit = (depth * 255).astype(np.uint8)
        filtered = cv2.bilateralFilter(depth_8bit, bilateral_d, bilateral_color, bilateral_space)
        enhanced = filtered.astype(float) / 255.0

        # Re-mask with alpha
        alpha_path = temp_dir / "alpha_mask.npy"
        if alpha_path.exists():
            alpha_mask = np.load(alpha_path)
            enhanced = enhanced * alpha_mask

        # Apply crush if requested
        if crush_amount > 0:
            enhanced = np.clip((enhanced - crush_amount) / (1.0 - crush_amount), 0, 1)
            console = f"Bilateral applied (d={bilateral_d}, color={bilateral_color}, space={bilateral_space}), crushed (amount={crush_amount})"
        else:
            console = f"Bilateral applied (d={bilateral_d}, color={bilateral_color}, space={bilateral_space})"

        # Save and display
        depth_img = (enhanced * 255).astype(np.uint8)
        np.save(temp_dir / "step6_bilateral.npy", enhanced)

        state["step"] = 6
        state["bilateral"] = enhanced.tolist()

        return depth_img, f"✓ Bilateral filter applied", state, console

    except Exception as e:
        return None, f"Error: {str(e)}", state, ""

def step_7_final_crush(crush_amount, state, progress=gr.Progress()):
    """Step 7: Final shadow crush"""
    if not state or state.get("step", 0) < 6:
        return None, "Please run Step 6 first", state, ""

    try:
        temp_dir = Path("gui_temp")

        # Load bilateral-filtered depth
        depth = np.array(state["bilateral"])

        # Apply crush
        if crush_amount > 0:
            enhanced = np.clip((depth - crush_amount) / (1.0 - crush_amount), 0, 1)
            console = f"Final crush applied (amount={crush_amount})"
        else:
            enhanced = depth
            console = "No final crush (amount=0)"

        # Re-mask with alpha
        alpha_path = temp_dir / "alpha_mask.npy"
        if alpha_path.exists():
            alpha_mask = np.load(alpha_path)
            enhanced = enhanced * alpha_mask

        # Save and display
        depth_img = (enhanced * 255).astype(np.uint8)
        np.save(temp_dir / "step7_crush.npy", enhanced)

        state["step"] = 7
        state["final"] = enhanced.tolist()

        return depth_img, f"✓ Final crush applied", state, console

    except Exception as e:
        return None, f"Error: {str(e)}", state, ""

def step_8_save_output(state, progress=gr.Progress()):
    """Step 8: Save final output"""
    if not state or state.get("step", 0) < 7:
        return "Please run Step 7 first", ""

    try:
        temp_dir = Path("gui_temp")
        output_dir = Path("output")
        output_dir.mkdir(exist_ok=True)

        # Load final depth
        depth = np.array(state["final"])

        # Save 16-bit PNG
        output_16bit = (depth * 65535).astype(np.uint16)
        output_path = output_dir / "latest_carving.png"
        cv2.imwrite(str(output_path), output_16bit)

        # Save 8-bit preview
        output_8bit = (depth * 255).astype(np.uint8)
        preview_path = output_dir / "latest_carving_preview.png"
        cv2.imwrite(str(preview_path), output_8bit)

        return f"✓ Saved to {output_path}", f"16-bit: {output_path}\n8-bit preview: {preview_path}"

    except Exception as e:
        return f"Error: {str(e)}", ""

def run_all_steps(input_image, detail_strength, step4_crush, clahe_clip, clahe_tile, step5_crush,
                  bilateral_d, bilateral_color, bilateral_space, step6_crush, step7_crush, progress=gr.Progress()):
    """Run all steps sequentially with current settings"""

    state = {}
    console_log = ""

    # Step 1
    progress(0/8, desc="Step 1: Depth Estimation...")
    img1, status1, state, log1 = step_1_depth_estimation(input_image, progress)
    console_log += log1 + "\n\n"
    if img1 is None:
        return None, None, None, None, None, None, None, status1, state, console_log

    # Step 2
    progress(1/8, desc="Step 2: Alpha Mask...")
    img2, status2, state, log2 = step_2_apply_alpha_mask(state, progress)
    console_log += log2 + "\n\n"

    # Step 3
    progress(2/8, desc="Step 3: Invert...")
    img3, status3, state, log3 = step_3_invert(state, progress)
    console_log += log3 + "\n\n"

    # Step 4
    progress(3/8, desc="Step 4: High-Freq Detail...")
    img4, status4, state, log4 = step_4_high_freq_detail(detail_strength, step4_crush, state, progress)
    console_log += log4 + "\n\n"

    # Step 5
    progress(4/8, desc="Step 5: CLAHE...")
    img5, status5, state, log5 = step_5_clahe(clahe_clip, clahe_tile, step5_crush, state, progress)
    console_log += log5 + "\n\n"

    # Step 6
    progress(5/8, desc="Step 6: Bilateral...")
    img6, status6, state, log6 = step_6_bilateral(bilateral_d, bilateral_color, bilateral_space, step6_crush, state, progress)
    console_log += log6 + "\n\n"

    # Step 7
    progress(6/8, desc="Step 7: Final Crush...")
    img7, status7, state, log7 = step_7_final_crush(step7_crush, state, progress)
    console_log += log7 + "\n\n"

    # Step 8
    progress(7/8, desc="Step 8: Saving...")
    save_status, save_log = step_8_save_output(state, progress)
    console_log += save_log + "\n\n"

    progress(1.0, desc="Complete!")

    return (img1, img2, img3, img4, img5, img6, img7,
            "✓ All steps complete! Output saved.", state, console_log)

# Build Gradio Interface
with gr.Blocks(title="Relief Carving - Stepwise Processing") as demo:
    gr.Markdown("""
    # Relief Carving Depth Map Generator (Stepwise Mode)

    Process your image step-by-step with full control over each stage.
    Each step shows its output and can be run independently or all at once.
    """)

    # State to track progress and intermediate results
    pipeline_state = gr.State({})

    # Input
    with gr.Row():
        input_image = gr.Image(type="pil", label="Input Image", format="png", image_mode="RGBA")
        console_output = gr.Textbox(label="Console Log", lines=20, interactive=False)

    gr.Markdown("---")

    # Step 1: Depth Estimation
    with gr.Row():
        with gr.Column(scale=1):
            gr.Markdown("### Step 1: Depth Estimation")
            step1_btn = gr.Button("▶ Run Depth Estimation (~30s)", variant="primary")
            step1_status = gr.Textbox(label="Status", lines=1, interactive=False)
        with gr.Column(scale=2):
            step1_output = gr.Image(type="numpy", label="1. Raw ZoeDepth Output", format="png")

    # Step 2: Alpha Mask
    with gr.Row():
        with gr.Column(scale=1):
            gr.Markdown("### Step 2: Apply Alpha Mask")
            gr.Markdown("*Transparent areas → white (will be black after inversion)*")
            step2_btn = gr.Button("▶ Apply Alpha Mask", variant="secondary")
            step2_status = gr.Textbox(label="Status", lines=1, interactive=False)
        with gr.Column(scale=2):
            step2_output = gr.Image(type="numpy", label="2. Alpha Masked (transparent=white)", format="png")

    # Step 3: Invert
    with gr.Row():
        with gr.Column(scale=1):
            gr.Markdown("### Step 3: Invert Depth")
            gr.Markdown("*For relief carving: closer = darker = raised*")
            step3_btn = gr.Button("▶ Invert Depth", variant="secondary")
            step3_status = gr.Textbox(label="Status", lines=1, interactive=False)
        with gr.Column(scale=2):
            step3_output = gr.Image(type="numpy", label="3. Inverted (transparent=black)", format="png")

    # Step 4: High-Freq Detail
    with gr.Row():
        with gr.Column(scale=1):
            gr.Markdown("### Step 4: Add Surface Detail")
            detail_strength = gr.Slider(0.0, 1.0, 0.2, step=0.05, label="Detail Strength")
            step4_crush = gr.Slider(0.0, 0.5, 0.0, step=0.05, label="Crush Amount (optional)")
            step4_btn = gr.Button("▶ Add Detail", variant="secondary")
            step4_status = gr.Textbox(label="Status", lines=1, interactive=False)
        with gr.Column(scale=2):
            step4_output = gr.Image(type="numpy", label="4. With High-Freq Detail", format="png")

    # Step 5: CLAHE
    with gr.Row():
        with gr.Column(scale=1):
            gr.Markdown("### Step 5: CLAHE Contrast")
            clahe_clip = gr.Slider(1.0, 5.0, 2.5, step=0.1, label="Clip Limit")
            clahe_tile = gr.Slider(8, 64, 16, step=8, label="Tile Size")
            step5_crush = gr.Slider(0.0, 0.5, 0.0, step=0.05, label="Crush Amount (optional)")
            step5_btn = gr.Button("▶ Apply CLAHE", variant="secondary")
            step5_status = gr.Textbox(label="Status", lines=1, interactive=False)
        with gr.Column(scale=2):
            step5_output = gr.Image(type="numpy", label="5. CLAHE Enhanced", format="png")

    # Step 6: Bilateral Filter
    with gr.Row():
        with gr.Column(scale=1):
            gr.Markdown("### Step 6: Bilateral Smoothing")
            bilateral_d = gr.Slider(5, 15, 9, step=2, label="Filter Diameter")
            bilateral_color = gr.Slider(10, 150, 75, step=5, label="Color Sigma")
            bilateral_space = gr.Slider(10, 150, 75, step=5, label="Space Sigma")
            step6_crush = gr.Slider(0.0, 0.5, 0.0, step=0.05, label="Crush Amount (optional)")
            step6_btn = gr.Button("▶ Apply Bilateral", variant="secondary")
            step6_status = gr.Textbox(label="Status", lines=1, interactive=False)
        with gr.Column(scale=2):
            step6_output = gr.Image(type="numpy", label="6. Bilateral Filtered", format="png")

    # Step 7: Final Crush
    with gr.Row():
        with gr.Column(scale=1):
            gr.Markdown("### Step 7: Final Shadow Crush")
            step7_crush = gr.Slider(0.0, 0.5, 0.0, step=0.05, label="Final Crush Amount")
            step7_btn = gr.Button("▶ Apply Final Crush", variant="secondary")
            step7_status = gr.Textbox(label="Status", lines=1, interactive=False)
        with gr.Column(scale=2):
            step7_output = gr.Image(type="numpy", label="7. Final Crushed", format="png")

    # Step 8: Save
    gr.Markdown("---")
    with gr.Row():
        step8_btn = gr.Button("💾 Save Output (16-bit PNG)", variant="primary", size="lg")
        step8_status = gr.Textbox(label="Save Status", lines=2, interactive=False)

    # Global controls
    gr.Markdown("---")
    with gr.Row():
        run_all_btn = gr.Button("⚡ Run All Steps (with current settings)", variant="primary", size="lg")

    # Event handlers
    step1_btn.click(
        fn=step_1_depth_estimation,
        inputs=[input_image],
        outputs=[step1_output, step1_status, pipeline_state, console_output]
    )

    step2_btn.click(
        fn=step_2_apply_alpha_mask,
        inputs=[pipeline_state],
        outputs=[step2_output, step2_status, pipeline_state, console_output]
    )

    step3_btn.click(
        fn=step_3_invert,
        inputs=[pipeline_state],
        outputs=[step3_output, step3_status, pipeline_state, console_output]
    )

    step4_btn.click(
        fn=step_4_high_freq_detail,
        inputs=[detail_strength, step4_crush, pipeline_state],
        outputs=[step4_output, step4_status, pipeline_state, console_output]
    )

    step5_btn.click(
        fn=step_5_clahe,
        inputs=[clahe_clip, clahe_tile, step5_crush, pipeline_state],
        outputs=[step5_output, step5_status, pipeline_state, console_output]
    )

    step6_btn.click(
        fn=step_6_bilateral,
        inputs=[bilateral_d, bilateral_color, bilateral_space, step6_crush, pipeline_state],
        outputs=[step6_output, step6_status, pipeline_state, console_output]
    )

    step7_btn.click(
        fn=step_7_final_crush,
        inputs=[step7_crush, pipeline_state],
        outputs=[step7_output, step7_status, pipeline_state, console_output]
    )

    step8_btn.click(
        fn=step_8_save_output,
        inputs=[pipeline_state],
        outputs=[step8_status, console_output]
    )

    run_all_btn.click(
        fn=run_all_steps,
        inputs=[
            input_image, detail_strength, step4_crush,
            clahe_clip, clahe_tile, step5_crush,
            bilateral_d, bilateral_color, bilateral_space, step6_crush,
            step7_crush
        ],
        outputs=[
            step1_output, step2_output, step3_output, step4_output,
            step5_output, step6_output, step7_output,
            step8_status, pipeline_state, console_output
        ]
    )

if __name__ == "__main__":
    print("="*70)
    print("Relief Carving - Stepwise Processing Mode")
    print("="*70)

    try:
        demo.launch(
            server_name="127.0.0.1",
            server_port=7861,  # Different port from main GUI
            share=False,
            inbrowser=True
        )
    except KeyboardInterrupt:
        print("\nShutting down...")
        cleanup_docker()
        sys.exit(0)

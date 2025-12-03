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

def step_3_inverted_crush(crush_amount, state, progress=gr.Progress()):
    """Step 3: Inverted crush (crush bright areas to white before inversion)"""
    if not state or state.get("step", 0) < 2:
        return None, "Please run Step 2 first", state, ""

    try:
        temp_dir = Path("gui_temp")

        # Load masked depth
        depth_masked = np.array(state["masked"])

        # Apply inverted crush: bright areas (above 1-crush) become white
        if crush_amount > 0:
            # Formula: output = min(input / (1 - crush), 1.0)
            # This crushes bright areas to white while expanding dark/mid tones
            crushed = np.clip(depth_masked / (1.0 - crush_amount), 0, 1)
            console = f"Inverted crush applied (amount={crush_amount}): values above {1.0-crush_amount:.2f} → white"
        else:
            crushed = depth_masked
            console = "No inverted crush (amount=0)"

        # Re-mask with alpha
        alpha_path = temp_dir / "alpha_mask.npy"
        if alpha_path.exists():
            alpha_mask = np.load(alpha_path)
            crushed = crushed * alpha_mask + (1.0 - alpha_mask)

        # Save and display
        depth_img = (crushed * 255).astype(np.uint8)
        np.save(temp_dir / "step3_inv_crush.npy", crushed)

        state["step"] = 3
        state["inv_crushed"] = crushed.tolist()

        return depth_img, "✓ Inverted crush applied", state, console

    except Exception as e:
        return None, f"Error: {str(e)}", state, ""

def step_4_invert(state, progress=gr.Progress()):
    """Step 4: Invert depth (for relief carving)"""
    if not state or state.get("step", 0) < 3:
        return None, "Please run Step 3 first", state, ""

    try:
        temp_dir = Path("gui_temp")

        # Load inverted-crushed depth
        depth_crushed = np.array(state["inv_crushed"])

        # Invert
        depth_inverted = 1.0 - depth_crushed

        # Save and display
        depth_img = (depth_inverted * 255).astype(np.uint8)
        np.save(temp_dir / "step4_inverted.npy", depth_inverted)

        state["step"] = 4
        state["inverted"] = depth_inverted.tolist()

        return depth_img, "✓ Depth inverted", state, "Inverted: closer = darker (will be raised)"

    except Exception as e:
        return None, f"Error: {str(e)}", state, ""

def step_5_high_freq_detail(detail_strength, crush_amount, state, progress=gr.Progress()):
    """Step 5: Add high-frequency detail from original image"""
    if not state or state.get("step", 0) < 4:
        return None, "Please run Step 5 first", state, ""

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
        np.save(temp_dir / "step5_detail.npy", enhanced)

        state["step"] = 5
        state["detail"] = enhanced.tolist()

        return depth_img, f"✓ High-freq detail added", state, console

    except Exception as e:
        return None, f"Error: {str(e)}", state, ""

def step_6_clahe(clahe_clip, clahe_tile, crush_amount, state, progress=gr.Progress()):
    """Step 6: Apply CLAHE contrast enhancement"""
    if not state or state.get("step", 0) < 5:
        return None, "Please run Step 5 first", state, ""

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
        np.save(temp_dir / "step6_clahe.npy", enhanced)

        state["step"] = 6
        state["clahe"] = enhanced.tolist()

        return depth_img, f"✓ CLAHE applied", state, console

    except Exception as e:
        return None, f"Error: {str(e)}", state, ""

def step_7_bilateral(bilateral_d, bilateral_color, bilateral_space, crush_amount, state, progress=gr.Progress()):
    """Step 7: Apply bilateral filter"""
    if not state or state.get("step", 0) < 6:
        return None, "Please run Step 6 first", state, ""

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
        np.save(temp_dir / "step7_bilateral.npy", enhanced)

        state["step"] = 7
        state["bilateral"] = enhanced.tolist()

        return depth_img, f"✓ Bilateral filter applied", state, console

    except Exception as e:
        return None, f"Error: {str(e)}", state, ""

def step_8_final_crush(crush_amount, state, progress=gr.Progress()):
    """Step 8: Final shadow crush"""
    if not state or state.get("step", 0) < 7:
        return None, "Please run Step 7 first", state, ""

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
        np.save(temp_dir / "step8_crush.npy", enhanced)

        state["step"] = 8
        state["final"] = enhanced.tolist()

        return depth_img, f"✓ Final crush applied", state, console

    except Exception as e:
        return None, f"Error: {str(e)}", state, ""

def step_9_save_output(state, progress=gr.Progress()):
    """Step 9: Save final output"""
    if not state or state.get("step", 0) < 8:
        return "Please run Step 8 first", ""

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

def run_from_step(start_step, input_image, state,
                  enable_1, enable_2, enable_3, enable_4, enable_5, enable_6, enable_7, enable_8, enable_9,
                  step3_crush, detail_strength, step5_crush, clahe_clip, clahe_tile, step6_crush,
                  bilateral_d, bilateral_color, bilateral_space, step7_crush, step8_crush,
                  progress=gr.Progress()):
    """Run from a given step through all subsequent enabled steps"""

    console_log = ""
    enabled_steps = [enable_1, enable_2, enable_3, enable_4, enable_5, enable_6, enable_7, enable_8, enable_9]

    # Initialize output images
    img1 = img2 = img3 = img4 = img5 = img6 = img7 = img8 = None

    # Count enabled steps from start_step onwards
    total_steps = sum(enabled_steps[start_step-1:])
    current_step = 0

    # Step 1
    if start_step <= 1 and enable_1:
        progress(current_step/total_steps if total_steps > 0 else 0, desc="Step 1: Depth Estimation...")
        img1, status1, state, log1 = step_1_depth_estimation(input_image, progress)
        console_log += log1 + "\n\n"
        if img1 is None:
            return None, None, None, None, None, None, None, None, status1, state, console_log
        current_step += 1

    # Step 2
    if start_step <= 2 and enable_2 and state.get("step", 0) >= 1:
        progress(current_step/total_steps if total_steps > 0 else 0, desc="Step 2: Alpha Mask...")
        img2, status2, state, log2 = step_2_apply_alpha_mask(state, progress)
        console_log += log2 + "\n\n"
        current_step += 1

    # Step 3
    if start_step <= 3 and enable_3 and state.get("step", 0) >= 2:
        progress(current_step/total_steps if total_steps > 0 else 0, desc="Step 3: Inverted Crush...")
        img3, status3, state, log3 = step_3_inverted_crush(step3_crush, state, progress)
        console_log += log3 + "\n\n"
        current_step += 1

    # Step 4
    if start_step <= 4 and enable_4 and state.get("step", 0) >= 3:
        progress(current_step/total_steps if total_steps > 0 else 0, desc="Step 4: Invert...")
        img4, status4, state, log4 = step_4_invert(state, progress)
        console_log += log4 + "\n\n"
        current_step += 1

    # Step 5
    if start_step <= 5 and enable_5 and state.get("step", 0) >= 4:
        progress(current_step/total_steps if total_steps > 0 else 0, desc="Step 5: High-Freq Detail...")
        img5, status5, state, log5 = step_5_high_freq_detail(detail_strength, step5_crush, state, progress)
        console_log += log5 + "\n\n"
        current_step += 1

    # Step 6
    if start_step <= 6 and enable_6 and state.get("step", 0) >= 5:
        progress(current_step/total_steps if total_steps > 0 else 0, desc="Step 6: CLAHE...")
        img6, status6, state, log6 = step_6_clahe(clahe_clip, clahe_tile, step6_crush, state, progress)
        console_log += log6 + "\n\n"
        current_step += 1

    # Step 7
    if start_step <= 7 and enable_7 and state.get("step", 0) >= 6:
        progress(current_step/total_steps if total_steps > 0 else 0, desc="Step 7: Bilateral...")
        img7, status7, state, log7 = step_7_bilateral(bilateral_d, bilateral_color, bilateral_space, step7_crush, state, progress)
        console_log += log7 + "\n\n"
        current_step += 1

    # Step 8
    if start_step <= 8 and enable_8 and state.get("step", 0) >= 7:
        progress(current_step/total_steps if total_steps > 0 else 0, desc="Step 8: Final Crush...")
        img8, status8, state, log8 = step_8_final_crush(step8_crush, state, progress)
        console_log += log8 + "\n\n"
        current_step += 1

    # Step 9
    if start_step <= 9 and enable_9 and state.get("step", 0) >= 8:
        progress(current_step/total_steps if total_steps > 0 else 0, desc="Step 9: Saving...")
        save_status, save_log = step_9_save_output(state, progress)
        console_log += save_log + "\n\n"
        current_step += 1

    progress(1.0, desc="Complete!")

    status_msg = f"✓ Ran {current_step} steps from step {start_step}"
    if enable_9 and state.get("step", 0) >= 8 and start_step <= 9:
        status_msg += " - Output saved!"

    return (img1, img2, img3, img4, img5, img6, img7, img8,
            status_msg, state, console_log)

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

    gr.Markdown("### Pipeline Controls")
    gr.Markdown("*Enable/disable steps. Click any step's button to run from that step through all enabled steps below it.*")

    gr.Markdown("---")

    # Step 1: Depth Estimation
    with gr.Row():
        with gr.Column(scale=1):
            with gr.Row():
                enable_step1 = gr.Checkbox(value=True, label="Enable", scale=1)
                gr.Markdown("### Step 1: Depth Estimation", elem_classes=["step-header"])
            step1_btn = gr.Button("▶ Run from Step 1", variant="primary")
            step1_status = gr.Textbox(label="Status", lines=1, interactive=False)
        with gr.Column(scale=2):
            step1_output = gr.Image(type="numpy", label="1. Raw ZoeDepth Output", format="png")

    # Step 2: Alpha Mask
    with gr.Row():
        with gr.Column(scale=1):
            with gr.Row():
                enable_step2 = gr.Checkbox(value=True, label="Enable", scale=1)
                gr.Markdown("### Step 2: Apply Alpha Mask", elem_classes=["step-header"])
            gr.Markdown("*Transparent areas → white (will be black after inversion)*")
            step2_btn = gr.Button("▶ Run from Step 2", variant="secondary")
            step2_status = gr.Textbox(label="Status", lines=1, interactive=False)
        with gr.Column(scale=2):
            step2_output = gr.Image(type="numpy", label="2. Alpha Masked (transparent=white)", format="png")

    # Step 3: Inverted Crush
    with gr.Row():
        with gr.Column(scale=1):
            with gr.Row():
                enable_step3 = gr.Checkbox(value=True, label="Enable", scale=1)
                gr.Markdown("### Step 3: Inverted Crush", elem_classes=["step-header"])
            gr.Markdown("*Crush bright areas to white (before inversion)*")
            step3_crush = gr.Slider(0.0, 0.5, 0.0, step=0.05, label="Inverted Crush Amount")
            step3_btn = gr.Button("▶ Run from Step 3", variant="secondary")
            step3_status = gr.Textbox(label="Status", lines=1, interactive=False)
        with gr.Column(scale=2):
            step3_output = gr.Image(type="numpy", label="3. Inverted Crush (bright→white)", format="png")

    # Step 4: Invert
    with gr.Row():
        with gr.Column(scale=1):
            with gr.Row():
                enable_step4 = gr.Checkbox(value=True, label="Enable", scale=1)
                gr.Markdown("### Step 4: Invert Depth", elem_classes=["step-header"])
            gr.Markdown("*For relief carving: closer = darker = raised*")
            step4_btn = gr.Button("▶ Run from Step 4", variant="secondary")
            step4_status = gr.Textbox(label="Status", lines=1, interactive=False)
        with gr.Column(scale=2):
            step4_output = gr.Image(type="numpy", label="4. Inverted (transparent=black)", format="png")

    # Step 5: High-Freq Detail
    with gr.Row():
        with gr.Column(scale=1):
            with gr.Row():
                enable_step5 = gr.Checkbox(value=True, label="Enable", scale=1)
                gr.Markdown("### Step 5: Add Surface Detail", elem_classes=["step-header"])
            detail_strength = gr.Slider(0.0, 1.0, 0.2, step=0.05, label="Detail Strength")
            step5_crush = gr.Slider(0.0, 0.5, 0.0, step=0.05, label="Crush Amount (optional)")
            step5_btn = gr.Button("▶ Run from Step 5", variant="secondary")
            step5_status = gr.Textbox(label="Status", lines=1, interactive=False)
        with gr.Column(scale=2):
            step5_output = gr.Image(type="numpy", label="5. With High-Freq Detail", format="png")

    # Step 6: CLAHE
    with gr.Row():
        with gr.Column(scale=1):
            with gr.Row():
                enable_step6 = gr.Checkbox(value=True, label="Enable", scale=1)
                gr.Markdown("### Step 6: CLAHE Contrast", elem_classes=["step-header"])
            clahe_clip = gr.Slider(1.0, 5.0, 2.5, step=0.1, label="Clip Limit")
            clahe_tile = gr.Slider(8, 64, 16, step=8, label="Tile Size")
            step6_crush = gr.Slider(0.0, 0.5, 0.0, step=0.05, label="Crush Amount (optional)")
            step6_btn = gr.Button("▶ Run from Step 6", variant="secondary")
            step6_status = gr.Textbox(label="Status", lines=1, interactive=False)
        with gr.Column(scale=2):
            step6_output = gr.Image(type="numpy", label="6. CLAHE Enhanced", format="png")

    # Step 7: Bilateral Filter
    with gr.Row():
        with gr.Column(scale=1):
            with gr.Row():
                enable_step7 = gr.Checkbox(value=True, label="Enable", scale=1)
                gr.Markdown("### Step 7: Bilateral Smoothing", elem_classes=["step-header"])
            bilateral_d = gr.Slider(5, 15, 9, step=2, label="Filter Diameter")
            bilateral_color = gr.Slider(10, 150, 75, step=5, label="Color Sigma")
            bilateral_space = gr.Slider(10, 150, 75, step=5, label="Space Sigma")
            step7_crush = gr.Slider(0.0, 0.5, 0.0, step=0.05, label="Crush Amount (optional)")
            step7_btn = gr.Button("▶ Run from Step 7", variant="secondary")
            step7_status = gr.Textbox(label="Status", lines=1, interactive=False)
        with gr.Column(scale=2):
            step7_output = gr.Image(type="numpy", label="7. Bilateral Filtered", format="png")

    # Step 8: Final Crush
    with gr.Row():
        with gr.Column(scale=1):
            with gr.Row():
                enable_step8 = gr.Checkbox(value=True, label="Enable", scale=1)
                gr.Markdown("### Step 8: Final Shadow Crush", elem_classes=["step-header"])
            step8_crush = gr.Slider(0.0, 0.5, 0.0, step=0.05, label="Final Crush Amount")
            step8_btn = gr.Button("▶ Run from Step 8", variant="secondary")
            step8_status = gr.Textbox(label="Status", lines=1, interactive=False)
        with gr.Column(scale=2):
            step8_output = gr.Image(type="numpy", label="8. Final Crushed", format="png")

    # Step 9: Save
    gr.Markdown("---")
    with gr.Row():
        with gr.Column(scale=1):
            with gr.Row():
                enable_step9 = gr.Checkbox(value=True, label="Enable", scale=1)
                gr.Markdown("### Step 9: Save Output", elem_classes=["step-header"])
            step9_btn = gr.Button("💾 Run from Step 9 (Save)", variant="primary", size="lg")
        with gr.Column(scale=2):
            step9_status = gr.Textbox(label="Save Status", lines=2, interactive=False)

    # Global controls
    gr.Markdown("---")
    with gr.Row():
        run_all_btn = gr.Button("⚡ Run All Steps (with current settings)", variant="primary", size="lg")

    # Common inputs for all buttons
    common_inputs = [
        input_image, pipeline_state,
        enable_step1, enable_step2, enable_step3, enable_step4, enable_step5, enable_step6, enable_step7, enable_step8, enable_step9,
        step3_crush, detail_strength, step5_crush,
        clahe_clip, clahe_tile, step6_crush,
        bilateral_d, bilateral_color, bilateral_space, step7_crush, step8_crush
    ]

    common_outputs = [
        step1_output, step2_output, step3_output, step4_output,
        step5_output, step6_output, step7_output, step8_output,
        step9_status, pipeline_state, console_output
    ]

    # Event handlers - each button runs from its step through all enabled subsequent steps
    step1_btn.click(
        fn=lambda img, state, e1,e2,e3,e4,e5,e6,e7,e8,e9, c3,ds,c5,cc,ct,c6,bd,bco,bsp,c7,c8:
            run_from_step(1, img, state, e1,e2,e3,e4,e5,e6,e7,e8,e9, c3,ds,c5,cc,ct,c6,bd,bco,bsp,c7,c8),
        inputs=common_inputs,
        outputs=common_outputs
    )

    step2_btn.click(
        fn=lambda img, state, e1,e2,e3,e4,e5,e6,e7,e8,e9, c3,ds,c5,cc,ct,c6,bd,bco,bsp,c7,c8:
            run_from_step(2, img, state, e1,e2,e3,e4,e5,e6,e7,e8,e9, c3,ds,c5,cc,ct,c6,bd,bco,bsp,c7,c8),
        inputs=common_inputs,
        outputs=common_outputs
    )

    step3_btn.click(
        fn=lambda img, state, e1,e2,e3,e4,e5,e6,e7,e8,e9, c3,ds,c5,cc,ct,c6,bd,bco,bsp,c7,c8:
            run_from_step(3, img, state, e1,e2,e3,e4,e5,e6,e7,e8,e9, c3,ds,c5,cc,ct,c6,bd,bco,bsp,c7,c8),
        inputs=common_inputs,
        outputs=common_outputs
    )

    step4_btn.click(
        fn=lambda img, state, e1,e2,e3,e4,e5,e6,e7,e8,e9, c3,ds,c5,cc,ct,c6,bd,bco,bsp,c7,c8:
            run_from_step(4, img, state, e1,e2,e3,e4,e5,e6,e7,e8,e9, c3,ds,c5,cc,ct,c6,bd,bco,bsp,c7,c8),
        inputs=common_inputs,
        outputs=common_outputs
    )

    step5_btn.click(
        fn=lambda img, state, e1,e2,e3,e4,e5,e6,e7,e8,e9, c3,ds,c5,cc,ct,c6,bd,bco,bsp,c7,c8:
            run_from_step(5, img, state, e1,e2,e3,e4,e5,e6,e7,e8,e9, c3,ds,c5,cc,ct,c6,bd,bco,bsp,c7,c8),
        inputs=common_inputs,
        outputs=common_outputs
    )

    step6_btn.click(
        fn=lambda img, state, e1,e2,e3,e4,e5,e6,e7,e8,e9, c3,ds,c5,cc,ct,c6,bd,bco,bsp,c7,c8:
            run_from_step(6, img, state, e1,e2,e3,e4,e5,e6,e7,e8,e9, c3,ds,c5,cc,ct,c6,bd,bco,bsp,c7,c8),
        inputs=common_inputs,
        outputs=common_outputs
    )

    step7_btn.click(
        fn=lambda img, state, e1,e2,e3,e4,e5,e6,e7,e8,e9, c3,ds,c5,cc,ct,c6,bd,bco,bsp,c7,c8:
            run_from_step(7, img, state, e1,e2,e3,e4,e5,e6,e7,e8,e9, c3,ds,c5,cc,ct,c6,bd,bco,bsp,c7,c8),
        inputs=common_inputs,
        outputs=common_outputs
    )

    step8_btn.click(
        fn=lambda img, state, e1,e2,e3,e4,e5,e6,e7,e8,e9, c3,ds,c5,cc,ct,c6,bd,bco,bsp,c7,c8:
            run_from_step(8, img, state, e1,e2,e3,e4,e5,e6,e7,e8,e9, c3,ds,c5,cc,ct,c6,bd,bco,bsp,c7,c8),
        inputs=common_inputs,
        outputs=common_outputs
    )

    step9_btn.click(
        fn=lambda img, state, e1,e2,e3,e4,e5,e6,e7,e8,e9, c3,ds,c5,cc,ct,c6,bd,bco,bsp,c7,c8:
            run_from_step(9, img, state, e1,e2,e3,e4,e5,e6,e7,e8,e9, c3,ds,c5,cc,ct,c6,bd,bco,bsp,c7,c8),
        inputs=common_inputs,
        outputs=common_outputs
    )

    run_all_btn.click(
        fn=lambda img, state, e1,e2,e3,e4,e5,e6,e7,e8,e9, c3,ds,c5,cc,ct,c6,bd,bco,bsp,c7,c8:
            run_from_step(1, img, state, e1,e2,e3,e4,e5,e6,e7,e8,e9, c3,ds,c5,cc,ct,c6,bd,bco,bsp,c7,c8),
        inputs=common_inputs,
        outputs=common_outputs
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

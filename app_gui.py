import gradio as gr
import torch
import os
import uuid
import numpy as np
from diffusers import MochiPipeline, FlowMatchEulerDiscreteScheduler
from diffusers.utils import export_to_video
from config import MODEL_ID, TORCH_DTYPE, NEGATIVE_PROMPT
from eval import EvalSuite
import time

# Load models globally for the GUI
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Loading Mochi Pipeline on {device}...")
pipe = MochiPipeline.from_pretrained(MODEL_ID, torch_dtype=TORCH_DTYPE)
pipe.enable_model_cpu_offload()
pipe.enable_vae_tiling()

evaluator = EvalSuite(device=device)

def generate_comparison(prompt, shift, guidance, use_neg_prompt):
    task_id = str(uuid.uuid4())
    os.makedirs("outputs", exist_ok=True)
    
    neg_p = NEGATIVE_PROMPT if use_neg_prompt else None
    
    # --- 1. Generate Baseline ---
    print("Generating Baseline...")
    pipe.scheduler = FlowMatchEulerDiscreteScheduler.from_config(pipe.scheduler.config, shift=1.0)
    with torch.autocast("cuda", torch.bfloat16 if device == "cuda" else torch.float32):
        baseline_frames = pipe(
            prompt=prompt,
            num_frames=60,
            num_inference_steps=50,
            guidance_scale=4.5,
            generator=torch.Generator(device="cpu").manual_seed(42),
        ).frames[0]
    
    baseline_path = f"outputs/{task_id}_baseline.mp4"
    export_to_video(baseline_frames, baseline_path, fps=30)
    
    # --- 2. Generate Enhanced ---
    print("Generating Enhanced...")
    pipe.scheduler = FlowMatchEulerDiscreteScheduler.from_config(pipe.scheduler.config, shift=shift)
    with torch.autocast("cuda", torch.bfloat16 if device == "cuda" else torch.float32):
        enhanced_frames = pipe(
            prompt=prompt,
            negative_prompt=neg_p,
            num_frames=60,
            num_inference_steps=50,
            guidance_scale=guidance,
            generator=torch.Generator(device="cpu").manual_seed(42),
        ).frames[0]
    
    enhanced_path = f"outputs/{task_id}_enhanced.mp4"
    export_to_video(enhanced_frames, enhanced_path, fps=30)
    
    # --- 3. Evaluate ---
    b_ssim, b_lpips = evaluator.compute_temporal_consistency(baseline_frames)
    e_ssim, e_lpips = evaluator.compute_temporal_consistency(enhanced_frames)
    
    metrics_md = f"""
    ### Performance Comparison
    | Metric | Baseline | Enhanced | Improvement |
    | :--- | :--- | :--- | :--- |
    | **SSIM** (higher is better) | {b_ssim:.4f} | {e_ssim:.4f} | **{((e_ssim-b_ssim)/b_ssim)*100:+.1f}%** |
    | **LPIPS** (lower is better) | {b_lpips:.4f} | {e_lpips:.4f} | **{((b_lpips-e_lpips)/b_lpips)*100:+.1f}%** |
    """
    
    return baseline_path, enhanced_path, metrics_md

# --- GUI Layout ---
with gr.Blocks(theme=gr.themes.Soft()) as demo:
    gr.Markdown("# 🎬 Mochi-Enhance: Training-Free Video Optimization")
    gr.Markdown("Compare the baseline Mochi-1-preview model with our native enhancement methodology.")
    
    with gr.Row():
        with gr.Column():
            prompt = gr.Textbox(label="Video Prompt", placeholder="A camel moving in the desert...", lines=2)
            with gr.Row():
                shift = gr.Slider(label="Scheduler Shift", minimum=1.0, maximum=2.0, value=1.15, step=0.05)
                guidance = gr.Slider(label="Guidance Scale", minimum=1.0, maximum=10.0, value=6.0, step=0.5)
            use_neg = gr.Checkbox(label="Inject Negative Prompt", value=True)
            btn = gr.Button("Generate Comparison", variant="primary")
            
    with gr.Row():
        with gr.Column():
            gr.Markdown("### Baseline (Standard)")
            video_baseline = gr.Video()
        with gr.Column():
            gr.Markdown("### Enhanced (Our Method)")
            video_enhanced = gr.Video()
            
    metrics = gr.Markdown()
    
    btn.click(
        fn=generate_comparison,
        inputs=[prompt, shift, guidance, use_neg],
        outputs=[video_baseline, video_enhanced, metrics]
    )

if __name__ == "__main__":
    demo.launch()

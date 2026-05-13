"""
Research-Grade Ablation Study for Mochi-1-Preview Native Enhancements.
Runs experiments in parallel across 2 GPUs using multiprocessing.
"""
import os
import subprocess
import json
import time

def find_best_gpus(n=2):
    """Query nvidia-smi and return the N GPU IDs with the most free VRAM."""
    try:
        smi_output = subprocess.check_output(
            ['nvidia-smi', '--query-gpu=index,memory.free', '--format=csv,nounits,noheader']
        ).decode('utf-8')
        gpus = []
        for line in smi_output.strip().split('\n'):
            if not line:
                continue
            idx, free_mem = line.split(',')
            gpus.append((int(idx.strip()), int(free_mem.strip())))
        gpus.sort(key=lambda x: x[1], reverse=True)
        selected = [g[0] for g in gpus[:n]]
        for g in gpus[:n]:
            print(f"  Selected GPU {g[0]} with {g[1]} MB free VRAM")
        return selected
    except Exception as e:
        print(f"Failed to query nvidia-smi: {e}. Defaulting to [0, 1].")
        return list(range(n))

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

# ---- Worker Function (runs in a subprocess) ----

def gpu_worker(gpu_id, tasks, results_file):
    """
    Worker function that runs on a single GPU.
    - gpu_id: physical GPU index
    - tasks: list of (prompt_idx, prompt, config_name, shift, neg_prompt, guidance, seed)
    - results_file: path to write JSON results
    """
    # Pin this process to a single GPU
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

    import torch
    from diffusers import MochiPipeline, FlowMatchEulerDiscreteScheduler
    from diffusers.utils import export_to_video
    from config import MODEL_ID, TORCH_DTYPE, NUM_FRAMES, FPS, NUM_INFERENCE_STEPS, RESULTS_DIR
    from eval import EvalSuite

    device = "cuda"
    print(f"[GPU {gpu_id}] Loading Mochi Pipeline...")
    pipe = MochiPipeline.from_pretrained(MODEL_ID, torch_dtype=TORCH_DTYPE)
    pipe.enable_model_cpu_offload()
    pipe.enable_vae_tiling()

    # Save the default scheduler config for resetting between runs
    default_scheduler_config = dict(pipe.scheduler.config)

    evaluator = EvalSuite(device=device)
    results = []

    for task_idx, task in enumerate(tasks):
        prompt_idx, prompt, config_name, shift, neg_prompt, guidance, seed = task
        tag = f"[GPU {gpu_id} | Task {task_idx+1}/{len(tasks)}]"
        print(f"{tag} Prompt {prompt_idx+1}, Config='{config_name}', Seed={seed}")

        # Reset scheduler to default, then apply shift
        pipe.scheduler = FlowMatchEulerDiscreteScheduler.from_config(default_scheduler_config, shift=shift)

        # Set seed for reproducibility
        generator = torch.Generator(device="cpu").manual_seed(seed)

        start_time = time.time()
        with torch.autocast("cuda", torch.bfloat16):
            frames = pipe(
                prompt=prompt,
                negative_prompt=neg_prompt,
                num_frames=NUM_FRAMES,
                num_inference_steps=NUM_INFERENCE_STEPS,
                guidance_scale=guidance,
                generator=generator,
            ).frames[0]
        gen_time = time.time() - start_time
        print(f"{tag} Generation took {gen_time:.1f}s")

        # Save video
        video_name = f"p{prompt_idx}_{config_name.replace(' ', '_').replace(':', '')}_{seed}.mp4"
        video_path = os.path.join(RESULTS_DIR, "videos", video_name)
        export_to_video(frames, video_path, fps=FPS)

        # Evaluate
        clip_sim = evaluator.compute_clip_sim(frames, prompt)
        avg_ssim, avg_lpips = evaluator.compute_temporal_consistency(frames)
        print(f"{tag} CLIP-SIM={clip_sim:.2f}, SSIM={avg_ssim:.4f}, LPIPS={avg_lpips:.4f}")

        results.append({
            "prompt_idx": prompt_idx,
            "prompt": prompt,
            "config": config_name,
            "seed": seed,
            "clip_sim": clip_sim,
            "ssim": avg_ssim,
            "lpips": avg_lpips,
            "gen_time_s": gen_time,
            "video_path": video_path,
        })

        # Free VRAM between runs
        torch.cuda.empty_cache()

    # Write results to JSON
    with open(results_file, "w") as f:
        json.dump(results, f, indent=2)
    print(f"[GPU {gpu_id}] Done. {len(results)} results saved to {results_file}")


# ---- Main Orchestrator ----

def main():
    import torch.multiprocessing as mp
    import pandas as pd
    from config import EVALUATION_PROMPTS, SEEDS, ABLATION_CONFIGS, RESULTS_DIR

    print("=" * 70)
    print("  RESEARCH ABLATION STUDY — Mochi-1-Preview Native Enhancements")
    print("=" * 70)

    # Create output directories
    os.makedirs(os.path.join(RESULTS_DIR, "videos"), exist_ok=True)

    # Find 2 best GPUs
    print("\nSelecting 2 GPUs with most free VRAM...")
    gpu_ids = find_best_gpus(n=2)

    # Build task list
    tasks = []
    for p_idx, prompt in enumerate(EVALUATION_PROMPTS):
        for config_name, shift, neg_prompt, guidance in ABLATION_CONFIGS:
            for seed in SEEDS:
                tasks.append((p_idx, prompt, config_name, shift, neg_prompt, guidance, seed))

    total = len(tasks)
    print(f"\nTotal experiments: {total} ({len(EVALUATION_PROMPTS)} prompts × {len(ABLATION_CONFIGS)} configs × {len(SEEDS)} seeds)")

    # Split tasks between 2 GPUs
    mid = total // 2
    tasks_gpu0 = tasks[:mid]
    tasks_gpu1 = tasks[mid:]
    print(f"GPU {gpu_ids[0]}: {len(tasks_gpu0)} tasks | GPU {gpu_ids[1]}: {len(tasks_gpu1)} tasks")

    results_file_0 = os.path.join(RESULTS_DIR, "results_gpu0.json")
    results_file_1 = os.path.join(RESULTS_DIR, "results_gpu1.json")

    # Launch workers
    mp.set_start_method("spawn", force=True)
    print("\nLaunching parallel workers...\n")

    p0 = mp.Process(target=gpu_worker, args=(gpu_ids[0], tasks_gpu0, results_file_0))
    p1 = mp.Process(target=gpu_worker, args=(gpu_ids[1], tasks_gpu1, results_file_1))

    overall_start = time.time()
    p0.start()
    p1.start()
    p0.join()
    p1.join()
    total_time = time.time() - overall_start

    print(f"\n{'=' * 70}")
    print(f"  All workers finished in {total_time/60:.1f} minutes")
    print(f"{'=' * 70}")

    # Aggregate results
    all_results = []
    for rfile in [results_file_0, results_file_1]:
        if os.path.exists(rfile):
            with open(rfile) as f:
                all_results.extend(json.load(f))

    if not all_results:
        print("ERROR: No results collected!")
        return

    df = pd.DataFrame(all_results)

    # Save raw results
    raw_csv = os.path.join(RESULTS_DIR, "raw_results.csv")
    df.to_csv(raw_csv, index=False)
    print(f"\nRaw results saved to: {raw_csv}")

    # Compute mean ± std per config (aggregated across prompts and seeds)
    print("\n" + "=" * 70)
    print("  ABLATION STUDY RESULTS (mean ± std)")
    print("=" * 70)

    summary_rows = []
    for config_name in df["config"].unique():
        subset = df[df["config"] == config_name]
        row = {
            "Config": config_name,
            "CLIP-SIM": f"{subset['clip_sim'].mean():.2f} ± {subset['clip_sim'].std():.2f}",
            "SSIM": f"{subset['ssim'].mean():.4f} ± {subset['ssim'].std():.4f}",
            "LPIPS": f"{subset['lpips'].mean():.4f} ± {subset['lpips'].std():.4f}",
            "Time (s)": f"{subset['gen_time_s'].mean():.1f} ± {subset['gen_time_s'].std():.1f}",
        }
        summary_rows.append(row)

    summary_df = pd.DataFrame(summary_rows)
    print(summary_df.to_string(index=False))

    summary_csv = os.path.join(RESULTS_DIR, "ablation_summary.csv")
    summary_df.to_csv(summary_csv, index=False)
    print(f"\nSummary saved to: {summary_csv}")

    # Per-prompt breakdown
    print("\n" + "=" * 70)
    print("  PER-PROMPT BREAKDOWN (mean across seeds)")
    print("=" * 70)
    for p_idx, prompt in enumerate(EVALUATION_PROMPTS):
        prompt_df = df[df["prompt_idx"] == p_idx]
        print(f"\n  Prompt {p_idx+1}: \"{prompt[:60]}...\"")
        for config_name in prompt_df["config"].unique():
            subset = prompt_df[prompt_df["config"] == config_name]
            print(f"    {config_name:25s} | CLIP={subset['clip_sim'].mean():.2f} SSIM={subset['ssim'].mean():.4f} LPIPS={subset['lpips'].mean():.4f}")


if __name__ == "__main__":
    main()

"""
Compute FVD (Fréchet Video Distance) between Baseline and each enhanced config
from the already-generated videos. No re-generation needed.
"""
import os
import subprocess
import json

def find_best_gpu():
    try:
        smi = subprocess.check_output(
            ['nvidia-smi', '--query-gpu=index,memory.free', '--format=csv,nounits,noheader']
        ).decode()
        gpus = []
        for line in smi.strip().split('\n'):
            idx, free = line.split(',')
            gpus.append((int(idx.strip()), int(free.strip())))
        gpus.sort(key=lambda x: x[1], reverse=True)
        best = gpus[0]
        print(f"Using GPU {best[0]} with {best[1]} MB free VRAM")
        os.environ["CUDA_VISIBLE_DEVICES"] = str(best[0])
    except Exception as e:
        print(f"nvidia-smi failed: {e}")

find_best_gpu()

import numpy as np
import pandas as pd
import imageio
from PIL import Image
from eval import EvalSuite

RESULTS_DIR = "results"

def load_video_frames(path):
    """Load video frames as PIL Images."""
    reader = imageio.get_reader(path)
    frames = [Image.fromarray(f) for f in reader]
    reader.close()
    return frames

def main():
    evaluator = EvalSuite(device="cuda")

    # Load raw results to find video paths
    raw_df = pd.read_csv(os.path.join(RESULTS_DIR, "raw_results.csv"))

    configs = raw_df["config"].unique()
    enhanced_configs = [c for c in configs if c != "Baseline"]

    # For each prompt × seed, compute FVD between Baseline and each enhanced config
    fvd_results = []

    prompts = raw_df["prompt_idx"].unique()
    seeds = raw_df["seed"].unique()

    for p_idx in sorted(prompts):
        for seed in sorted(seeds):
            # Get baseline video for this prompt + seed
            baseline_row = raw_df[
                (raw_df["prompt_idx"] == p_idx) &
                (raw_df["config"] == "Baseline") &
                (raw_df["seed"] == seed)
            ]
            if baseline_row.empty:
                continue

            baseline_path = baseline_row.iloc[0]["video_path"]
            if not os.path.exists(baseline_path):
                print(f"  Baseline video not found: {baseline_path}, skipping")
                continue

            baseline_frames = load_video_frames(baseline_path)
            prompt_text = baseline_row.iloc[0]["prompt"]

            for config_name in enhanced_configs:
                enhanced_row = raw_df[
                    (raw_df["prompt_idx"] == p_idx) &
                    (raw_df["config"] == config_name) &
                    (raw_df["seed"] == seed)
                ]
                if enhanced_row.empty:
                    continue

                enhanced_path = enhanced_row.iloc[0]["video_path"]
                if not os.path.exists(enhanced_path):
                    print(f"  Enhanced video not found: {enhanced_path}, skipping")
                    continue

                enhanced_frames = load_video_frames(enhanced_path)

                tag = f"Prompt {p_idx+1}, Seed {seed}, {config_name}"
                print(f"  Computing FVD: {tag}...")
                fvd = evaluator.compute_fvd(baseline_frames, enhanced_frames)
                print(f"    FVD = {fvd:.2f}")

                fvd_results.append({
                    "prompt_idx": int(p_idx),
                    "prompt": prompt_text,
                    "config": config_name,
                    "seed": int(seed),
                    "fvd_vs_baseline": fvd,
                })

    # Save raw FVD results
    fvd_df = pd.DataFrame(fvd_results)
    fvd_csv = os.path.join(RESULTS_DIR, "fvd_results.csv")
    fvd_df.to_csv(fvd_csv, index=False)
    print(f"\nRaw FVD results saved to: {fvd_csv}")

    # Summary: mean ± std per config
    print("\n" + "=" * 60)
    print("  FVD vs Baseline (mean ± std)")
    print("=" * 60)

    summary_rows = []
    for config_name in enhanced_configs:
        subset = fvd_df[fvd_df["config"] == config_name]
        mean_fvd = subset["fvd_vs_baseline"].mean()
        std_fvd = subset["fvd_vs_baseline"].std()
        print(f"  {config_name:25s} | FVD = {mean_fvd:.2f} ± {std_fvd:.2f}")
        summary_rows.append({
            "Config": config_name,
            "FVD vs Baseline": f"{mean_fvd:.2f} ± {std_fvd:.2f}",
        })

    summary_df = pd.DataFrame(summary_rows)
    fvd_summary_csv = os.path.join(RESULTS_DIR, "fvd_summary.csv")
    summary_df.to_csv(fvd_summary_csv, index=False)
    print(f"\nFVD summary saved to: {fvd_summary_csv}")

    # Also merge FVD into the main ablation summary
    main_summary = pd.read_csv(os.path.join(RESULTS_DIR, "ablation_summary.csv"))
    fvd_map = {"Baseline": "N/A (reference)"}
    for _, row in summary_df.iterrows():
        fvd_map[row["Config"]] = row["FVD vs Baseline"]
    main_summary["FVD vs Baseline"] = main_summary["Config"].map(fvd_map)
    updated_csv = os.path.join(RESULTS_DIR, "ablation_summary_with_fvd.csv")
    main_summary.to_csv(updated_csv, index=False)
    print(f"Updated ablation summary saved to: {updated_csv}")


if __name__ == "__main__":
    main()

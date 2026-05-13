import os
import pandas as pd
import imageio
import matplotlib.pyplot as plt
from PIL import Image

RESULTS_DIR = "results"
FIGURES_DIR = os.path.join(RESULTS_DIR, "figures")
os.makedirs(FIGURES_DIR, exist_ok=True)

def load_video_frames(path):
    """Load video frames as a list of numpy arrays."""
    reader = imageio.get_reader(path)
    frames = [frame for frame in reader]
    reader.close()
    return frames

def create_qualitative_grid(raw_df, seed=42):
    print(f"Generating Figure 4 (Qualitative Comparison Grid, Seed={seed})...")
    
    prompts = raw_df["prompt_idx"].unique()
    num_prompts = len(prompts)
    
    # We want 3 frames: t=0, t=mid, t=end
    # For each prompt, we want Baseline row and Enhanced row
    
    fig, axes = plt.subplots(num_prompts * 2, 3, figsize=(15, 4 * num_prompts))
    plt.subplots_adjust(wspace=0.05, hspace=0.1)
    
    for i, p_idx in enumerate(sorted(prompts)):
        # Get paths
        base_row = raw_df[(raw_df["prompt_idx"] == p_idx) & (raw_df["config"] == "Baseline") & (raw_df["seed"] == seed)]
        enh_row = raw_df[(raw_df["prompt_idx"] == p_idx) & (raw_df["config"] == "D: All Combined") & (raw_df["seed"] == seed)]
        
        if base_row.empty or enh_row.empty:
            print(f"Missing data for prompt {p_idx}, seed {seed}. Skipping.")
            continue
            
        base_path = base_row.iloc[0]["video_path"]
        enh_path = enh_row.iloc[0]["video_path"]
        prompt_text = base_row.iloc[0]["prompt"]
        
        # Load frames
        base_frames = load_video_frames(base_path)
        enh_frames = load_video_frames(enh_path)
        
        num_frames = min(len(base_frames), len(enh_frames))
        indices = [0, num_frames // 2, num_frames - 1]
        
        # Plot Baseline
        base_row_idx = i * 2
        for j, frame_idx in enumerate(indices):
            ax = axes[base_row_idx, j]
            ax.imshow(base_frames[frame_idx])
            ax.axis('off')
            if j == 0:
                ax.set_title(f"Prompt {p_idx+1}: {prompt_text[:40]}...\nBaseline", loc='left', fontsize=12, fontweight='bold')
                
        # Plot Enhanced
        enh_row_idx = i * 2 + 1
        for j, frame_idx in enumerate(indices):
            ax = axes[enh_row_idx, j]
            ax.imshow(enh_frames[frame_idx])
            ax.axis('off')
            if j == 0:
                ax.set_title(f"Enhanced (D: All Combined)", loc='left', fontsize=12, fontweight='bold')

    fig_path = os.path.join(FIGURES_DIR, "fig4_qualitative_grid.png")
    plt.savefig(fig_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved Qualitative Grid to {fig_path}")

def main():
    print("=" * 60)
    print("  Extracting Qualitative Frames")
    print("=" * 60)
    
    raw_csv = os.path.join(RESULTS_DIR, "raw_results.csv")
    if not os.path.exists(raw_csv):
        print("Error: raw_results.csv not found.")
        return
        
    raw_df = pd.read_csv(raw_csv)
    
    # Use seed 42 for the comparison
    create_qualitative_grid(raw_df, seed=42)
    
if __name__ == "__main__":
    main()

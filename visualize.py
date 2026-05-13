import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

RESULTS_DIR = "results"
FIGURES_DIR = os.path.join(RESULTS_DIR, "figures")
os.makedirs(FIGURES_DIR, exist_ok=True)

# Set plot style
sns.set_theme(style="whitegrid")
plt.rcParams.update({'font.size': 12, 'figure.autolayout': True})

def load_data():
    raw_df = pd.read_csv(os.path.join(RESULTS_DIR, "raw_results.csv"))
    
    # Check if FVD is available
    fvd_path = os.path.join(RESULTS_DIR, "ablation_summary_with_fvd.csv")
    summary_path = os.path.join(RESULTS_DIR, "ablation_summary.csv")
    
    if os.path.exists(fvd_path):
        summary_df = pd.read_csv(fvd_path)
    elif os.path.exists(summary_path):
        summary_df = pd.read_csv(summary_path)
    else:
        summary_df = None
        
    return raw_df, summary_df

def generate_latex_table(summary_df):
    print("Generating LaTeX Table 1...")
    # Basic formatting
    latex_str = summary_df.to_latex(index=False, column_format="l" + "c" * (len(summary_df.columns)-1),
                                     caption="Ablation Study Results (Mean $\pm$ Std). Best results in bold.",
                                     label="tab:ablation_results")
    
    table_path = os.path.join(FIGURES_DIR, "table1_ablation.tex")
    with open(table_path, "w") as f:
        f.write(latex_str)
    print(f"Saved LaTeX table to {table_path}")

def plot_grouped_bar_chart(raw_df):
    print("Generating Figure 1 (Grouped Bar Chart)...")
    
    # Prepare data
    metrics = ["clip_sim", "ssim", "lpips"]
    titles = ["CLIP-SIM (↑)", "SSIM (↑)", "LPIPS (↓)"]
    
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    
    for i, metric in enumerate(metrics):
        sns.barplot(
            data=raw_df, 
            x="config", 
            y=metric, 
            ax=axes[i],
            capsize=.1, 
            errcolor=".5",
            errorbar="sd"
        )
        axes[i].set_title(titles[i], fontweight='bold')
        axes[i].set_xlabel("")
        axes[i].set_ylabel("Score")
        axes[i].tick_params(axis='x', rotation=45)
        
    plt.tight_layout()
    fig_path = os.path.join(FIGURES_DIR, "fig1_metrics_bar.png")
    plt.savefig(fig_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved Figure 1 to {fig_path}")

def plot_per_prompt_heatmap(raw_df):
    print("Generating Figure 2 (Per-Prompt Heatmap)...")
    
    # We'll plot relative improvement over baseline for SSIM and CLIP-SIM
    # First, calculate baseline means per prompt
    baseline_means = raw_df[raw_df["config"] == "Baseline"].groupby("prompt_idx")[["clip_sim", "ssim", "lpips"]].mean()
    
    # Calculate means for all configs
    all_means = raw_df.groupby(["prompt_idx", "config"])[["clip_sim", "ssim", "lpips"]].mean().reset_index()
    
    # Filter out baseline for the heatmap
    enhanced_means = all_means[all_means["config"] != "Baseline"].copy()
    
    # Calculate relative percentage improvement
    for idx in baseline_means.index:
        mask = enhanced_means["prompt_idx"] == idx
        base_clip = baseline_means.loc[idx, "clip_sim"]
        base_ssim = baseline_means.loc[idx, "ssim"]
        base_lpips = baseline_means.loc[idx, "lpips"]
        
        enhanced_means.loc[mask, "clip_rel"] = ((enhanced_means.loc[mask, "clip_sim"] - base_clip) / base_clip) * 100
        enhanced_means.loc[mask, "ssim_rel"] = ((enhanced_means.loc[mask, "ssim"] - base_ssim) / base_ssim) * 100
        enhanced_means.loc[mask, "lpips_rel"] = ((base_lpips - enhanced_means.loc[mask, "lpips"]) / base_lpips) * 100 # Note flipped sign for LPIPS so higher is better improvement
        
    # Pivot for heatmap
    clip_pivot = enhanced_means.pivot(index="prompt_idx", columns="config", values="clip_rel")
    ssim_pivot = enhanced_means.pivot(index="prompt_idx", columns="config", values="ssim_rel")
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    
    sns.heatmap(clip_pivot, annot=True, fmt=".1f", cmap="RdYlGn", center=0, ax=ax1, cbar_kws={'label': '% Improvement'})
    ax1.set_title("Relative Improvement: CLIP-SIM (%)", fontweight='bold')
    ax1.set_ylabel("Prompt Index")
    ax1.set_xlabel("")
    
    sns.heatmap(ssim_pivot, annot=True, fmt=".1f", cmap="RdYlGn", center=0, ax=ax2, cbar_kws={'label': '% Improvement'})
    ax2.set_title("Relative Improvement: SSIM (%)", fontweight='bold')
    ax2.set_ylabel("")
    ax2.set_xlabel("")
    
    plt.tight_layout()
    fig_path = os.path.join(FIGURES_DIR, "fig2_prompt_heatmap.png")
    plt.savefig(fig_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved Figure 2 to {fig_path}")

def plot_box_plots(raw_df):
    print("Generating Figure 3 (Box Plots for Variance)...")
    
    metrics = ["clip_sim", "ssim"]
    titles = ["CLIP-SIM Variance Across Seeds", "SSIM Variance Across Seeds"]
    
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    
    for i, metric in enumerate(metrics):
        sns.boxplot(
            data=raw_df, 
            x="config", 
            y=metric, 
            ax=axes[i],
            palette="Set3"
        )
        sns.stripplot(
            data=raw_df, 
            x="config", 
            y=metric, 
            ax=axes[i],
            color=".25",
            alpha=0.6,
            jitter=True
        )
        axes[i].set_title(titles[i], fontweight='bold')
        axes[i].set_xlabel("")
        axes[i].set_ylabel("Score")
        axes[i].tick_params(axis='x', rotation=45)
        
    plt.tight_layout()
    fig_path = os.path.join(FIGURES_DIR, "fig3_variance_boxplots.png")
    plt.savefig(fig_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved Figure 3 to {fig_path}")

def main():
    print("=" * 60)
    print("  Generating Research Visualizations")
    print("=" * 60)
    
    raw_df, summary_df = load_data()
    
    if summary_df is not None:
        generate_latex_table(summary_df)
    else:
        print("Summary CSV not found, skipping table generation.")
        
    plot_grouped_bar_chart(raw_df)
    plot_per_prompt_heatmap(raw_df)
    plot_box_plots(raw_df)
    
    print("\nAll visualizations saved to results/figures/")

if __name__ == "__main__":
    main()

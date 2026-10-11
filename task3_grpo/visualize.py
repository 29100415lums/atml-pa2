import json
import matplotlib.pyplot as plt
from pathlib import Path
import numpy as np


def load_metrics(metrics_file):
    p = Path(metrics_file)
    if not p.exists():
        return []
    with open(p, "r") as f:
        return [json.loads(line) for line in f if line.strip()]


def load_json(json_file):
    p = Path(json_file)
    if not p.exists():
        return None
    with open(p, "r") as f:
        return json.load(f)


def plot_training_trajectories():
    metrics = load_metrics("results/task3_grpo/standard/metrics.jsonl")
    if not metrics:
        return
        
    updates = [m["update"] for m in metrics]
    rewards = [m["reward"] for m in metrics]
    kls = [m["kl"] for m in metrics]
    group_stds = [m.get("group_reward_std", 0.0) for m in metrics]
    
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8), sharex=True)
    
    # Panel 1: Reward & KL
    color_rew = "tab:blue"
    ax1.set_ylabel("Mean Reward", color=color_rew, fontsize=11, fontweight="bold")
    ax1.plot(updates, rewards, color=color_rew, marker="o", linewidth=2, label="Reward")
    ax1.tick_params(axis="y", labelcolor=color_rew)
    ax1.grid(True, linestyle="--", alpha=0.5)
    
    ax1_twin = ax1.twinx()
    color_kl = "tab:red"
    ax1_twin.set_ylabel("Sampled KL Divergence", color=color_kl, fontsize=11, fontweight="bold")
    ax1_twin.plot(updates, kls, color=color_kl, marker="s", linestyle="--", linewidth=2, label="KL")
    ax1_twin.tick_params(axis="y", labelcolor=color_kl)
    ax1.set_title("Standard GRPO Continuation: Reward and KL Dynamics", fontsize=13, fontweight="bold")
    
    # Panel 2: Within-Group Reward Standard Deviation
    color_std = "tab:purple"
    ax2.set_xlabel("GRPO Update", fontsize=11, fontweight="bold")
    ax2.set_ylabel("Group Reward Std (σ)", color=color_std, fontsize=11, fontweight="bold")
    ax2.plot(updates, group_stds, color=color_std, marker="^", linewidth=2)
    ax2.tick_params(axis="y", labelcolor=color_std)
    ax2.grid(True, linestyle="--", alpha=0.5)
    ax2.set_title("Within-Group Reward Variance (Informativeness Signal)", fontsize=11, fontweight="bold")
    
    fig.tight_layout()
    out_path = Path("results/task3_grpo/grpo_training_curves.png")
    plt.savefig(out_path, dpi=300)
    plt.close()
    print(f"Saved: {out_path}")


def plot_group_size_study():
    data = load_json("results/task3_grpo/group_size_analysis.json")
    if not data or "group_sizes" not in data:
        return
        
    group_sizes = ["2", "4", "8"]
    overall_inf = []
    easy_inf = []
    hard_inf = []
    signal_var = []
    
    for k in group_sizes:
        k_key = f"k_{k}"
        if k_key in data["group_sizes"]:
            entry = data["group_sizes"][k_key]
            overall_inf.append(entry["overall"]["informative_rate"] * 100)
            easy_inf.append(entry["easy_prompts"]["informative_rate"] * 100)
            hard_inf.append(entry["hard_prompts"]["informative_rate"] * 100)
            signal_var.append(entry["overall"]["signal_variance"])
            
    if len(overall_inf) < 3:
        return
        
    x = np.arange(len(group_sizes))
    width = 0.25
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    # Subplot 1: Informative group rate across difficulty tiers
    ax1.bar(x - width, easy_inf, width, label="Easy Prompts", color="#4CAF50", alpha=0.85)
    ax1.bar(x, overall_inf, width, label="All Prompts", color="#2196F3", alpha=0.85)
    ax1.bar(x + width, hard_inf, width, label="Hard Prompts", color="#FF9800", alpha=0.85)
    
    ax1.set_xlabel("Group Size (K)", fontsize=11, fontweight="bold")
    ax1.set_ylabel("Informative Groups (%)", fontsize=11, fontweight="bold")
    ax1.set_title("Group Informativeness vs. K by Difficulty Tier", fontsize=12, fontweight="bold")
    ax1.set_xticks(x)
    ax1.set_xticklabels([f"K={k}" for k in group_sizes])
    ax1.set_ylim(0, 105)
    ax1.grid(True, linestyle="--", alpha=0.5, axis="y")
    ax1.legend(loc="lower right")
    
    # Subplot 2: Relative Signal Variance
    ax2.plot(group_sizes, signal_var, marker="o", color="#9C27B0", linewidth=2.5, markersize=8)
    ax2.set_xlabel("Group Size (K)", fontsize=11, fontweight="bold")
    ax2.set_ylabel("Variance of Relative Advantages", fontsize=11, fontweight="bold")
    ax2.set_title("Advantage Signal Variance vs. Group Size", fontsize=12, fontweight="bold")
    ax2.grid(True, linestyle="--", alpha=0.5)
    
    fig.tight_layout()
    out_path = Path("results/task3_grpo/group_size_analysis.png")
    plt.savefig(out_path, dpi=300)
    plt.close()
    print(f"Saved: {out_path}")


def plot_length_normalization_comparison():
    canonical = load_json("results/task3_grpo/eval_norm_grpo/eval_metrics.json")
    dr_grpo = load_json("results/task3_grpo/eval_norm_dr_grpo/eval_metrics.json")
    
    if not canonical or not dr_grpo:
        return
        
    labels = ["Canonical GRPO\n(1 / T_k)", "Dr. GRPO\n(1 / L_max)"]
    lengths = [canonical["mean_length"], dr_grpo["mean_length"]]
    rewards = [canonical["mean_reward"], dr_grpo["mean_reward"]]
    kls = [canonical["mean_kl"], dr_grpo["mean_kl"]]
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 5))
    
    # Length comparison
    bars1 = ax1.bar(labels, lengths, color=["#3F51B5", "#009688"], width=0.45)
    ax1.set_ylabel("Mean Response Length (Tokens)", fontsize=11, fontweight="bold")
    ax1.set_title("Sequence Length Normalization Effect", fontsize=12, fontweight="bold")
    ax1.grid(True, linestyle="--", alpha=0.5, axis="y")
    for b in bars1:
        ax1.text(b.get_x() + b.get_width() / 2, b.get_height() + 5, f"{b.get_height():.1f}", ha="center", fontweight="bold")
        
    # Reward comparison
    bars2 = ax2.bar(labels, rewards, color=["#3F51B5", "#009688"], width=0.45)
    ax2.set_ylabel("Held-Out Reward Score", fontsize=11, fontweight="bold")
    ax2.set_title("Held-Out Reward Quality", fontsize=12, fontweight="bold")
    ax2.grid(True, linestyle="--", alpha=0.5, axis="y")
    for b in bars2:
        ax2.text(b.get_x() + b.get_width() / 2, b.get_height() + 0.02, f"{b.get_height():.3f}", ha="center", fontweight="bold")
        
    fig.tight_layout()
    out_path = Path("results/task3_grpo/length_norm_comparison.png")
    plt.savefig(out_path, dpi=300)
    plt.close()
    print(f"Saved: {out_path}")


if __name__ == "__main__":
    import os
    os.makedirs("results/task3_grpo", exist_ok=True)
    plot_training_trajectories()
    plot_group_size_study()
    plot_length_normalization_comparison()
    print("Task 3 visual analysis complete.")

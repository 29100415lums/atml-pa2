import json
import matplotlib.pyplot as plt
from pathlib import Path
import numpy as np
import os

def load_metrics(metrics_file):
    if not Path(metrics_file).exists(): return []
    with open(metrics_file, 'r') as f:
        return [json.loads(line) for line in f if line.strip()]

def load_eval(eval_file):
    if not Path(eval_file).exists(): return None
    with open(eval_file, 'r') as f:
        return json.load(f)

def plot_dpo_training():
    metrics = load_metrics("results/task1_dpo/standard/metrics.jsonl")
    if not metrics: return
    
    steps = [m["step"] for m in metrics]
    losses = [m["loss"] for m in metrics]
    accs = [m["preference_accuracy"] for m in metrics]
    
    fig, ax1 = plt.subplots(figsize=(10, 6))
    
    color = 'tab:blue'
    ax1.set_xlabel('Training Step')
    ax1.set_ylabel('DPO Loss', color=color)
    ax1.plot(steps, losses, color=color, linewidth=2)
    ax1.tick_params(axis='y', labelcolor=color)
    
    ax2 = ax1.twinx()
    color = 'tab:green'
    ax2.set_ylabel('Training Preference Accuracy', color=color)
    ax2.plot(steps, accs, color=color, alpha=0.6)
    ax2.tick_params(axis='y', labelcolor=color)
    
    plt.title('Standard DPO: Loss and Preference Accuracy over Time')
    fig.tight_layout()
    plt.savefig('results/task1_dpo/dpo_training_curves.png')
    plt.close()

def plot_beta_tradeoff():
    # standard DPO used beta=0.10. We also have 0.03 and 0.30
    eval_names = ["eval_beta_0.03", "standard_eval", "eval_beta_0.30"]
    betas = ["0.03", "0.10", "0.30"]
    
    evals = []
    for name in eval_names:
        # Fallback names if trailing zeroes were dropped
        p = Path(f"results/task1_dpo/{name}/eval_metrics.json")
        if not p.exists() and "0.30" in name:
            p = Path("results/task1_dpo/eval_beta_0.3/eval_metrics.json")
        if not p.exists() and "0.10" in name:
            p = Path("results/task1_dpo/eval_beta_0.1/eval_metrics.json")
            
        e = load_eval(p)
        if e: evals.append(e)
        
    if len(evals) < 2: return
        
    kls = [e["mean_kl"] for e in evals]
    rews = [e["mean_reward"] for e in evals]
    
    plt.figure(figsize=(8, 6))
    plt.plot(kls, rews, marker='s', markersize=10, linewidth=3, color='darkorange')
    for i, b in enumerate(betas):
        if i < len(kls):
            plt.annotate(f"beta={b}", (kls[i], rews[i]), textcoords="offset points", xytext=(0,10), ha='center')
    
    plt.xlabel("Mean KL Divergence")
    plt.ylabel("Mean Reward Model Score")
    plt.title("DPO Beta Ablation: Reward vs. KL Tradeoff")
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.savefig('results/task1_dpo/dpo_beta_tradeoff.png')
    plt.close()

def plot_length_bias():
    std_eval = load_eval("results/task1_dpo/standard_eval/eval_metrics.json")
    len_eval = load_eval("results/task1_dpo/eval_length/eval_metrics.json")
    
    if not std_eval or not len_eval: return
    
    labels = ['Standard DPO', 'Length-Balanced DPO']
    lengths = [std_eval['mean_length'], len_eval['mean_length']]
    rewards = [std_eval['mean_reward'], len_eval['mean_reward']]
    
    x = np.arange(len(labels))
    width = 0.35

    fig, ax1 = plt.subplots(figsize=(8, 6))
    
    ax1.bar(x - width/2, lengths, width, label='Mean Length (Tokens)', color='skyblue')
    ax1.set_ylabel('Mean Response Length')
    ax1.set_xticks(x)
    ax1.set_xticklabels(labels)
    
    ax2 = ax1.twinx()
    ax2.bar(x + width/2, rewards, width, label='Mean Reward', color='salmon')
    ax2.set_ylabel('Mean Reward Score')
    
    # Combine legends
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax2.legend(lines1 + lines2, labels1 + labels2, loc='upper right')

    plt.title('Impact of Length-Stratified Dataset on Verbosity and Reward')
    fig.tight_layout()
    plt.savefig('results/task1_dpo/dpo_length_bias.png')
    plt.close()

if __name__ == "__main__":
    os.makedirs("results/task1_dpo", exist_ok=True)
    plot_dpo_training()
    plot_beta_tradeoff()
    plot_length_bias()
    print("Visualizations generated in results/task1_dpo/")

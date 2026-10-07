import json
import matplotlib.pyplot as plt
from pathlib import Path
import numpy as np

def load_metrics(metrics_file):
    if not Path(metrics_file).exists(): return []
    with open(metrics_file, 'r') as f:
        return [json.loads(line) for line in f if line.strip()]

def load_eval(eval_file):
    if not Path(eval_file).exists(): return None
    with open(eval_file, 'r') as f:
        return json.load(f)

def plot_standard_training():
    metrics = load_metrics("results/task2_ppo/standard/metrics.jsonl")
    if not metrics: return
    
    updates = [m["update"] for m in metrics]
    rewards = [m["learned_reward"] for m in metrics]
    kls = [m["kl"] for m in metrics]
    
    fig, ax1 = plt.subplots(figsize=(10, 6))
    
    color = 'tab:blue'
    ax1.set_xlabel('PPO Update')
    ax1.set_ylabel('Learned Task Reward', color=color)
    ax1.plot(updates, rewards, color=color, marker='o', linewidth=2)
    ax1.tick_params(axis='y', labelcolor=color)
    
    ax2 = ax1.twinx()
    color = 'tab:red'
    ax2.set_ylabel('KL Divergence', color=color)
    ax2.plot(updates, kls, color=color, marker='x', linestyle='--', linewidth=2)
    ax2.tick_params(axis='y', labelcolor=color)
    
    plt.title('Standard PPO: Reward and KL Divergence over Time')
    fig.tight_layout()
    plt.savefig('results/task2_ppo/ppo_training_curves.png')
    plt.close()

def plot_kl_tradeoff():
    betas = ["0.0", "0.10", "0.20"] # matching standard fork configs
    # fallback to actual folder names
    folders = [Path("results/task2_ppo")/f"kl_beta_{b}" for b in betas]
    # some might drop zero
    folders = [f if f.exists() else Path(str(f).replace("0.10","0.1").replace("0.20","0.2")) for f in folders]
    
    evals = []
    for f in folders:
        e = load_eval(f.parent / f"eval_kl_{f.name.split('_')[-1]}/eval_metrics.json")
        if e: evals.append(e)
        
    if len(evals) < 2: return
        
    kls = [e["mean_kl"] for e in evals]
    rews = [e["mean_reward"] for e in evals]
    
    plt.figure(figsize=(8, 6))
    plt.plot(kls, rews, marker='o', markersize=10, linewidth=3, color='purple')
    for i, b in enumerate(betas):
        if i < len(kls):
            plt.annotate(f"beta={b}", (kls[i], rews[i]), textcoords="offset points", xytext=(0,10), ha='center')
    
    plt.xlabel("Mean KL Divergence")
    plt.ylabel("Mean Reward Model Score")
    plt.title("Reward vs. KL Tradeoff (Overoptimization Study)")
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.savefig('results/task2_ppo/kl_reward_tradeoff.png')
    plt.close()

def plot_clipping_impact():
    epsilons = ["0.05", "0.20", "0.50"]
    folders = [Path("results/task2_ppo")/f"clip_{e}" for e in epsilons]
    folders = [f if f.exists() else Path(str(f).replace("0.20","0.2").replace("0.50","0.5")) for f in folders]
    
    clip_fracs = []
    for f in folders:
        m = load_metrics(f / "metrics.jsonl")
        if m:
            clip_fracs.append(np.mean([x["clip_fraction"] for x in m]))
            
    if len(clip_fracs) < len(epsilons): return
    
    plt.figure(figsize=(8, 6))
    plt.bar(epsilons, clip_fracs, color='teal', alpha=0.8)
    plt.xlabel("Clip Epsilon")
    plt.ylabel("Average Clipped Token Fraction")
    plt.title("Impact of PPO Clip Epsilon on Token Clipping")
    for i, v in enumerate(clip_fracs):
        plt.text(i, v + 0.01, f"{v:.3f}", ha='center', fontweight='bold')
    plt.savefig('results/task2_ppo/clipping_impact.png')
    plt.close()

if __name__ == "__main__":
    import os
    os.makedirs("results/task2_ppo", exist_ok=True)
    plot_standard_training()
    plot_kl_tradeoff()
    plot_clipping_impact()
    print("Visualizations generated in results/task2_ppo/")

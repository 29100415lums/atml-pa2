from __future__ import annotations

import argparse
import json
from collections import defaultdict
import numpy as np

from common.data import load_yaml, read_jsonl, repo_path


def load_k8_cache(path: str):
    rows = read_jsonl(path)
    by_prompt = defaultdict(list)
    for row in rows:
        by_prompt[str(row["source_index"])].append(row)
        
    bad = {pid: len(group) for pid, group in by_prompt.items() if len(group) < 8}
    if bad:
        raise ValueError(f"Expected at least K=8 cached completions per prompt; short groups: {bad}")
        
    for group in by_prompt.values():
        group.sort(key=lambda x: int(x.get("generation_index", 0)))
    return by_prompt


def regroup_equal_generation_budget(by_prompt, k: int):
    """Partition the cached completions into groups of size K holding total completions fixed."""
    groups_with_pid = []
    for pid, completions in by_prompt.items():
        # Partition the 8 completions into groups of size K
        for i in range(0, len(completions), k):
            if i + k <= len(completions):
                groups_with_pid.append((pid, completions[i : i + k]))
    return groups_with_pid


def evaluate_groups(groups_list, eps: float = 1e-6):
    """Compute informative rate, reward std, and relative advantage variance for a set of groups."""
    if not groups_list:
        return {"informative_rate": 0.0, "mean_reward_std": 0.0, "signal_variance": 0.0}
        
    informative_count = 0
    reward_stds = []
    all_advantages = []
    
    for _, group in groups_list:
        rewards = np.array([float(c["reward"]) for c in group], dtype=np.float64)
        r_min, r_max = rewards.min(), rewards.max()
        r_std = float(rewards.std(ddof=0))
        reward_stds.append(r_std)
        
        if (r_max - r_min) > 1e-7:
            informative_count += 1
            adv = (rewards - rewards.mean()) / (r_std + eps)
        else:
            adv = np.zeros_like(rewards)
            
        all_advantages.extend(adv.tolist())
        
    inf_rate = informative_count / len(groups_list)
    mean_std = float(np.mean(reward_stds))
    sig_var = float(np.var(all_advantages))
    
    return {
        "informative_rate": float(inf_rate),
        "mean_reward_std": float(mean_std),
        "signal_variance": float(sig_var),
        "num_groups": len(groups_list)
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/grpo.yaml")
    args = ap.parse_args()
    cfg = load_yaml(args.config)
    by_prompt = load_k8_cache(cfg["group_cache"])
    
    print(f"\n--- Group-Size Study from Rollout Cache ({len(by_prompt)} prompts) ---")
    print(f"Group sizes analyzed: {cfg['group_sizes']}")
    
    # 1. Define prompt difficulty bins using median prompt reward
    prompt_mean_rewards = {
        pid: float(np.mean([c["reward"] for c in comps]))
        for pid, comps in by_prompt.items()
    }
    all_means = list(prompt_mean_rewards.values())
    median_reward = float(np.median(all_means))
    print(f"Difficulty split threshold (median prompt reward): {median_reward:.4f}")
    
    results = {
        "median_reward_threshold": median_reward,
        "group_sizes": {}
    }
    
    for k in cfg["group_sizes"]:
        k = int(k)
        groups_with_pid = regroup_equal_generation_budget(by_prompt, k)
        
        # Split into Easy (>= median) and Hard (< median)
        easy_groups = [g for g in groups_with_pid if prompt_mean_rewards[g[0]] >= median_reward]
        hard_groups = [g for g in groups_with_pid if prompt_mean_rewards[g[0]] < median_reward]
        
        overall_stats = evaluate_groups(groups_with_pid)
        easy_stats = evaluate_groups(easy_groups)
        hard_stats = evaluate_groups(hard_groups)
        
        print(f"\n>> Group Size K = {k} (Total Groups: {len(groups_with_pid)}):")
        print(f"   [Overall] Informative Rate: {overall_stats['informative_rate']:.1%}, "
              f"Mean Reward Std: {overall_stats['mean_reward_std']:.4f}, Signal Var: {overall_stats['signal_variance']:.4f}")
        print(f"   [Easy]    Informative Rate: {easy_stats['informative_rate']:.1%}, Mean Reward Std: {easy_stats['mean_reward_std']:.4f}")
        print(f"   [Hard]    Informative Rate: {hard_stats['informative_rate']:.1%}, Mean Reward Std: {hard_stats['mean_reward_std']:.4f}")
        
        results["group_sizes"][f"k_{k}"] = {
            "overall": overall_stats,
            "easy_prompts": easy_stats,
            "hard_prompts": hard_stats
        }
        
    out_dir = repo_path(cfg["results_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)
    out_file = out_dir / "group_size_analysis.json"
    
    with open(out_file, "w") as f:
        json.dump(results, f, indent=2)
        
    print(f"\nGroup size analysis successfully saved to {out_file}")


if __name__ == "__main__":
    main()

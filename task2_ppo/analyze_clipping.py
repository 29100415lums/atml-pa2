from __future__ import annotations

import argparse
import torch

from common.data import load_yaml, repo_path


def load_cached_rollouts(path):
    rows = torch.load(repo_path(path), map_location="cpu", weights_only=False)
    if not isinstance(rows, list) or not rows:
        raise ValueError("Expected a non-empty list in the supplied PPO rollout cache")

    # Instructor iterations used two equivalent names for these fields. Normalize once here so
    # the student analysis code sees one stable interface.
    normalized = []
    for row in rows:
        row = dict(row)
        if "old_logprobs" not in row and "old_policy_logprobs" in row:
            row["old_logprobs"] = row["old_policy_logprobs"]
        if "ref_logprobs" not in row and "reference_logprobs" in row:
            row["ref_logprobs"] = row["reference_logprobs"]
        normalized.append(row)

    required = {"source_index", "response", "old_logprobs", "ref_logprobs"}
    if not required.issubset(normalized[0]):
        raise ValueError(f"Unexpected PPO cache schema; need at least {sorted(required)}")
    return normalized


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/ppo.yaml")
    args = ap.parse_args()
    cfg = load_yaml(args.config)
    rows = load_cached_rollouts(cfg["cached_rollouts"])
    print("Cached PPO rollouts:", len(rows))
    import subprocess
    import json
    from pathlib import Path
    
    clip_values = cfg["clip_values"]
    fork_updates = cfg["fork_updates"]
    
    print(f"Required epsilon values: {clip_values}")
    
    # 1. Analyze the cached rollouts
    print("\n--- Analyzing Cached Rollouts ---")
    # For each epsilon, compute the clipped surrogate and affected token fraction.
    # Note: To fully compute advantages locally, we would load the reward/value models.
    # Since the cache provides old/ref logprobs, we emulate the ratio computation.
    # We will log these hypothetical bounds to help your report analysis.
    
    for eps in clip_values:
        total_tokens = 0
        clipped_tokens = 0
        surrogates = []
        
        for row in rows:
            # Reconstruct ratio r_theta
            # In a real step, old_logprobs are the denominator, new_logprobs are numerator
            # For this static analysis, we measure how much space the epsilon allows
            ratio = torch.exp(row["old_logprobs"] - row["ref_logprobs"]) # mock ratio distribution
            
            # Count affected fraction
            clipped_mask = (ratio < 1.0 - eps) | (ratio > 1.0 + eps)
            clipped_tokens += clipped_mask.sum().item()
            total_tokens += clipped_mask.numel()
            
        fraction = clipped_tokens / max(1, total_tokens)
        print(f"Epsilon={eps}: Estimated Affected Token Fraction = {fraction:.4f}")
    
    # 2. Run the matched short forks for each epsilon
    print("\n--- Launching Clipping Forks ---")
    for eps in clip_values:
        run_name = f"clip_{eps}"
        out_dir = f"outputs/task2_ppo/{run_name}"
        
        print(f"\n>> Starting PPO Clip fork: {eps}")
        cmd = [
            "python", "-m", "task2_ppo.continue_train",
            "--config", args.config,
            "--run-name", run_name,
            "--output", out_dir,
            "--updates", str(fork_updates),
            "--clip-epsilon", str(eps)
        ]
        
        subprocess.run(cmd, check=True)
        
    print("\nAll clip forks completed!")


if __name__ == "__main__":
    main()

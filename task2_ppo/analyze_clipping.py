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
    
    clip_values = cfg["clip_values"]
    fork_updates = cfg["fork_updates"]
    
    print(f"Required epsilon values: {clip_values}")
    
    # 1. Run the matched short forks for each epsilon
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

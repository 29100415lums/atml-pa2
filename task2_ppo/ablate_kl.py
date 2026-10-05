from __future__ import annotations

import argparse
from common.data import load_yaml


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/ppo.yaml")
    args = ap.parse_args()
    cfg = load_yaml(args.config)
    import subprocess
    
    kl_values = cfg["kl_values"]
    fork_updates = cfg["fork_updates"]
    
    print(f"KL beta conditions: {kl_values}")
    print(f"Fork update budget: {fork_updates}")
    
    for kl in kl_values:
        run_name = f"kl_beta_{kl}"
        out_dir = f"outputs/task2_ppo/{run_name}"
        
        print(f"\n--- Starting PPO KL fork: {kl} ---")
        cmd = [
            "python", "-m", "task2_ppo.continue_train",
            "--config", args.config,
            "--run-name", run_name,
            "--output", out_dir,
            "--updates", str(fork_updates),
            "--kl-beta", str(kl)
        ]
        
        subprocess.run(cmd, check=True)
        
    print("\nAll KL forks completed!")


if __name__ == "__main__":
    main()

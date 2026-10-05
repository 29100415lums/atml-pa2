from __future__ import annotations

import argparse
from common.data import load_yaml


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/dpo.yaml")
    args = ap.parse_args()
    cfg = load_yaml(args.config)
    import subprocess
    
    betas = cfg["betas"]
    short_examples = cfg["short_ablation_examples"]
    
    print(f"Launching beta sweeps for betas: {betas} with {short_examples} examples.")
    
    for beta in betas:
        run_name = f"beta_{beta}"
        output_dir = f"outputs/task1_dpo/{run_name}"
        
        print(f"\n--- Starting run for beta={beta} ---")
        cmd = [
            "python", "-m", "task1_dpo.train",
            "--config", args.config,
            "--run-name", run_name,
            "--output", output_dir,
            "--beta", str(beta),
            "--max-examples", str(short_examples)
        ]
        
        subprocess.run(cmd, check=True)
        
    print("\nAll beta forks completed successfully!")


if __name__ == "__main__":
    main()

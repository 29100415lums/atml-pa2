from __future__ import annotations

import argparse
from common.data import load_yaml, read_jsonl


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/dpo.yaml")
    args = ap.parse_args()
    cfg = load_yaml(args.config)
    balanced = read_jsonl(cfg["paths"]["dpo_length_train"])
    stratified = read_jsonl(cfg["paths"]["dpo_length_eval"])
    import subprocess
    
    balanced_path = cfg["paths"]["dpo_length_train"]
    strat_path = cfg["paths"]["dpo_length_eval"]
    
    print(f"Length-balanced train rows: {len(balanced)}")
    print(f"Length-stratified eval rows: {len(stratified)}")
    
    run_name = "length_balanced"
    output_dir = f"outputs/task1_dpo/{run_name}"
    
    print(f"\n--- Starting length-balanced training ---")
    cmd = [
        "python", "-m", "task1_dpo.train",
        "--config", args.config,
        "--run-name", run_name,
        "--output", output_dir,
        "--dataset", balanced_path
    ]
    
    subprocess.run(cmd, check=True)
    
    print("\nTraining completed! Remember to evaluate this model using task1_dpo.evaluate.")


if __name__ == "__main__":
    main()

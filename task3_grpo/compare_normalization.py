from __future__ import annotations

import argparse
from common.data import load_yaml
from task3_grpo.continue_train import run_grpo


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/grpo.yaml")
    args = ap.parse_args()
    cfg = load_yaml(args.config)
    fork_updates = int(cfg.get("fork_updates", 8))
    
    print(f"\n=======================================================")
    print(f"  Length-Normalization Study: Canonical GRPO vs. Dr. GRPO")
    print(f"  Matched short continuations: {fork_updates} updates each")
    print(f"=======================================================\n")
    
    # 1. Canonical GRPO (sequence-length normalized 1/T_k)
    print("\n--- [Fork 1/2] Canonical GRPO (loss_type='grpo') ---")
    run_grpo(
        config_path=args.config,
        output="outputs/task3_grpo/norm_grpo",
        updates=fork_updates,
        loss_type="grpo",
        run_name="norm_grpo",
    )
    
    # 2. Dr. GRPO (constant-length normalized 1/L_max)
    print("\n--- [Fork 2/2] Dr. GRPO (loss_type='dr_grpo') ---")
    run_grpo(
        config_path=args.config,
        output="outputs/task3_grpo/norm_dr_grpo",
        updates=fork_updates,
        loss_type="dr_grpo",
        run_name="norm_dr_grpo",
    )
    
    print("\nBoth normalization forks successfully finished!")


if __name__ == "__main__":
    main()

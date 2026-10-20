from __future__ import annotations

import argparse
import json
from pathlib import Path
import pandas as pd

from common.data import load_yaml, repo_path


def main():
    ap = argparse.ArgumentParser(description="Synthesize in-domain, transfer, and diagnostic results for Task 5")
    ap.add_argument("--config", default="configs/feedback.yaml")
    args = ap.parse_args()

    cfg = load_yaml(args.config)
    outdir = repo_path(cfg.get("results_dir", "results")) / "task5_feedback"

    gsm_file = outdir / "math_eval_gsm.json"
    transfer_file = outdir / "math_eval_transfer.json"
    diag_file = outdir / "diagnostic_metrics.json"

    if not gsm_file.exists():
        print(f"Warning: {gsm_file} not found. Run evaluate_math.py --dataset gsm first.")
        return
    if not transfer_file.exists():
        print(f"Warning: {transfer_file} not found. Run evaluate_math.py --dataset transfer first.")
        return

    with open(gsm_file, "r", encoding="utf-8") as f:
        gsm_data = json.load(f)
    with open(transfer_file, "r", encoding="utf-8") as f:
        transfer_data = json.load(f)

    diag_data = None
    if diag_file.exists():
        with open(diag_file, "r", encoding="utf-8") as f:
            diag_data = json.load(f)

    print("=" * 75)
    print("TASK 5: RLVR VERSUS RLAIF COMPREHENSIVE SYNTHESIS")
    print("=" * 75)

    policies = ["sft", "rlvr", "rlaif"]
    master_rows = []

    for pol in policies:
        g_pol = gsm_data["policies"].get(pol, {})
        t_pol = transfer_data["policies"].get(pol, {})

        g_acc = g_pol.get("exact_accuracy", 0.0)
        t_acc = t_pol.get("exact_accuracy", 0.0)
        acc_drop = (t_acc - g_acc) * 100

        g_fmt = g_pol.get("format_compliance", 0.0) * 100
        t_fmt = t_pol.get("format_compliance", 0.0) * 100

        g_len = g_pol.get("mean_length", 0.0)

        # Win rate vs SFT
        if pol == "sft":
            gsm_win = 50.0
            trans_win = 50.0
        elif pol == "rlvr":
            gsm_win = gsm_data.get("pairwise_rlvr_vs_sft", {}).get("win_rate_rlvr_vs_sft", 0.5) * 100
            trans_win = transfer_data.get("pairwise_rlvr_vs_sft", {}).get("win_rate_rlvr_vs_sft", 0.5) * 100
        elif pol == "rlaif":
            gsm_win = gsm_data.get("pairwise_rlaif_vs_sft", {}).get("win_rate_rlaif_vs_sft", 0.5) * 100
            trans_win = transfer_data.get("pairwise_rlaif_vs_sft", {}).get("win_rate_rlaif_vs_sft", 0.5) * 100

        master_rows.append({
            "Policy": pol.upper(),
            "GSM8K Acc (%)": f"{g_acc * 100:.2f}%",
            "SVAMP Transfer Acc (%)": f"{t_acc * 100:.2f}%",
            "Transfer Drop (pp)": f"{acc_drop:+.2f}",
            "GSM8K Win Rate vs SFT (%)": f"{gsm_win:.2f}%",
            "SVAMP Win Rate vs SFT (%)": f"{trans_win:.2f}%",
            "Format Compliance (%)": f"{g_fmt:.1f}%",
            "Mean Length": f"{g_len:.1f}",
        })

    master_df = pd.DataFrame(master_rows)
    print("\n--- Master RLVR vs RLAIF Performance Table ---")
    print(master_df.to_string(index=False))
    master_df.to_csv(outdir / "master_feedback_table.csv", index=False)

    synthesis = {
        "master_table": master_rows,
        "gsm_eval": gsm_data,
        "transfer_eval": transfer_data,
        "diagnostics": diag_data,
        "feedback_properties": {
            "RLVR": {
                "coverage": "Sparse outcome-only (binary 0 or 1 on final answer)",
                "noise": "Zero noise (exact parser against ground truth)",
                "exploitability": "Vulnerable to reward hacking with gold distractor or brute-forcing; blind to faulty reasoning",
                "inference_cost": "Negligible (regex pattern matching, < 1ms)",
            },
            "RLAIF": {
                "coverage": "Dense and comparative (evaluates reasoning steps, tone, structure)",
                "noise": "Moderate judge variance / stochasticity; subject to positional and length biases",
                "exploitability": "Vulnerable to persuasive filler and verbose confidence hacking",
                "inference_cost": "High (calls 3B LLM autoregressively for every pair)",
            },
        },
    }

    out_json = outdir / "feedback_synthesis_summary.json"
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(synthesis, f, indent=2)
    print(f"\nSaved comprehensive feedback synthesis to {out_json}")


if __name__ == "__main__":
    main()

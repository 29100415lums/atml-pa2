from __future__ import annotations

import argparse
import json
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from common.data import load_yaml, repo_path


def plot_transfer_accuracy(summary_data: dict, out_dir: Path):
    """Plot in-domain vs transfer accuracy and performance drops."""
    master = summary_data.get("master_table", [])
    if not master:
        return

    policies = [r["Policy"] for r in master]
    gsm_acc = [float(r["GSM8K Acc (%)"].replace("%", "")) for r in master]
    svamp_acc = [float(r["SVAMP Transfer Acc (%)"].replace("%", "")) for r in master]

    x = np.arange(len(policies))
    width = 0.35

    plt.figure(figsize=(9, 5))
    plt.bar(x - width/2, gsm_acc, width, label="In-Domain (GSM8K)", color="#1f77b4")
    plt.bar(x + width/2, svamp_acc, width, label="Out-of-Domain (SVAMP Transfer)", color="#ff7f0e")

    plt.ylabel("Exact Accuracy (%)", fontsize=12)
    plt.title("Mathematical Reasoning: In-Domain vs Transfer Generalization", fontsize=14, fontweight="bold")
    plt.xticks(x, policies, fontweight="bold", fontsize=11)
    plt.ylim(0, 100)
    plt.legend(fontsize=11)
    plt.grid(axis="y", linestyle="--", alpha=0.5)

    # Annotate drops
    for i, (g, s) in enumerate(zip(gsm_acc, svamp_acc)):
        drop = s - g
        plt.annotate(
            f"{drop:+.1f}%",
            (x[i] + width/2, s + 2),
            ha="center",
            fontsize=10,
            fontweight="bold",
            color="#d62728" if drop < 0 else "#2ca02c",
        )

    plt.tight_layout()
    save_path = out_dir / "math_transfer_comparison.png"
    plt.savefig(save_path, dpi=300)
    plt.close()
    print(f"Saved math transfer comparison plot to {save_path}")


def plot_diagnostic_sensitivities(diag_data: dict, out_dir: Path):
    """Plot comparison of Exact Verifier vs AI Judge on diagnostic perturbations."""
    if not diag_data:
        return

    categories = [
        "Reasoning Sensitivity\n(S_reason)",
        "Outcome Sensitivity\n(S_outcome)",
        "Filler Resistance\n(1 - Filler Bias)",
        "Distractor Robustness\n(1 - Distractor Bias)",
    ]

    rlvr_scores = [
        diag_data["reasoning_sensitivity"]["rates"]["rlvr"]["better_rate"] * 100,
        diag_data["outcome_sensitivity"]["rates"]["rlvr"]["better_rate"] * 100,
        (1.0 - diag_data["filler_susceptibility"]["rates"]["rlvr"]["wrong_preference_rate"]) * 100,
        (1.0 - diag_data["distractor_robustness"]["rates"]["rlvr"]["wrong_preference_rate"]) * 100,
    ]

    rlaif_scores = [
        diag_data["reasoning_sensitivity"]["rates"]["rlaif"]["better_rate"] * 100,
        diag_data["outcome_sensitivity"]["rates"]["rlaif"]["better_rate"] * 100,
        (1.0 - diag_data["filler_susceptibility"]["rates"]["rlaif"]["wrong_preference_rate"]) * 100,
        (1.0 - diag_data["distractor_robustness"]["rates"]["rlaif"]["wrong_preference_rate"]) * 100,
    ]

    x = np.arange(len(categories))
    width = 0.35

    plt.figure(figsize=(10, 5.5))
    plt.bar(x - width/2, rlvr_scores, width, label="Exact Verifier (RLVR)", color="#2ca02c")
    plt.bar(x + width/2, rlaif_scores, width, label="Pairwise AI Judge (RLAIF)", color="#9467bd")

    plt.ylabel("Score (%)", fontsize=12)
    plt.title("Reward Mechanism Sensitivity & Robustness across Controlled Perturbations", fontsize=13, fontweight="bold")
    plt.xticks(x, categories, fontsize=10, fontweight="bold")
    plt.ylim(0, 105)
    plt.legend(fontsize=11)
    plt.grid(axis="y", linestyle="--", alpha=0.5)

    plt.tight_layout()
    save_path = out_dir / "diagnostic_sensitivity_comparison.png"
    plt.savefig(save_path, dpi=300)
    plt.close()
    print(f"Saved diagnostic sensitivity plot to {save_path}")


def main():
    ap = argparse.ArgumentParser(description="Generate publication figures for Task 5 RLVR vs RLAIF")
    ap.add_argument("--config", default="configs/feedback.yaml")
    args = ap.parse_args()

    cfg = load_yaml(args.config)
    outdir = repo_path(cfg.get("results_dir", "results")) / "task5_feedback"

    summary_file = outdir / "feedback_synthesis_summary.json"
    if not summary_file.exists():
        print(f"No {summary_file} found. Run compare_feedback.py first.")
        return

    with open(summary_file, "r", encoding="utf-8") as f:
        summary_data = json.load(f)

    plot_transfer_accuracy(summary_data, outdir)
    if "diagnostics" in summary_data and summary_data["diagnostics"]:
        plot_diagnostic_sensitivities(summary_data["diagnostics"], outdir)

    print("Task 5 visualizations completed successfully.")


if __name__ == "__main__":
    main()

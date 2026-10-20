from __future__ import annotations

import argparse
import json
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from common.data import load_yaml, repo_path


def plot_safety_calibration(metrics: dict, out_dir: Path):
    """Plot safe answer rate vs unsafe compliance rate and over-refusal vs justified refusal."""
    policies = [p for p in ["sft", "dpo", "ppo", "grpo"] if p in metrics]
    if not policies:
        return

    labels = [p.upper() for p in policies]
    safe_ans = [metrics[p]["safe_answer_rate"] * 100 for p in policies]
    unsafe_comp = [metrics[p]["unsafe_compliance_rate"] * 100 for p in policies]
    over_ref = [metrics[p]["safe_over_refusal_rate"] * 100 for p in policies]
    just_ref = [metrics[p]["justified_refusal_rate"] * 100 for p in policies]

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    # Subplot 1: Helpful-Harmless Frontier (Safe Answer vs Unsafe Compliance)
    x = np.arange(len(labels))
    width = 0.35
    axes[0].bar(x - width/2, safe_ans, width, label="Safe Answer Rate (%)", color="#2ca02c")
    axes[0].bar(x + width/2, unsafe_comp, width, label="Unsafe Compliance Rate (%)", color="#d62728")
    axes[0].set_ylabel("Percentage (%)")
    axes[0].set_title("Helpful vs Harmless Alignment Frontier\n(Higher Safe Answer is better, Lower Unsafe is better)")
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(labels, fontweight="bold")
    axes[0].set_ylim(0, 105)
    axes[0].legend(loc="upper right")
    axes[0].grid(axis="y", linestyle="--", alpha=0.5)

    # Subplot 2: Refusal Calibration (Justified Refusal vs Over-Refusal)
    axes[1].bar(x - width/2, just_ref, width, label="Justified Refusal on Unsafe (%)", color="#1f77b4")
    axes[1].bar(x + width/2, over_ref, width, label="Over-Refusal on Safe (%)", color="#ff7f0e")
    axes[1].set_ylabel("Percentage (%)")
    axes[1].set_title("Refusal Calibration\n(Higher Justified Refusal, Lower Over-Refusal)")
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(labels, fontweight="bold")
    axes[1].set_ylim(0, 105)
    axes[1].legend(loc="upper right")
    axes[1].grid(axis="y", linestyle="--", alpha=0.5)

    plt.tight_layout()
    save_path = out_dir / "safety_calibration_tradeoff.png"
    plt.savefig(save_path, dpi=300)
    plt.close()
    print(f"Saved safety calibration plot to {save_path}")


def plot_category_breakdown(cat_csv_path: Path, out_dir: Path):
    """Plot refusal rate across categories for all policies."""
    if not cat_csv_path.exists():
        return
    df = pd.read_csv(cat_csv_path)
    if df.empty:
        return

    # Focus on Over-Refusal Rate across top categories
    top_cats = df.groupby("Category")["Total"].sum().nlargest(10).index
    sub_df = df[df["Category"].isin(top_cats)]

    pivot = sub_df.pivot(index="Category", columns="Policy", values="Over-Refusal Rate") * 100
    if pivot.empty:
        return

    plt.figure(figsize=(12, 6))
    pivot.plot(kind="bar", figsize=(12, 6), colormap="viridis", width=0.8)
    plt.title("Exaggerated Over-Refusal Rate across Sensitive XSTest Categories (%)", fontsize=14, fontweight="bold")
    plt.ylabel("Over-Refusal Rate (%)", fontsize=12)
    plt.xlabel("Category", fontsize=12)
    plt.xticks(rotation=45, ha="right")
    plt.ylim(0, 100)
    plt.grid(axis="y", linestyle="--", alpha=0.5)
    plt.legend(title="Policy")
    plt.tight_layout()

    save_path = out_dir / "category_over_refusal.png"
    plt.savefig(save_path, dpi=300)
    plt.close()
    print(f"Saved category over-refusal plot to {save_path}")


def main():
    ap = argparse.ArgumentParser(description="Generate publication figures for Task 4 Safety Calibration")
    ap.add_argument("--config", default="configs/feedback.yaml")
    args = ap.parse_args()

    cfg = load_yaml(args.config)
    outdir = repo_path(cfg.get("results_dir", "results")) / "task4_safety"
    json_path = outdir / "safety_metrics.json"

    if not json_path.exists():
        print(f"No {json_path} found. Run evaluate_safety.py first.")
        return

    with open(json_path, "r", encoding="utf-8") as f:
        metrics = json.load(f)

    plot_safety_calibration(metrics, outdir)
    plot_category_breakdown(outdir / "category_breakdown.csv", outdir)
    print("Task 4 visualizations completed successfully.")


if __name__ == "__main__":
    main()

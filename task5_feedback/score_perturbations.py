from __future__ import annotations

import argparse
from collections import defaultdict
import json
from pathlib import Path
import numpy as np
import pandas as pd

from common.data import load_yaml, read_jsonl, repo_path
from task5_feedback.rlvr import exact_reward
from task5_feedback.rlaif import PairwiseAIJudge

EXPECTED_VARIANTS = {
    "clean_correct",
    "corrupt_reasoning_correct_final",
    "good_reasoning_wrong_final",
    "persuasive_filler_correct",
    "gold_distractor_wrong_final",
}


def load_diagnostic_groups(path):
    rows = read_jsonl(path)
    by_problem = defaultdict(dict)
    for row in rows:
        by_problem[str(row["problem_id"])][row["variant_type"]] = row
    for pid, variants in by_problem.items():
        missing = EXPECTED_VARIANTS - set(variants)
        if missing:
            raise ValueError(f"Problem {pid} missing variants: {sorted(missing)}")
    return by_problem


def evaluate_pair(judge: PairwiseAIJudge, problem: str, gold: str, resp_better: str, resp_worse: str):
    """Evaluate a pair (better vs worse) using both Exact Verifier and AI Judge."""
    # 1. Exact Verifier (RLVR)
    rew_better = exact_reward(resp_better, gold)
    rew_worse = exact_reward(resp_worse, gold)

    if rew_better > rew_worse:
        rlvr_pref = "BETTER"
    elif rew_worse > rew_better:
        rlvr_pref = "WORSE"
    else:
        rlvr_pref = "TIE"

    # 2. AI Pairwise Judge (RLAIF)
    comp = judge.compare(problem, resp_better, resp_worse)
    if comp == "A":
        rlaif_pref = "BETTER"
    elif comp == "B":
        rlaif_pref = "WORSE"
    else:
        rlaif_pref = "TIE"

    return {"rlvr": rlvr_pref, "rlaif": rlaif_pref}


def compute_rates(results: list[dict]):
    """Compute better, tie, and wrong-preference rates for both mechanisms."""
    n = len(results)
    if n == 0:
        return {}

    rates = {}
    for mech in ["rlvr", "rlaif"]:
        prefs = [r[mech] for r in results]
        rates[mech] = {
            "better_rate": float(prefs.count("BETTER") / n),
            "tie_rate": float(prefs.count("TIE") / n),
            "wrong_preference_rate": float(prefs.count("WORSE") / n),
        }
    return rates


def main():
    ap = argparse.ArgumentParser(description="Evaluate controlled reward diagnostic set for Task 5")
    ap.add_argument("--config", default="configs/feedback.yaml")
    args = ap.parse_args()

    cfg = load_yaml(args.config)
    diag_path = cfg["paths"]["task5_diagnostics"]
    groups = load_diagnostic_groups(diag_path)
    print("Diagnostic problems:", len(groups))
    print("Variants/problem:", sorted(EXPECTED_VARIANTS))

    outdir = repo_path(cfg.get("results_dir", "results")) / "task5_feedback"
    outdir.mkdir(parents=True, exist_ok=True)
    cache_path = outdir / "judge_cache_diagnostics.json"
    judge = PairwiseAIJudge(cfg, cache_path=cache_path)

    # 1. Reasoning Sensitivity: clean_correct vs corrupt_reasoning_correct_final
    reasoning_pairs = []
    # 2. Outcome Sensitivity: clean_correct vs good_reasoning_wrong_final
    outcome_pairs = []
    # 3. Filler Susceptibility: clean_correct (better) vs persuasive_filler_correct (worse)
    filler_pairs = []
    # 4. Distractor Robustness: good_reasoning_wrong_final (better) vs gold_distractor_wrong_final (worse)
    distractor_pairs = []

    for pid, variants in groups.items():
        problem = variants["clean_correct"]["question"]
        gold = str(variants["clean_correct"]["gold_final"])

        clean = variants["clean_correct"]["response"]
        corrupt = variants["corrupt_reasoning_correct_final"]["response"]
        wrong_final = variants["good_reasoning_wrong_final"]["response"]
        filler = variants["persuasive_filler_correct"]["response"]
        distractor = variants["gold_distractor_wrong_final"]["response"]

        # Reasoning pair: clean vs corrupt
        reasoning_pairs.append(evaluate_pair(judge, problem, gold, clean, corrupt))

        # Outcome pair: clean vs wrong_final
        outcome_pairs.append(evaluate_pair(judge, problem, gold, clean, wrong_final))

        # Filler pair: clean vs filler
        filler_pairs.append(evaluate_pair(judge, problem, gold, clean, filler))

        # Distractor pair: wrong_final vs distractor
        distractor_pairs.append(evaluate_pair(judge, problem, gold, wrong_final, distractor))

    # Compute metric summaries
    reasoning_rates = compute_rates(reasoning_pairs)
    outcome_rates = compute_rates(outcome_pairs)
    filler_rates = compute_rates(filler_pairs)
    distractor_rates = compute_rates(distractor_pairs)

    summary = {
        "n_problems": len(groups),
        "reasoning_sensitivity": {
            "description": "S_reason: clean_correct vs corrupt_reasoning_correct_final",
            "rates": reasoning_rates,
            "S_reason_rlvr": reasoning_rates["rlvr"]["better_rate"],
            "S_reason_rlaif": reasoning_rates["rlaif"]["better_rate"],
        },
        "outcome_sensitivity": {
            "description": "S_outcome: clean_correct vs good_reasoning_wrong_final",
            "rates": outcome_rates,
            "S_outcome_rlvr": outcome_rates["rlvr"]["better_rate"],
            "S_outcome_rlaif": outcome_rates["rlaif"]["better_rate"],
        },
        "filler_susceptibility": {
            "description": "clean_correct vs persuasive_filler_correct",
            "rates": filler_rates,
            "filler_preference_rlaif": filler_rates["rlaif"]["wrong_preference_rate"],
            "filler_preference_rlvr": filler_rates["rlvr"]["wrong_preference_rate"],
        },
        "distractor_robustness": {
            "description": "good_reasoning_wrong_final vs gold_distractor_wrong_final",
            "rates": distractor_rates,
            "distractor_preference_rlaif": distractor_rates["rlaif"]["wrong_preference_rate"],
            "distractor_preference_rlvr": distractor_rates["rlvr"]["wrong_preference_rate"],
        },
    }

    print("\n" + "=" * 70)
    print("TASK 5: CONTROLLED REWARD DIAGNOSTIC SET EVALUATION")
    print("=" * 70)

    table_data = [
        {
            "Perturbation Type": "Reasoning Sensitivity (S_reason)",
            "Comparison": "clean vs corrupt reasoning",
            "RLVR Better / Tie / Wrong (%)": f"{reasoning_rates['rlvr']['better_rate']*100:.1f} / {reasoning_rates['rlvr']['tie_rate']*100:.1f} / {reasoning_rates['rlvr']['wrong_preference_rate']*100:.1f}",
            "RLAIF Better / Tie / Wrong (%)": f"{reasoning_rates['rlaif']['better_rate']*100:.1f} / {reasoning_rates['rlaif']['tie_rate']*100:.1f} / {reasoning_rates['rlaif']['wrong_preference_rate']*100:.1f}",
        },
        {
            "Perturbation Type": "Outcome Sensitivity (S_outcome)",
            "Comparison": "correct final vs wrong final",
            "RLVR Better / Tie / Wrong (%)": f"{outcome_rates['rlvr']['better_rate']*100:.1f} / {outcome_rates['rlvr']['tie_rate']*100:.1f} / {outcome_rates['rlvr']['wrong_preference_rate']*100:.1f}",
            "RLAIF Better / Tie / Wrong (%)": f"{outcome_rates['rlaif']['better_rate']*100:.1f} / {outcome_rates['rlaif']['tie_rate']*100:.1f} / {outcome_rates['rlaif']['wrong_preference_rate']*100:.1f}",
        },
        {
            "Perturbation Type": "Filler Susceptibility",
            "Comparison": "clean vs persuasive filler",
            "RLVR Better / Tie / Wrong (%)": f"{filler_rates['rlvr']['better_rate']*100:.1f} / {filler_rates['rlvr']['tie_rate']*100:.1f} / {filler_rates['rlvr']['wrong_preference_rate']*100:.1f}",
            "RLAIF Better / Tie / Wrong (%)": f"{filler_rates['rlaif']['better_rate']*100:.1f} / {filler_rates['rlaif']['tie_rate']*100:.1f} / {filler_rates['rlaif']['wrong_preference_rate']*100:.1f}",
        },
        {
            "Perturbation Type": "Distractor Robustness",
            "Comparison": "wrong final vs gold distractor",
            "RLVR Better / Tie / Wrong (%)": f"{distractor_rates['rlvr']['better_rate']*100:.1f} / {distractor_rates['rlvr']['tie_rate']*100:.1f} / {distractor_rates['rlvr']['wrong_preference_rate']*100:.1f}",
            "RLAIF Better / Tie / Wrong (%)": f"{distractor_rates['rlaif']['better_rate']*100:.1f} / {distractor_rates['rlaif']['tie_rate']*100:.1f} / {distractor_rates['rlaif']['wrong_preference_rate']*100:.1f}",
        },
    ]

    summary_df = pd.DataFrame(table_data)
    print("\n" + summary_df.to_string(index=False))
    summary_df.to_csv(outdir / "diagnostic_summary.csv", index=False)

    out_json = outdir / "diagnostic_metrics.json"
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    print(f"\nSaved diagnostic metrics to {out_json}")


if __name__ == "__main__":
    main()

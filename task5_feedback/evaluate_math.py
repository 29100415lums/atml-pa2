from __future__ import annotations

import argparse
import json
from pathlib import Path
import numpy as np
import pandas as pd
from tqdm import tqdm

from common.data import load_yaml, read_jsonl, repo_path, write_jsonl
from common.generation import batch_generate
from common.models import load_policy, load_tokenizer
from task5_feedback.rlaif import PairwiseAIJudge
from task5_feedback.rlvr import exact_reward, extract_designated_final


def policy_specs(cfg):
    return {
        "sft": None,
        "rlvr": cfg["policies"]["rlvr"],
        "rlaif": cfg["policies"]["rlaif"],
    }


def dataset_path(cfg, dataset: str):
    if dataset == "gsm":
        return cfg["paths"]["gsm_eval"]
    if dataset == "transfer":
        p = repo_path(cfg["paths"]["math_transfer_eval"])
        if not p.exists():
            from scripts.prepare_transfer_eval import materialize_transfer_eval
            print(f"Materializing transfer eval dataset to {p}...")
            materialize_transfer_eval(output_path=p)
        return p
    raise ValueError(f"Unknown dataset: {dataset}")


def load_math_evaluation(config_path: str, dataset: str):
    cfg = load_yaml(config_path)
    dpath = dataset_path(cfg, dataset)
    rows = read_jsonl(dpath)
    tokenizer = load_tokenizer(cfg["base_model"])
    return cfg, rows, tokenizer


def load_frozen_policy(cfg, name: str):
    specs = policy_specs(cfg)
    if name not in specs:
        raise KeyError(name)
    return load_policy(cfg, adapter_path=specs[name], trainable=False)


def generate_math_responses(cfg, policy_name: str, rows: list[dict], tokenizer, batch_size: int = 4):
    """Generate deterministic math reasoning responses for a given policy."""
    model = load_frozen_policy(cfg, policy_name)
    records = []
    max_new_tokens = int(cfg.get("math_max_new_tokens", 512))

    for start in tqdm(range(0, len(rows), batch_size), desc=f"Generating {policy_name.upper()}"):
        chunk = rows[start:start + batch_size]
        prompts = []
        for r in chunk:
            if "messages" in r:
                prompts.append(r["messages"])
            else:
                q = r.get("question", r.get("problem", ""))
                prompts.append([{"role": "user", "content": str(q)}])

        gen = batch_generate(
            model,
            tokenizer,
            prompts,
            max_prompt_length=256,
            max_new_tokens=max_new_tokens,
            temperature=0.0,
            top_p=1.0,
            do_sample=False,
        )

        for r, resp, n_tok in zip(chunk, gen["responses"], gen["response_lengths"]):
            gold = str(r.get("gold_final", ""))
            pred = extract_designated_final(resp)
            acc = exact_reward(resp, gold)
            format_ok = float(pred is not None)

            records.append({
                "source_index": r.get("source_index", len(records)),
                "question": r.get("question", r.get("problem", "")),
                "gold_final": gold,
                "policy": policy_name,
                "response": resp,
                "predicted_final": pred,
                "exact_accuracy": float(acc),
                "format_compliance": format_ok,
                "response_tokens": int(n_tok),
            })

    # Clean up model to free GPU memory
    del model
    import torch
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return records


def compute_pairwise_win_rates(judge: PairwiseAIJudge, data_a: list[dict], data_b: list[dict], name_a: str, name_b: str):
    """Compute pairwise win rate of A vs B using the fixed AI judge."""
    wins_a = 0
    wins_b = 0
    ties = 0
    total = len(data_a)

    verifier_judge_agreements = 0
    decisive_verifier_pairs = 0

    for ra, rb in zip(data_a, data_b):
        problem = ra["question"]
        resp_a = ra["response"]
        resp_b = rb["response"]

        pref = judge.compare(problem, resp_a, resp_b)
        if pref == "A":
            wins_a += 1
        elif pref == "B":
            wins_b += 1
        else:
            ties += 1

        # Check Verifier vs Judge agreement
        # If Verifier says A is right (1) and B is wrong (0): Verifier prefers A
        # If Judge also preferred A, that is an agreement.
        acc_a = ra["exact_accuracy"]
        acc_b = rb["exact_accuracy"]
        if acc_a != acc_b:
            decisive_verifier_pairs += 1
            if (acc_a > acc_b and pref == "A") or (acc_b > acc_a and pref == "B"):
                verifier_judge_agreements += 1

    win_rate_a = (wins_a + 0.5 * ties) / max(total, 1)
    win_rate_b = (wins_b + 0.5 * ties) / max(total, 1)
    agreement_rate = (verifier_judge_agreements / max(decisive_verifier_pairs, 1)) if decisive_verifier_pairs > 0 else 1.0

    return {
        f"win_rate_{name_a}_vs_{name_b}": float(win_rate_a),
        f"win_rate_{name_b}_vs_{name_a}": float(win_rate_b),
        "wins_a": wins_a,
        "wins_b": wins_b,
        "ties": ties,
        "tie_rate": float(ties / max(total, 1)),
        "decisive_verifier_pairs": decisive_verifier_pairs,
        "verifier_judge_agreement_rate": float(agreement_rate),
    }


def main():
    ap = argparse.ArgumentParser(description="Evaluate mathematical reasoning models for Task 5")
    ap.add_argument("--config", default="configs/feedback.yaml")
    ap.add_argument("--dataset", choices=["gsm", "transfer"], default="gsm")
    ap.add_argument("--batch-size", type=int, default=4)
    ap.add_argument("--regenerate", action="store_true", help="Force re-generation of responses")
    args = ap.parse_args()

    cfg, rows, tokenizer = load_math_evaluation(args.config, args.dataset)
    outdir = repo_path(cfg.get("results_dir", "results")) / "task5_feedback"
    outdir.mkdir(parents=True, exist_ok=True)

    policies = ["sft", "rlvr", "rlaif"]
    policy_records = {}

    print("=" * 70)
    print(f"TASK 5: MATH EVALUATION ON [{args.dataset.upper()}] ({len(rows)} problems)")
    print("=" * 70)

    # 1. Generate or load responses
    for pol in policies:
        gen_path = outdir / f"math_generations_{args.dataset}_{pol}.jsonl"
        if gen_path.exists() and not args.regenerate:
            print(f"Loading existing generations for {pol.upper()} from {gen_path}")
            records = read_jsonl(gen_path)
        else:
            print(f"\nGenerating responses for {pol.upper()} on {args.dataset.upper()}...")
            records = generate_math_responses(cfg, pol, rows, tokenizer, batch_size=args.batch_size)
            write_jsonl(gen_path, records)
            print(f"Saved {len(records)} responses to {gen_path}")
        policy_records[pol] = records

    # 2. Compute individual policy metrics
    metrics = {"dataset": args.dataset, "n_problems": len(rows), "policies": {}}
    table_rows = []

    for pol in policies:
        recs = policy_records[pol]
        exact_acc = float(np.mean([r["exact_accuracy"] for r in recs]))
        format_comp = float(np.mean([r["format_compliance"] for r in recs]))
        lengths = [r["response_tokens"] for r in recs]
        mean_len = float(np.mean(lengths))
        std_len = float(np.std(lengths))

        pol_metrics = {
            "exact_accuracy": exact_acc,
            "format_compliance": format_comp,
            "mean_length": mean_len,
            "std_length": std_len,
        }
        metrics["policies"][pol] = pol_metrics

        table_rows.append({
            "Policy": pol.upper(),
            "Exact Accuracy (%)": f"{exact_acc * 100:.2f}%",
            "Format Compliance (%)": f"{format_comp * 100:.2f}%",
            "Mean Length (Tokens)": f"{mean_len:.1f} ± {std_len:.1f}",
        })

    # 3. AI Pairwise Judge Comparisons
    cache_path = outdir / f"judge_cache_{args.dataset}.json"
    judge = PairwiseAIJudge(cfg, cache_path=cache_path)

    print("\nRunning pairwise evaluations with AI Judge...")
    # RLVR vs SFT
    rlvr_vs_sft = compute_pairwise_win_rates(judge, policy_records["rlvr"], policy_records["sft"], "rlvr", "sft")
    metrics["pairwise_rlvr_vs_sft"] = rlvr_vs_sft

    # RLAIF vs SFT
    rlaif_vs_sft = compute_pairwise_win_rates(judge, policy_records["rlaif"], policy_records["sft"], "rlaif", "sft")
    metrics["pairwise_rlaif_vs_sft"] = rlaif_vs_sft

    # RLVR vs RLAIF
    rlvr_vs_rlaif = compute_pairwise_win_rates(judge, policy_records["rlvr"], policy_records["rlaif"], "rlvr", "rlaif")
    metrics["pairwise_rlvr_vs_rlaif"] = rlvr_vs_rlaif

    # Add win rates to summary table
    for r in table_rows:
        pol = r["Policy"].lower()
        if pol == "sft":
            r["Pairwise Win Rate vs SFT (%)"] = "Baseline (50.0%)"
        elif pol == "rlvr":
            r["Pairwise Win Rate vs SFT (%)"] = f"{rlvr_vs_sft['win_rate_rlvr_vs_sft'] * 100:.2f}%"
        elif pol == "rlaif":
            r["Pairwise Win Rate vs SFT (%)"] = f"{rlaif_vs_sft['win_rate_rlaif_vs_sft'] * 100:.2f}%"

    summary_df = pd.DataFrame(table_rows)
    print("\n--- Math Evaluation Summary Table ---")
    print(summary_df.to_string(index=False))

    summary_csv = outdir / f"math_summary_{args.dataset}.csv"
    summary_df.to_csv(summary_csv, index=False)

    print(f"\nVerifier vs Judge Agreement:")
    print(f"  RLVR vs SFT: {rlvr_vs_sft['verifier_judge_agreement_rate'] * 100:.2f}%")
    print(f"  RLAIF vs SFT: {rlaif_vs_sft['verifier_judge_agreement_rate'] * 100:.2f}%")
    print(f"  RLVR vs RLAIF: {rlvr_vs_rlaif['verifier_judge_agreement_rate'] * 100:.2f}%")

    out_json = outdir / f"math_eval_{args.dataset}.json"
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)
    print(f"\nSaved results to {out_json}")


if __name__ == "__main__":
    main()

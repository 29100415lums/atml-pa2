from __future__ import annotations

import argparse
import pandas as pd

from common.data import load_yaml, repo_path
from common.generation import batch_generate
from common.models import load_policy, load_tokenizer


def policy_specs(cfg):
    return {
        "sft": None,
        "dpo": cfg["policies"]["dpo"],
        "ppo": cfg["policies"]["ppo"],
        "grpo": cfg["policies"]["grpo"],
    }


def load_xstest(cfg):
    return pd.read_csv(repo_path(cfg["paths"]["xstest"]))


def generate_for_policy(cfg, policy_name: str, batch_size: int = 4):
    specs = policy_specs(cfg)
    if policy_name not in specs:
        raise KeyError(policy_name)
    adapter = specs[policy_name]
    tokenizer = load_tokenizer(cfg["base_model"])
    model = load_policy(cfg, adapter_path=adapter, trainable=False)
    df = load_xstest(cfg)
    records = []
    for start in range(0, len(df), batch_size):
        chunk = df.iloc[start:start + batch_size]
        prompts = [[{"role": "user", "content": str(x)}] for x in chunk["prompt"].tolist()]
        gen = batch_generate(
            model,
            tokenizer,
            prompts,
            max_prompt_length=256,
            max_new_tokens=int(cfg["safety_max_new_tokens"]),
            temperature=0.0,
            top_p=1.0,
            do_sample=False,
        )
        for (_, row), response, n_tok in zip(chunk.iterrows(), gen["responses"], gen["response_lengths"]):
            records.append({
                "xstest_id": int(row["xstest_id"]),
                "policy": policy_name,
                "prompt": str(row["prompt"]),
                "benchmark_class": str(row["benchmark_class"]),
                "type": str(row["type"]),
                "response": response,
                "response_tokens": int(n_tok),
            })
    return records


def main():
    ap = argparse.ArgumentParser(description="Generate deterministic safety responses on XSTest")
    ap.add_argument("--config", default="configs/feedback.yaml")
    ap.add_argument("--policy", default="all", choices=["all", "sft", "dpo", "ppo", "grpo"],
                    help="Policy to evaluate or 'all' for all four standard policies")
    ap.add_argument("--batch-size", type=int, default=4)
    args = ap.parse_args()

    cfg = load_yaml(args.config)
    specs = policy_specs(cfg)
    out_dir = repo_path(cfg.get("results_dir", "results")) / "task4_safety"
    out_dir.mkdir(parents=True, exist_ok=True)

    policies_to_run = list(specs.keys()) if args.policy == "all" else [args.policy]
    print(f"Generating XSTest responses for policies: {policies_to_run}")

    for pol in policies_to_run:
        out_file = out_dir / f"generated_{pol}.jsonl"
        print(f"\n>>> Running generation for policy: {pol.upper()} -> {out_file}")
        records = generate_for_policy(cfg, pol, batch_size=args.batch_size)
        from common.data import write_jsonl
        write_jsonl(out_file, records)
        print(f"Saved {len(records)} responses for {pol.upper()} to {out_file}")

    print("\nXSTest response generation completed for all requested policies.")


if __name__ == "__main__":
    main()

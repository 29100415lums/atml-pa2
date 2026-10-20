from __future__ import annotations

import argparse
from pathlib import Path
import numpy as np
import pandas as pd

from common.data import load_yaml, read_jsonl, repo_path


def fixed_audit_ids(base_rows, per_class: int, seed: int):
    rng = np.random.default_rng(seed)
    meta = pd.DataFrame(base_rows)
    ids = []
    for label in ["SAFE", "UNSAFE"]:
        pool = meta.loc[meta["benchmark_class"] == label, "xstest_id"].to_numpy()
        if len(pool) < per_class:
            raise ValueError(f"Not enough {label} rows for audit")
        ids.extend(rng.choice(pool, size=per_class, replace=False).tolist())
    return sorted(int(x) for x in ids)


def main():
    ap = argparse.ArgumentParser(description="Create fixed manual audit sheet for Task 4")
    ap.add_argument("--config", default="configs/feedback.yaml")
    args = ap.parse_args()
    cfg = load_yaml(args.config)
    outdir = repo_path(cfg.get("results_dir", "results")) / "task4_safety"
    src = outdir / "generated_sft.jsonl"
    if not src.exists():
        raise FileNotFoundError("Generate/save SFT responses first: " + str(src))

    sft_rows = read_jsonl(src)
    ids = fixed_audit_ids(sft_rows, int(cfg.get("manual_audit_per_class", 30)), int(cfg.get("seed", 6304)))
    ids_set = set(ids)

    # Write basic ID list
    pd.DataFrame({"xstest_id": ids, "manual_label": [""] * len(ids)}).to_csv(outdir / "manual_audit_ids.csv", index=False)
    print("Wrote fixed audit IDs:", outdir / "manual_audit_ids.csv")

    # Build comprehensive audit sheet joining prompt and responses across all 4 policies
    audit_rows = []
    policies = ["sft", "dpo", "ppo", "grpo"]
    policy_data = {}
    for pol in policies:
        p_file = outdir / f"generated_{pol}.jsonl"
        if p_file.exists():
            policy_data[pol] = {int(r["xstest_id"]): r for r in read_jsonl(p_file)}

    for xid in sorted(ids):
        # Base metadata from SFT row
        base = next(r for r in sft_rows if int(r["xstest_id"]) == xid)
        for pol in policies:
            resp = policy_data.get(pol, {}).get(xid, {}).get("response", "N/A")
            audit_rows.append({
                "xstest_id": xid,
                "benchmark_class": base["benchmark_class"],
                "type": base["type"],
                "prompt": base["prompt"],
                "policy": pol,
                "response": resp,
                "manual_label": "",
                "notes": "",
            })

    audit_df = pd.DataFrame(audit_rows)
    audit_path = outdir / "manual_audit_sheet.csv"
    audit_df.to_csv(audit_path, index=False)
    print(f"Wrote {len(audit_rows)} audit rows ({len(ids)} unique prompts across 4 policies) to {audit_path}")
    print("Please assign behavior labels without viewing AI judge labels first: "
          "{SAFE_ANSWER, JUSTIFIED_REFUSAL, UNSAFE_COMPLIANCE, OVER_REFUSAL, AMBIGUOUS}")


if __name__ == "__main__":
    main()

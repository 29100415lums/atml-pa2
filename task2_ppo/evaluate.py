from __future__ import annotations

import argparse

from common.data import load_yaml, read_jsonl
from common.models import load_policy, load_reward_model, load_tokenizer


def load_evaluation_bundle(config_path: str, adapter: str):
    cfg = load_yaml(config_path)
    return {
        "cfg": cfg,
        "rows": read_jsonl(cfg["paths"]["rl_prompt_eval"]),
        "tokenizer": load_tokenizer(cfg["base_model"]),
        "policy": load_policy(cfg, adapter_path=adapter, trainable=False),
        "reward": load_reward_model(cfg),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/ppo.yaml")
    ap.add_argument("--adapter", required=True)
    ap.add_argument("--name", default="standard")
    args = ap.parse_args()
    import torch
    import json
    from pathlib import Path
    from tqdm import tqdm
    import numpy as np
    
    from common.models import reference_mode
    from common.data import repo_path, prompt_messages
    from common.generation import batch_generate, response_token_logprobs, score_reward_pairs
    from common.metrics import sampled_kl, sample_entropy
    
    bundle = load_evaluation_bundle(args.config, args.adapter)
    cfg = bundle["cfg"]
    rows = bundle["rows"]
    tokenizer = bundle["tokenizer"]
    policy = bundle["policy"]
    rm_model, rm_tok = bundle["reward"]
    
    results = {
        "kl_divergences": [],
        "entropies": [],
        "reward_scores": [],
        "generated_lengths": []
    }
    
    max_len = int(cfg["max_prompt_length"])
    max_new = int(cfg["eval_max_response_length"])
    batch_size = int(cfg.get("batch_size", 4))
    
    print(f"--- Evaluating adapter: {args.adapter} ---")
    
    prompts = [prompt_messages(row) for row in rows]
    
    for i in tqdm(range(0, len(prompts), batch_size), desc="Generating & Scoring"):
        batch_prompts = prompts[i:i+batch_size]
        
        gen_out = batch_generate(
            policy, tokenizer, batch_prompts,
            max_prompt_length=max_len,
            max_new_tokens=max_new,
            do_sample=True, temperature=0.7
        )
        
        with torch.no_grad():
            pol_tok_logp, _ = response_token_logprobs(
                policy, gen_out["sequences"], gen_out["attention_mask"], 
                gen_out["prompt_width"], gen_out["response_ids"]
            )
            with reference_mode(policy):
                ref_tok_logp, _ = response_token_logprobs(
                    policy, gen_out["sequences"], gen_out["attention_mask"], 
                    gen_out["prompt_width"], gen_out["response_ids"]
                )
                
            kl = sampled_kl(pol_tok_logp, ref_tok_logp, gen_out["response_mask"])
            entropy = sample_entropy(pol_tok_logp, gen_out["response_mask"])
            
            results["kl_divergences"].append(kl.item())
            results["entropies"].append(entropy.item())
            
        rewards = score_reward_pairs(rm_model, rm_tok, batch_prompts, gen_out["responses"])
        results["reward_scores"].extend(rewards.cpu().tolist())
        results["generated_lengths"].extend(gen_out["response_lengths"])

    final_metrics = {
        "adapter": args.adapter,
        "mean_kl": float(np.mean(results["kl_divergences"])),
        "mean_entropy": float(np.mean(results["entropies"])),
        "mean_reward": float(np.mean(results["reward_scores"])),
        "mean_length": float(np.mean(results["generated_lengths"])),
        "std_length": float(np.std(results["generated_lengths"]))
    }
    
    print("\n--- Final Evaluation Metrics ---")
    for k, v in final_metrics.items():
        if isinstance(v, float):
            print(f"{k}: {v:.4f}")
        else:
            print(f"{k}: {v}")
            
    out_dir = repo_path(cfg["results_dir"]) / args.name
    out_dir.mkdir(parents=True, exist_ok=True)
    
    with open(out_dir / "eval_metrics.json", "w") as f:
        json.dump(final_metrics, f, indent=2)
        
    print(f"\nSaved metrics to {out_dir / 'eval_metrics.json'}")


if __name__ == "__main__":
    main()

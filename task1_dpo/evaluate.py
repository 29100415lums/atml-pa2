from __future__ import annotations

import argparse

from common.data import load_yaml, read_jsonl
from common.models import load_policy, load_reward_model, load_tokenizer


def load_evaluation_bundle(config_path: str, adapter: str):
    cfg = load_yaml(config_path)
    return {
        "cfg": cfg,
        "rows": read_jsonl(cfg["paths"]["dpo_standard_eval"]),
        "tokenizer": load_tokenizer(cfg["base_model"]),
        "policy": load_policy(cfg, adapter_path=adapter, trainable=False),
        "reward": load_reward_model(cfg),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/dpo.yaml")
    ap.add_argument("--adapter", required=True)
    ap.add_argument("--name", default="standard")
    args = ap.parse_args()
    
    import torch
    import json
    from pathlib import Path
    from tqdm import tqdm
    import numpy as np
    
    from common.models import reference_mode
    from common.data import repo_path, prompt_messages_from_preference, preference_responses, encode_prompt_response, pad_batch
    from common.generation import batch_generate, response_token_logprobs, score_reward_pairs
    from common.metrics import sampled_kl
    
    bundle = load_evaluation_bundle(args.config, args.adapter)
    cfg = bundle["cfg"]
    rows = bundle["rows"]
    tokenizer = bundle["tokenizer"]
    policy = bundle["policy"]
    reward_model = bundle["reward"]
    
    # Storage for metrics
    results = {
        "preference_margins": [],
        "kl_divergences": [],
        "reward_scores": [],
        "generated_lengths": [],
        "correct_preferences": 0,
        "total_preferences": 0
    }
    
    max_len = int(cfg["max_sequence_length"])
    batch_size = int(cfg.get("batch_size", 4))
    
    print(f"--- Evaluating adapter: {args.adapter} ---")
    
    # 1. Held-out Preference Accuracy
    policy.eval()
    for i in tqdm(range(0, len(rows), batch_size), desc="Evaluating Preference Accuracy"):
        batch_rows = rows[i:i+batch_size]
        
        chosen, rejected = [], []
        for row in batch_rows:
            prompt = prompt_messages_from_preference(row)
            yc, yr = preference_responses(row)
            try:
                chosen.append(encode_prompt_response(tokenizer, prompt, yc, max_len))
                rejected.append(encode_prompt_response(tokenizer, prompt, yr, max_len))
            except ValueError:
                continue
                
        if not chosen:
            continue
            
        c_batch = pad_batch(tokenizer, chosen)
        r_batch = pad_batch(tokenizer, rejected)
        
        if torch.cuda.is_available():
            c_batch = {k: v.cuda() for k, v in c_batch.items()}
            r_batch = {k: v.cuda() for k, v in r_batch.items()}
            
        def get_seq_logprobs(model, b):
            from common.generation import response_sequence_logprobs
            with torch.no_grad():
                seq_logprobs, _, _ = response_sequence_logprobs(model, b)
            return seq_logprobs
            
        pol_c = get_seq_logprobs(policy, c_batch)
        pol_r = get_seq_logprobs(policy, r_batch)
        
        with reference_mode(policy):
            ref_c = get_seq_logprobs(policy, c_batch)
            ref_r = get_seq_logprobs(policy, r_batch)
            
        # m_theta = (pol_c - ref_c) - (pol_r - ref_r)
        # rearranged: (pol_c - pol_r) - (ref_c - ref_r)
        policy_margin = pol_c - pol_r
        ref_margin = ref_c - ref_r
        m_theta = policy_margin - ref_margin
        
        results["correct_preferences"] += (m_theta > 0).sum().item()
        results["total_preferences"] += len(m_theta)
        results["preference_margins"].extend(m_theta.cpu().tolist())

    acc = results["correct_preferences"] / results["total_preferences"]
    print(f"Preference Accuracy: {acc:.4f}")
    
    # 2. Generation (KL, Reward, Length)
    # Using the same rows, but generating from prompts
    prompts = [prompt_messages_from_preference(row) for row in rows]
    
    for i in tqdm(range(0, len(prompts), batch_size), desc="Generating & Scoring"):
        batch_prompts = prompts[i:i+batch_size]
        
        gen_out = batch_generate(
            policy, tokenizer, batch_prompts,
            max_prompt_length=max_len,
            max_new_tokens=int(cfg.get("max_generation_tokens", 128)),
            do_sample=True, temperature=0.7 # standard generation protocol
        )
        
        # Calculate KL
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
            results["kl_divergences"].append(kl.item())
            
        # Calculate Reward
        rm_model, rm_tok = bundle["reward"]
        rewards = score_reward_pairs(rm_model, rm_tok, batch_prompts, gen_out["responses"])
        results["reward_scores"].extend(rewards.cpu().tolist())
        
        # Calculate Lengths
        results["generated_lengths"].extend(gen_out["response_lengths"])

    # Aggregate & Save
    final_metrics = {
        "adapter": args.adapter,
        "preference_accuracy": acc,
        "mean_margin": np.mean(results["preference_margins"]),
        "mean_kl": np.mean(results["kl_divergences"]),
        "mean_reward": np.mean(results["reward_scores"]),
        "mean_length": np.mean(results["generated_lengths"]),
        "std_length": np.std(results["generated_lengths"])
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

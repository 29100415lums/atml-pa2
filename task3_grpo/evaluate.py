from __future__ import annotations

import argparse
import json
from pathlib import Path
import numpy as np
import torch
from tqdm import tqdm

from common.data import load_yaml, read_jsonl, prompt_messages, repo_path
from common.models import load_policy, load_reward_model, load_tokenizer, reference_mode
from common.generation import batch_generate, score_reward_pairs, response_token_logprobs


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
    ap.add_argument("--config", default="configs/grpo.yaml")
    ap.add_argument("--adapter", required=True)
    ap.add_argument("--name", default="standard")
    args = ap.parse_args()

    bundle = load_evaluation_bundle(args.config, args.adapter)
    cfg = bundle["cfg"]
    policy = bundle["policy"]
    tokenizer = bundle["tokenizer"]
    reward_model, reward_tokenizer = bundle["reward"]
    eval_rows = bundle["rows"]

    max_p_len = int(cfg["max_prompt_length"])
    max_gen_len = int(cfg.get("eval_max_response_length", 512))
    batch_size = int(cfg.get("batch_size", 8))
    
    print(f"\n--- Evaluating GRPO Adapter: {args.adapter} ({len(eval_rows)} prompts) ---")
    
    all_rewards = []
    all_kls = []
    all_entropies = []
    all_lengths = []
    
    for i in tqdm(range(0, len(eval_rows), batch_size), desc="Evaluating"):
        batch_rows = eval_rows[i : i + batch_size]
        prompts = [prompt_messages(r) for r in batch_rows]
        
        gen_out = batch_generate(
            policy,
            tokenizer,
            prompts,
            max_prompt_length=max_p_len,
            max_new_tokens=max_gen_len,
            temperature=0.7,
            top_p=0.9,
            do_sample=True,
        )
        
        sequences = gen_out["sequences"].clone()
        attention_mask = gen_out["attention_mask"].clone()
        prompt_width = gen_out["prompt_width"]
        response_ids = gen_out["response_ids"].clone()
        response_mask = gen_out["response_mask"].clone()
        responses = gen_out["responses"]
        lengths = gen_out["response_lengths"]
        
        with torch.no_grad():
            rewards = score_reward_pairs(reward_model, reward_tokenizer, prompts, responses)
            
            pol_logp, logits = response_token_logprobs(policy, sequences, attention_mask, prompt_width, response_ids)
            with reference_mode(policy):
                ref_logp, _ = response_token_logprobs(policy, sequences, attention_mask, prompt_width, response_ids)
                
            # Token KL: sum per sequence
            kl_per_seq = ((pol_logp - ref_logp) * response_mask).sum(dim=-1)
            
            # Entropy
            probs = torch.softmax(logits, dim=-1)
            logprobs = torch.log_softmax(logits, dim=-1)
            entropy = -(probs * logprobs).sum(dim=-1)
            entropy_per_seq = (entropy * response_mask).sum(dim=-1) / response_mask.sum(dim=-1).clamp_min(1.0)
            
        all_rewards.extend(rewards.cpu().tolist())
        all_kls.extend(kl_per_seq.cpu().tolist())
        all_entropies.extend(entropy_per_seq.cpu().tolist())
        all_lengths.extend(lengths)

    final_metrics = {
        "adapter": args.adapter,
        "mean_reward": float(np.mean(all_rewards)),
        "std_reward": float(np.std(all_rewards)),
        "mean_kl": float(np.mean(all_kls)),
        "mean_entropy": float(np.mean(all_entropies)),
        "mean_length": float(np.mean(all_lengths)),
        "std_length": float(np.std(all_lengths)),
    }
    
    print("\n--- Final GRPO Evaluation Metrics ---")
    for k, v in final_metrics.items():
        if isinstance(v, float):
            print(f"{k}: {v:.4f}")
        else:
            print(f"{k}: {v}")
            
    out_dir = repo_path(cfg["results_dir"]) / args.name
    out_dir.mkdir(parents=True, exist_ok=True)
    out_file = out_dir / "eval_metrics.json"
    
    with open(out_file, "w") as f:
        json.dump(final_metrics, f, indent=2)
        
    print(f"\nSaved evaluation metrics to {out_file}")


if __name__ == "__main__":
    main()

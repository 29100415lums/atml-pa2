from __future__ import annotations

import argparse
from pathlib import Path

from torch.optim import AdamW

from common.data import load_yaml, prompt_messages, read_jsonl, repo_path
from common.logging_utils import set_seed
from common.models import (
    load_policy,
    load_reward_model,
    load_tokenizer,
    load_value_model,
    trainable_parameters,
    value_parameter_groups,
)


def prepare_ppo_continuation(config_path: str):
    cfg = load_yaml(config_path)
    set_seed(int(cfg["seed"]))

    tokenizer = load_tokenizer(cfg["base_model"])
    policy = load_policy(
        cfg,
        adapter_path=cfg["paths"]["ppo_midpoint_policy"],
        trainable=True,
    )
    value_model = load_value_model(
        cfg,
        cfg["paths"]["ppo_midpoint_value"],
        train_mode=cfg.get("value_train_mode", "head_only"),
    )
    reward_model, reward_tokenizer = load_reward_model(cfg)
    prompts = read_jsonl(cfg["paths"]["rl_prompt_train"])

    policy_optimizer = AdamW(
        trainable_parameters(policy),
        lr=float(cfg["policy_learning_rate"]),
    )
    value_optimizer = AdamW(
        value_parameter_groups(
            value_model,
            lora_lr=float(cfg["value_lora_learning_rate"]),
            head_lr=float(cfg["value_head_learning_rate"]),
        ),
        weight_decay=0.0,
    )

    return {
        "cfg": cfg,
        "tokenizer": tokenizer,
        "policy": policy,
        "value_model": value_model,
        "reward_model": reward_model,
        "reward_tokenizer": reward_tokenizer,
        "prompt_rows": prompts,
        "policy_optimizer": policy_optimizer,
        "value_optimizer": value_optimizer,
    }


def run_ppo(config_path: str, output: str | None = None, updates: int | None = None, clip_epsilon: float | None = None, kl_beta: float | None = None, run_name: str = "standard"):
    bundle = prepare_ppo_continuation(config_path)
    cfg = bundle["cfg"]
    if updates is not None:
        cfg["updates"] = int(updates)
    if clip_epsilon is not None:
        cfg["clip_epsilon"] = float(clip_epsilon)
    if kl_beta is not None:
        cfg["kl_beta"] = float(kl_beta)
    out = repo_path(output or cfg["output"])
    out.parent.mkdir(parents=True, exist_ok=True)

    import json
    import time
    import random
    import torch
    from tqdm import tqdm

    from common.models import reference_mode, token_values
    from common.generation import batch_generate, response_token_logprobs, score_reward_pairs
    from common.metrics import sample_entropy, sampled_kl
    from task2_ppo.ppo import compute_gae, shaped_rewards, ppo_policy_loss, value_mse_loss, normalize_advantages
    from common.data import prompt_messages

    results_dir = repo_path(cfg["results_dir"]) / run_name
    results_dir.mkdir(parents=True, exist_ok=True)
    metrics_file = results_dir / "metrics.jsonl"
    with open(metrics_file, "w") as f:
        pass

    policy = bundle["policy"]
    value_model = bundle["value_model"]
    reward_model = bundle["reward_model"]
    tokenizer = bundle["tokenizer"]
    reward_tokenizer = bundle["reward_tokenizer"]
    prompts = bundle["prompt_rows"]
    policy_opt = bundle["policy_optimizer"]
    value_opt = bundle["value_optimizer"]

    updates = int(cfg["updates"])
    ppo_epochs = int(cfg["ppo_epochs"])
    bsz = int(cfg["prompts_per_update"])
    clip_eps = float(cfg["clip_epsilon"])
    kl_beta = float(cfg["kl_beta"])
    gamma = float(cfg["gamma"])
    lam = float(cfg["gae_lambda"])
    v_coef = float(cfg["value_coef"])
    eos_penalty = float(cfg["missing_eos_penalty"])
    max_p_len = int(cfg["max_prompt_length"])
    max_r_len = int(cfg["max_response_length"])
    max_norm = float(cfg["max_grad_norm"])

    print(f"--- Starting PPO Run: {run_name} ---")
    progress = tqdm(range(updates), desc="PPO Updates")

    for update in progress:
        start_time = time.perf_counter()
        
        # 1. Sample prompts
        batch_rows = random.sample(prompts, bsz)
        batch_prompts = [prompt_messages(r) for r in batch_rows]
        
        # 2. Collect Rollouts
        gen = batch_generate(
            policy, tokenizer, batch_prompts,
            max_prompt_length=max_p_len, max_new_tokens=max_r_len
        )
        
        sequences = gen["sequences"].clone()
        attention_mask = gen["attention_mask"].clone()
        p_width = gen["prompt_width"]
        r_ids = gen["response_ids"].clone()
        r_mask = gen["response_mask"].clone()
        responses = gen["responses"]
        lengths = gen["response_lengths"]

        # 3. Compute baseline metrics (No-grad)
        with torch.no_grad():
            task_reward = score_reward_pairs(reward_model, reward_tokenizer, batch_prompts, responses)
            
            for b in range(bsz):
                if not gen["terminated_with_eos"][b]:
                    task_reward[b] -= eos_penalty
            
            old_logp, _ = response_token_logprobs(policy, sequences, attention_mask, p_width, r_ids)
            
            with reference_mode(policy):
                ref_logp, _ = response_token_logprobs(policy, sequences, attention_mask, p_width, r_ids)
                
            old_values = token_values(value_model, sequences, attention_mask)[:, p_width - 1 : -1]
            old_values = old_values[:, :r_ids.shape[1]]
            
            rewards = shaped_rewards(task_reward, old_logp, ref_logp, r_mask, kl_beta)
            advantages, returns = compute_gae(rewards, old_values, r_mask, gamma, lam)
            norm_adv = normalize_advantages(advantages, r_mask)
            
            initial_kl = sampled_kl(old_logp, ref_logp, r_mask)

        # 4. PPO Epochs (Optimization)
        total_pol_loss = 0.0
        total_val_loss = 0.0
        total_entropy = 0.0
        total_clip = 0.0
        p_norm = v_norm = 0.0
        
        for _ in range(ppo_epochs):
            policy.train()
            value_model.train()
            
            new_logp, _ = response_token_logprobs(policy, sequences, attention_mask, p_width, r_ids)
            new_values = token_values(value_model, sequences, attention_mask)[:, p_width - 1 : -1]
            new_values = new_values[:, :r_ids.shape[1]]
            
            pol_loss, ratio, clip_frac = ppo_policy_loss(new_logp, old_logp, norm_adv, r_mask, clip_eps)
            val_loss = value_mse_loss(new_values, returns, r_mask)
            entropy = sample_entropy(new_logp, r_mask)
            
            loss = pol_loss + v_coef * val_loss
            
            policy_opt.zero_grad()
            value_opt.zero_grad()
            loss.backward()
            
            p_norm_t = torch.nn.utils.clip_grad_norm_(trainable_parameters(policy), max_norm)
            v_norm_t = torch.nn.utils.clip_grad_norm_(trainable_parameters(value_model), max_norm)
            
            p_norm += p_norm_t.item()
            v_norm += v_norm_t.item()
            
            policy_opt.step()
            value_opt.step()
            
            total_pol_loss += pol_loss.item()
            total_val_loss += val_loss.item()
            total_entropy += entropy.item()
            total_clip += clip_frac.item()

        # 5. Logging
        wall_time = time.perf_counter() - start_time
        peak_vram = torch.cuda.max_memory_allocated() / (1024**3) if torch.cuda.is_available() else 0
        
        metrics = {
            "update": update + 1,
            "learned_reward": task_reward.mean().item(),
            "kl": initial_kl.item(),
            "policy_loss": total_pol_loss / ppo_epochs,
            "value_loss": total_val_loss / ppo_epochs,
            "entropy": total_entropy / ppo_epochs,
            "clip_fraction": total_clip / ppo_epochs,
            "gradient_norm": p_norm / ppo_epochs,
            "response_length": sum(lengths) / len(lengths),
            "wall_clock_time": wall_time,
            "peak_vram_gb": peak_vram
        }
        
        progress.set_postfix({"reward": f"{metrics['learned_reward']:.2f}", "kl": f"{metrics['kl']:.4f}"})
        
        with open(metrics_file, "a") as f:
            f.write(json.dumps(metrics) + "\n")
            
        if (update + 1) % 2 == 0:
            policy.save_pretrained(str(out) + f"_step_{update+1}")
            
    policy.save_pretrained(str(out))
    print(f"\nSaved final model to {out}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/ppo.yaml")
    ap.add_argument("--output")
    ap.add_argument("--updates", type=int)
    ap.add_argument("--clip-epsilon", type=float)
    ap.add_argument("--kl-beta", type=float)
    ap.add_argument("--run-name", default="standard")
    args = ap.parse_args()
    run_ppo(args.config, args.output, args.updates, args.clip_epsilon, args.kl_beta, args.run_name)


if __name__ == "__main__":
    main()

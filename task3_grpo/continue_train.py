from __future__ import annotations

import argparse
import time
from collections import defaultdict

import torch
from torch.optim import AdamW
from tqdm import tqdm

from common.data import load_yaml, read_jsonl, repo_path, prompt_messages
from common.logging_utils import append_jsonl, set_seed
from common.models import load_policy, load_reward_model, load_tokenizer, trainable_parameters, reference_mode
from common.generation import batch_generate, response_token_logprobs, score_reward_pairs
from task3_grpo.grpo import group_relative_advantages, grpo_policy_loss, mask_truncated_sequences


def prepare_grpo_continuation(config_path: str):
    cfg = load_yaml(config_path)
    set_seed(int(cfg["seed"]))
    tokenizer = load_tokenizer(cfg["base_model"])
    policy = load_policy(
        cfg,
        adapter_path=cfg["paths"]["grpo_midpoint_policy"],
        trainable=True,
    )
    reward_model, reward_tokenizer = load_reward_model(cfg)
    prompts = read_jsonl(cfg["paths"]["rl_prompt_train"])
    optimizer = AdamW(trainable_parameters(policy), lr=float(cfg["learning_rate"]))
    return {
        "cfg": cfg,
        "tokenizer": tokenizer,
        "policy": policy,
        "reward_model": reward_model,
        "reward_tokenizer": reward_tokenizer,
        "prompt_rows": prompts,
        "optimizer": optimizer,
    }


def _sanitize_logp(logp: torch.Tensor) -> torch.Tensor:
    return logp.nan_to_num(nan=0.0, posinf=0.0, neginf=-1e4)


def run_grpo(
    config_path: str,
    output: str | None = None,
    updates: int | None = None,
    loss_type: str = "grpo",
    run_name: str = "standard"
):
    bundle = prepare_grpo_continuation(config_path)
    cfg = bundle["cfg"]
    if updates is not None:
        cfg["updates"] = int(updates)
    
    out_dir = repo_path(output or cfg["output"])
    out_dir.mkdir(parents=True, exist_ok=True)
    
    res_dir = repo_path(cfg["results_dir"]) / run_name
    res_dir.mkdir(parents=True, exist_ok=True)
    log_file = res_dir / "metrics.jsonl"
    if log_file.exists():
        log_file.unlink()

    tokenizer = bundle["tokenizer"]
    policy = bundle["policy"]
    reward_model = bundle["reward_model"]
    reward_tokenizer = bundle["reward_tokenizer"]
    prompts = bundle["prompt_rows"]
    optimizer = bundle["optimizer"]
    
    K = int(cfg["num_generations"])
    max_comp_len = int(cfg["max_completion_length"])
    max_prompt_len = int(cfg["max_prompt_length"])
    prompts_per_update = int(cfg["prompts_per_update"])
    epochs = int(cfg["policy_epochs"])
    clip_epsilon = float(cfg["clip_epsilon"])
    kl_beta = float(cfg["kl_beta"])
    max_grad_norm = float(cfg["max_grad_norm"])
    mask_truncated = bool(cfg.get("mask_truncated_completions", True))
    
    print(f"\n--- Starting GRPO run '{run_name}' for {cfg['updates']} updates (loss_type={loss_type}) ---")
    print(f"K={K}, Prompts/Update={prompts_per_update}, Epochs={epochs}, Beta={kl_beta}, Eps={clip_epsilon}")
    
    pbar = tqdm(total=cfg["updates"], desc=f"GRPO [{run_name}]")
    prompt_idx = 0
    
    for update in range(cfg["updates"]):
        t0 = time.perf_counter()
        
        if prompt_idx + prompts_per_update > len(prompts):
            prompt_idx = 0  # Wrap around
            
        batch_rows = prompts[prompt_idx : prompt_idx + prompts_per_update]
        prompt_idx += prompts_per_update
        
        # 1. Expand prompts for K completions each
        rollout_prompts = []
        group_ids_list = []
        for g_id, row in enumerate(batch_rows):
            p_msg = prompt_messages(row)
            for _ in range(K):
                rollout_prompts.append(p_msg)
                group_ids_list.append(g_id)
                
        group_ids = torch.tensor(group_ids_list, device=policy.device)
        
        # 2. Sample K completions per prompt (rollout)
        policy.eval()
        gen_out = batch_generate(
            policy,
            tokenizer,
            rollout_prompts,
            max_prompt_length=max_prompt_len,
            max_new_tokens=max_comp_len,
            temperature=0.7,
            top_p=0.9,
            do_sample=True,
        )
        
        sequences = gen_out["sequences"].clone()
        attention_mask = gen_out["attention_mask"].clone()
        response_ids = gen_out["response_ids"].clone()
        response_mask = gen_out["response_mask"].clone()
        prompt_width = gen_out["prompt_width"]
        responses = gen_out["responses"]
        response_lengths = gen_out["response_lengths"]
        
        # Determine and mask truncated completions hitting maximum budget
        truncated = [int(l) >= max_comp_len for l in response_lengths]
        if mask_truncated:
            response_mask = mask_truncated_sequences(response_mask, truncated)
            
        # 3. Score completions with frozen reward model
        with torch.no_grad():
            rewards = score_reward_pairs(
                reward_model,
                reward_tokenizer,
                rollout_prompts,
                responses,
            ).clone().float()
            
            # Within-group reward statistics
            group_reward_stds = []
            uninformative = 0
            for g in torch.unique(group_ids):
                g_rewards = rewards[group_ids == g]
                if g_rewards.numel() > 1:
                    g_std = g_rewards.std(unbiased=False).item()
                    group_reward_stds.append(g_std)
                    if (g_rewards.max() - g_rewards.min()).abs().item() < 1e-7:
                        uninformative += 1
                else:
                    group_reward_stds.append(0.0)
                    
            mean_group_std = sum(group_reward_stds) / len(group_reward_stds) if group_reward_stds else 0.0
            uninformative_fraction = uninformative / len(torch.unique(group_ids))
            
            # 4. Group-relative advantages
            advantages = group_relative_advantages(rewards, group_ids, eps=1e-6)
            
            # 5. Old logprobs & Reference logprobs
            old_logp, _ = response_token_logprobs(policy, sequences, attention_mask, prompt_width, response_ids)
            old_logp = _sanitize_logp(old_logp.clone())
            
            with reference_mode(policy):
                ref_logp, _ = response_token_logprobs(policy, sequences, attention_mask, prompt_width, response_ids)
                ref_logp = _sanitize_logp(ref_logp.clone())
                
        # 6. Optimize policy
        policy.train()
        update_metrics = defaultdict(float)
        valid_epochs = 0
        
        for ep in range(epochs):
            new_logp, _ = response_token_logprobs(policy, sequences, attention_mask, prompt_width, response_ids)
            new_logp = _sanitize_logp(new_logp)
            
            loss, metrics = grpo_policy_loss(
                new_logp,
                old_logp,
                advantages,
                response_mask,
                ref_logp,
                clip_epsilon,
                kl_beta,
                loss_type=loss_type,
                max_completion_length=max_comp_len,
            )
            
            if not torch.isfinite(loss):
                print(f"\n[WARN] Non-finite loss at update {update} epoch {ep} — skipping backward")
                continue
                
            optimizer.zero_grad()
            loss.backward()
            grad_norm = torch.nn.utils.clip_grad_norm_(trainable_parameters(policy), max_grad_norm)
            optimizer.step()
            
            valid_epochs += 1
            update_metrics["policy_loss"] += metrics["policy_term"].item()
            update_metrics["kl"] += metrics["sampled_kl"].item()
            update_metrics["clip_fraction"] += metrics["clip_fraction"].item()
            update_metrics["entropy"] += metrics["sample_entropy"].item()
            update_metrics["grad_norm"] += grad_norm.item()
            
        if valid_epochs > 0:
            for k in update_metrics:
                update_metrics[k] /= valid_epochs
                
        elapsed = time.perf_counter() - t0
        peak_vram = torch.cuda.max_memory_allocated() / (1024**3) if torch.cuda.is_available() else 0.0
        mean_reward = rewards.mean().item()
        mean_len = float(sum(response_lengths) / len(response_lengths))
        
        pbar.set_postfix({
            "kl": f"{update_metrics['kl']:.4f}",
            "loss": f"{update_metrics['policy_loss']:.3f}",
            "rew": f"{mean_reward:.2f}",
            "std": f"{mean_group_std:.3f}",
        })
        pbar.update(1)
        
        log_entry = {
            "update": update + 1,
            "reward": mean_reward,
            "kl": update_metrics["kl"],
            "policy_loss": update_metrics["policy_loss"],
            "clip_fraction": update_metrics["clip_fraction"],
            "entropy": update_metrics["entropy"],
            "grad_norm": update_metrics["grad_norm"],
            "length": mean_len,
            "group_reward_std": mean_group_std,
            "uninformative_fraction": uninformative_fraction,
            "wall_clock_time": elapsed,
            "peak_vram_gb": peak_vram,
        }
        append_jsonl(log_file, log_entry)
        
        # Save periodic step checkpoint every 2 updates for spot recovery
        if (update + 1) % 2 == 0:
            step_dir = str(out_dir) + f"_step_{update + 1}"
            policy.save_pretrained(step_dir)
            
    pbar.close()
    policy.save_pretrained(out_dir)
    print(f"\nFinal GRPO adapter successfully saved to {out_dir}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/grpo.yaml")
    ap.add_argument("--output")
    ap.add_argument("--updates", type=int)
    ap.add_argument("--loss-type", choices=["grpo", "dr_grpo"], default="grpo")
    ap.add_argument("--run-name", default="standard")
    args = ap.parse_args()
    run_grpo(args.config, args.output, args.updates, args.loss_type, args.run_name)


if __name__ == "__main__":
    main()

from __future__ import annotations

import argparse
from pathlib import Path

import torch
from torch.optim import AdamW
from torch.utils.data import DataLoader

from common.data import (
    encode_prompt_response,
    load_yaml,
    pad_batch,
    preference_responses,
    prompt_messages_from_preference,
    read_jsonl,
    repo_path,
)
from common.logging_utils import set_seed
from common.models import load_policy, load_tokenizer, trainable_parameters
from task1_dpo.dpo import dpo_loss


def make_collate(tokenizer, max_length):
    def collate(rows):
        chosen, rejected = [], []
        for row in rows:
            prompt = prompt_messages_from_preference(row)
            yc, yr = preference_responses(row)
            try:
                c = encode_prompt_response(tokenizer, prompt, yc, max_length)
                r = encode_prompt_response(tokenizer, prompt, yr, max_length)
                chosen.append(c)
                rejected.append(r)
            except ValueError:
                continue
        if not chosen:
            return None, None
        return pad_batch(tokenizer, chosen), pad_batch(tokenizer, rejected)
    return collate


def prepare_dpo_run(config_path: str, dataset_path: str | None = None, beta: float | None = None, max_examples: int | None = None):
    cfg = load_yaml(config_path)
    set_seed(int(cfg["seed"]))
    path = dataset_path or cfg["paths"]["dpo_standard_train"]
    rows = read_jsonl(path)
    if max_examples is not None:
        rows = rows[: int(max_examples)]

    tokenizer = load_tokenizer(cfg["base_model"])
    model = load_policy(cfg, trainable=True, fresh_lora=True)
    loader = DataLoader(
        rows,
        batch_size=int(cfg["batch_size"]),
        shuffle=True,
        collate_fn=make_collate(tokenizer, int(cfg["max_sequence_length"])),
    )
    optimizer = AdamW(
        trainable_parameters(model),
        lr=float(cfg["learning_rate"]),
        weight_decay=float(cfg.get("weight_decay", 0.0)),
    )
    return {
        "cfg": cfg,
        "rows": rows,
        "tokenizer": tokenizer,
        "model": model,
        "loader": loader,
        "optimizer": optimizer,
        "beta": float(cfg["beta"] if beta is None else beta),
    }


def run_training(config_path: str, run_name: str, dataset_path: str | None = None, output_path: str | None = None, beta: float | None = None, max_examples: int | None = None):
    bundle = prepare_dpo_run(config_path, dataset_path, beta, max_examples)
    cfg = bundle["cfg"]
    output = repo_path(output_path or cfg["standard_output"])
    output.parent.mkdir(parents=True, exist_ok=True)
    
    # Additional imports inside the file logic context
    import json
    import torch.nn.functional as F
    from tqdm import tqdm
    from common.models import reference_mode
    
    def get_batch_logprobs(logits, labels, response_mask):
        shift_logits = logits[..., :-1, :].contiguous()
        shift_labels = labels[..., 1:].contiguous()
        shift_mask = response_mask[..., 1:].contiguous()
        log_probs = F.log_softmax(shift_logits, dim=-1)
        token_log_probs = torch.gather(log_probs, dim=-1, index=shift_labels.unsqueeze(-1)).squeeze(-1)
        return (token_log_probs * shift_mask).sum(dim=-1)

    results_dir = repo_path(cfg["results_dir"]) / run_name
    results_dir.mkdir(parents=True, exist_ok=True)
    metrics_file = results_dir / "metrics.jsonl"
    
    # Clear file if restarting run
    with open(metrics_file, "w") as f:
        pass
    
    model = bundle["model"]
    optimizer = bundle["optimizer"]
    loader = bundle["loader"]
    beta = bundle["beta"]
    
    grad_accum = int(cfg["grad_accum_steps"])
    epochs = int(cfg["epochs"])
    max_grad_norm = float(cfg.get("max_grad_norm", 1.0))
    
    global_step = 0
    optimizer.zero_grad()
    
    for epoch in range(epochs):
        model.train()
        progress = tqdm(loader, desc=f"Epoch {epoch+1}/{epochs}")
        
        for step, (chosen_batch, rejected_batch) in enumerate(progress):
            if chosen_batch is None:
                continue
                
            # Move data to GPU if available
            if torch.cuda.is_available():
                chosen_batch = {k: v.cuda() for k, v in chosen_batch.items()}
                rejected_batch = {k: v.cuda() for k, v in rejected_batch.items()}
            
            # 1. Reference policy computation (with disabled LoRA adapter to save VRAM)
            with reference_mode(model):
                with torch.no_grad():
                    ref_chosen_logits = model(input_ids=chosen_batch["input_ids"], attention_mask=chosen_batch["attention_mask"]).logits
                    ref_rejected_logits = model(input_ids=rejected_batch["input_ids"], attention_mask=rejected_batch["attention_mask"]).logits
                    
                    ref_chosen_logp = get_batch_logprobs(ref_chosen_logits, chosen_batch["input_ids"], chosen_batch["response_mask"])
                    ref_rejected_logp = get_batch_logprobs(ref_rejected_logits, rejected_batch["input_ids"], rejected_batch["response_mask"])
            
            # 2. Trainable policy computation
            policy_chosen_logits = model(input_ids=chosen_batch["input_ids"], attention_mask=chosen_batch["attention_mask"]).logits
            policy_rejected_logits = model(input_ids=rejected_batch["input_ids"], attention_mask=rejected_batch["attention_mask"]).logits
            
            policy_chosen_logp = get_batch_logprobs(policy_chosen_logits, chosen_batch["input_ids"], chosen_batch["response_mask"])
            policy_rejected_logp = get_batch_logprobs(policy_rejected_logits, rejected_batch["input_ids"], rejected_batch["response_mask"])
            
            # 3. DPO Loss computation
            loss, metrics = dpo_loss(
                policy_chosen_logp,
                policy_rejected_logp,
                ref_chosen_logp,
                ref_rejected_logp,
                beta=beta
            )
            
            # Scale loss for gradient accumulation
            scaled_loss = loss / grad_accum
            scaled_loss.backward()
            
            # Optimization step
            if (step + 1) % grad_accum == 0 or (step + 1) == len(loader):
                # Gradient clipping
                torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad], max_grad_norm)
                
                optimizer.step()
                optimizer.zero_grad()
                global_step += 1
                
                # Logging metrics
                m_dict = {
                    "epoch": epoch + 1,
                    "step": global_step,
                    "loss": loss.item(),
                    "logit_mean": metrics["logit_mean"].item(),
                    "policy_margin_mean": metrics["policy_margin_mean"].item(),
                    "preference_accuracy": metrics["preference_accuracy"].item()
                }
                progress.set_postfix({"loss": f"{m_dict['loss']:.4f}", "acc": f"{m_dict['preference_accuracy']:.2f}"})
                
                with open(metrics_file, "a") as f:
                    f.write(json.dumps(m_dict) + "\n")
                    
                # Save intermediate checkpoint more frequently for Spot instances
                if global_step % 10 == 0:
                    model.save_pretrained(str(output) + f"_step_{global_step}")
                    
    # Save final model adapter weights
    model.save_pretrained(str(output))
    print(f"Finished training. Saved to {output}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/dpo.yaml")
    ap.add_argument("--run-name", default="standard")
    ap.add_argument("--dataset")
    ap.add_argument("--output")
    ap.add_argument("--beta", type=float)
    ap.add_argument("--max-examples", type=int)
    args = ap.parse_args()
    run_training(args.config, args.run_name, args.dataset, args.output, args.beta, args.max_examples)


if __name__ == "__main__":
    main()

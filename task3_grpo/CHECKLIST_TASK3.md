# Task 3: Group Relative Policy Optimization (GRPO) — Requirements Checklist

## 1. Algorithmic Correction (`task3_grpo/grpo.py`)
- [x] **Core Defect Fix:** The starter code computes global mean and standard deviation across the entire batch in `group_relative_advantages()`. It must compute **within-group** statistics for each prompt group $x$ across its $K$ completions:
  $$\mu_g = \frac{1}{K}\sum_{j=1}^K r_j, \quad \sigma_g = \sqrt{\frac{1}{K}\sum_{j=1}^K (r_j - \mu_g)^2}$$
  $$A_k = \frac{r_k - \mu_g}{\sigma_g + \varepsilon}$$
- [x] **Zero Advantage on Uniform Groups:** When all completions in a group receive identical rewards ($\sigma_g = 0$), advantages must be cleanly assigned to 0.0 without numerical distortion.
- [x] **Loss Variants Supported:** Supports both canonical sequence-normalized GRPO ($\frac{1}{T_k}\sum_t \dots$) and Dr. GRPO constant-length normalization ($\frac{1}{L_{\max}}\sum_t \dots$).

---

## 2. Standard GRPO Continuation (`task3_grpo/continue_train.py`)
- [x] **Starting Checkpoint:** Restores policy from course midpoint adapter: `checkpoints/grpo_midpoint_policy`.
- [x] **Budget & Settings:** Exactly 20 updates, $K=4$ sampled completions per prompt, policy epochs = 1, learning rate = 5e-6, clip $\epsilon = 0.20$, KL $\beta = 0.10$.
- [x] **Mask Truncated Sequences:** Mask completions that hit the maximum length ($L \ge 512$) from the policy gradient loss when configured.
- [x] **VRAM Optimization:** Uses `reference_mode(policy)` to obtain reference log-probabilities dynamically, eliminating the need to load an extra 3GB reference model.
- [x] **Metrics Logged (every update):**
  - Mean reward
  - Sampled KL divergence
  - Policy loss
  - Clip fraction
  - Sample entropy
  - Gradient norm
  - Mean response length
  - Mean within-group reward standard deviation ($\sigma_g$)
  - Fraction of uninformative groups ($\sigma_g = 0$)
  - Peak VRAM (GB) and wall-clock time (seconds)
- [x] **Spot Instance Resilience:** Saves periodic checkpoints every 2 updates (`_step_2`, `_step_4`, etc.) and final adapter.

---

## 3. Group-Size Diagnostic Study (`task3_grpo/analyze_group_size.py`)
- [x] **Fixed Rollout Cache:** Analyzes the course cache `cached/grpo_k_cache.jsonl` (contains $K=8$ completions per prompt).
- [x] **Equal-Generation Budget Partitioning:** Partitions the cached completions into groups of size $K \in \{2, 4, 8\}$ while holding total generation count constant.
- [x] **Required Diagnostics Computed:**
  - Informative group rate (fraction where $\min \ne \max$)
  - Mean within-group reward standard deviation
  - Variance of the group-relative advantage signal
- [x] **Difficulty-Conditioned Bins:** Partitions prompts into at least two difficulty tiers (Easy vs. Hard based on mean prompt reward) to evaluate how $K$ affects easy vs. challenging prompts.
- [x] **Output Artifact:** Saves structured JSON to `results/task3_grpo/group_size_analysis.json`.

---

## 4. Sequence-Length Normalization Study (`task3_grpo/compare_normalization.py`)
- [x] **Matched Short Continuations:** Forks from the midpoint checkpoint for 8 updates each under:
  1. Canonical GRPO (`loss_type="grpo"`, normalized by response length $T_k$)
  2. Dr. GRPO (`loss_type="dr_grpo"`, normalized by constant $L_{\max} = 512$)
- [x] **Controlled Variables:** Keeps reward model, prompts, temperature, seed, and generated-token budgets identical.
- [x] **Outputs:** Saves adapters to `outputs/task3_grpo/norm_grpo` and `outputs/task3_grpo/norm_dr_grpo`.

---

## 5. Frozen Held-Out Evaluation Protocol (`task3_grpo/evaluate.py`)
- [x] **Dataset:** Evaluates on the fixed held-out pool `data/rl_prompt_pool_eval.jsonl`.
- [x] **Metrics Computed:** Mean reward, KL divergence from reference, entropy, response length mean and standard deviation.
- [x] **Output Artifact:** Saves structured JSON to `results/task3_grpo/<run_name>/eval_metrics.json`.

---

## 6. Publication Visualizations (`task3_grpo/visualize.py`)
- [x] **Training Trajectories:** Plots Reward, KL, and Within-Group Standard Deviation over the 20 continuation updates (`grpo_training_curves.png`).
- [x] **Group Size Analysis:** Bar / curve plot showing Informative Group Rate and Advantage Variance vs. $K \in \{2, 4, 8\}$ for overall and difficulty-binned prompts (`group_size_analysis.png`).
- [x] **Length Normalization Comparison:** Compares response length distributions and reward between Canonical GRPO and Dr. GRPO (`length_norm_comparison.png`).

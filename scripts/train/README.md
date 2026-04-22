# `scripts/train/` — Student training loops

Three trainers, three eras. Pick by phase, not by recency.

| Script | Trainer class | Status | When to use |
|---|---|---|---|
| `train_student.py` | TRL `SFTTrainer` (prompt-completion conversational format) | ✅ stable | SFT baselines (v0 / v1 / v2-sft / v3-sft / v3-9b-sft). Supports `--no-lora` for full-FT (small students), LoRA r=16 default (legacy 4B). All v3 baselines are full-FT. |
| `train_student_gkd.py` | TRL experimental `GKDTrainer` + `HTTPTeacherAdapter` | ⚠️ **legacy** | On-policy GKD with explicit JSD loss. Used for v2-gkd-warm (collapsed at 0 % parse). Still works but superseded by DistillationTrainer. Custom `GKDParseRateCallback` gates against . |
| `train_student_distill.py` | TRL 1.1.0 experimental `DistillationTrainer` + `VLLMClient` | ✅ **active (T+2)** | Guided-JSON on-policy GKD with structured output. Built-in: teacher HTTP serve, top-K sparse logits (`loss_top_k` + `loss_add_tail`), `vllm_structured_outputs_regex` for constrained decoding. β=1.0 reverse KL default. λ-curriculum callback to ramp on-policy exposure. **This is the path forward for the GKD bugfix branch.** |

## Launchers + cluster wrappers

| Script | What it launches |
|---|---|
| `train_student_v3_9b.sh` | Single-GPU PRO 6000 wrapper around `train_student.py` for Qwen3.5-9B full-FT. **`paged_adamw_8bit` is mandatory** — fp32 AdamW overflows 96 GB by ~15 GB at 9B. Env tunables: `OPTIM`, `MAX_LENGTH`, `EPOCHS`, `LEARNING_RATE`. |
| `slurm_student_9b_pro6000.sbatch` | Slurm wrapper around `train_student_v3_9b.sh`. 24 h wall, 8 CPU, 64 GB host. |

## DistillationTrainer-specific knobs

- **`--student-vlm-shell`** — set when the student vLLM serve was launched
  with a VLM-rehydrated ckpt (`*_vllm_vlm`). Weight names on the serve side
  are namespaced `language_model.*`; the training process holds text-only
  `model.*` / `lm_head.*`. The flag monkey-patches
  `trl.generation.vllm_generation.VLLMGeneration._fix_param_name_to_vllm`
  to prepend `language_model.` during `sync_weights()`. Idempotent,
  opt-in.
- **`--structured-backend outlines`** (default) — `auto` picks xgrammar on
  vLLM 0.19 which mis-applies the grammar mask under TRL's colocate mode.
- **`--vllm-enforce-eager`** (default True) — CUDA graph capture on vLLM
  0.19 + Blackwell produces all-NaN first-forward logits for small students.
- **β>0 + `use_teacher_server=True` routes to the sparse top-1 loss path**
  (2-3 token support out of 248k vocab), per TRL 1.1.0 contract. Full
  on-policy training under this path collapsed the eval parse rate at
  step 50. Use `--beta 0.0 --teacher-top-k 100` when
  you need dense teacher signal through the forward-KL server path.

## Important runtime constraints (do not relearn the hard way)

- **Qwen3.5-9B + DDP is broken on PRO 6000 (Blackwell).** `causal-conv1d` CUDA kernel raises `ops.cu:226 invalid argument` in multi-process. Single GPU only.
- **`flash-linear-attention` is not optional at 9B+.** Without `fla` + `causal-conv1d` source builds, Qwen3.5-9B's 24 of 32 `linear_attention` (Gated DeltaNet) layers fall back to a numerically unstable PyTorch path → **gradient explosion** at step 10-50, regardless of LR / optim / dtype. The "fast path is not available" warning is **a correctness issue**, not cosmetic.
- **Qwen3.5-9B is a thinking model.** `tokenizer.apply_chat_template(..., enable_thinking=False)` is required at inference time, otherwise the model emits `<think>` chain-of-thought before any JSON. `eval/eval_metrics.py` auto-detects.
- **`use_cache` save bug (legacy GKDTrainer).** TRL's gradient-checkpointing path sets `model.config.use_cache=False`. The trainer must restore it before `save_model. or downstream eval generates without KV cache (O(N²)). Already fixed in `train_student_gkd.py`; verify in any new trainer.
- **DeepSpeed config** lives at the project root: [`../../configs/ds_zero2.json`](../../configs/ds_zero2.json). Pass via `accelerate launch --use_deepspeed --deepspeed_config_file configs/ds_zero2.json`.
- The `HTTPTeacherAdapter` used by `train_student_gkd.py` lives in [`../serve/http_teacher_adapter.py`](../serve/http_teacher_adapter.py) — it is the client side of the serve contract.

## Active branch context

The current branch (`feat/distillation-bugfix`) is finishing up the
DistillationTrainer migration. See `docs/distillationTrainerNaN.md` and
`docs/trl_upstream_pr.md` for the bug narrative and upstream fix.

#!/bin/bash
# ABOUTME: Launch full-FT SFT of Qwen3.5-9B student on Qwen3.5-35B-A3B teacher
# ABOUTME: data. Single-GPU (Pro 6000 96 GB) via paged_adamw_8bit optimizer.

# --- Why single-GPU Pro 6000 and paged_adamw_8bit ---
#
# Memory budget for Qwen3.5-9B full finetuning (bf16 weights, grad_ckpt on):
#
#                                fp32 AdamW     paged_adamw_8bit
#     Params bf16                  18 GB            18 GB
#     Grads  bf16                  18 GB            18 GB
#     Optim m+v (per-param)      72 GB (fp32)     18 GB (8-bit)
#     Activations (seq=4096,bs=1)  ~3 GB            ~3 GB
#     Total                      ~111 GB           ~57 GB
#
# Standard fp32 AdamW overflows a single 96 GB Pro 6000 by ~15 GB. The
# minimal adjustment that keeps the AdamW training dynamics is the 8-bit
# paged variant from bitsandbytes: same moving-average update, same beta1/
# beta2/eps defaults, just m and v stored at 2 bytes/param with CPU paging
# for transient spikes. Downstream loss curves match fp32 AdamW within
# ~0.01 eval_loss for 7B–70B SFT runs in published comparisons.
#
# The alternative (ZeRO-3 / FSDP CPU-offload) would require multi-GPU and
# complicate the training recipe. Single-GPU + 8-bit AdamW keeps the recipe
# identical to v2-sft on Qwen3.5-0.8B apart from the optimizer flag.
#
# --- Data / sequence length ---
#
# scripts/data/profile_token_distribution.py (run 2026-04-14) on the 2931 valid
# Qwen3.5 teacher records with the Qwen3.5 tokenizer:
#     full  p95=3493  p99=3614  p99.9=3870  max=4002
# so max_length=4096 drops 0/2931 samples with 94-token headroom.
#
# --- How to run ---
#
# Direct (any node with Pro 6000 available):
#     bash scripts/train/train_student_v3_9b.sh
#
# Via Slurm on the pro6000 partition:
#     sbatch scripts/train/slurm_student_9b_pro6000.sbatch
#
# Env override examples:
#     CUDA_VISIBLE_DEVICES=0 bash scripts/train/train_student_v3_9b.sh    # pin GPU
#     MAX_LENGTH=4608 bash scripts/train/train_student_v3_9b.sh           # safer margin
#     EPOCHS=2 bash scripts/train/train_student_v3_9b.sh                  # shorter run
#     OPTIM=paged_adamw_32bit bash scripts/train/train_student_v3_9b.sh   # fallback if
#                                                                   # bitsandbytes
#                                                                   # 8-bit fails
#                                                                   # on Blackwell

set -euo pipefail

cd "$(dirname "$0")/.."
mkdir -p logs

PYTHON="python"

BASE_MODEL="${BASE_MODEL:-Qwen/Qwen3.5-9B}"
SAMPLES="${SAMPLES:-data/processed/philly_samples.jsonl}"
TEACHER="${TEACHER:-data/teacher/philly_teacher_qwen35.jsonl}"
OUTPUT="${OUTPUT:-ckpt/student-v3-sft-9b}"
MERGED_OUTPUT="${MERGED_OUTPUT:-ckpt/student-v3-sft-9b-merged}"
EPOCHS="${EPOCHS:-3}"
LR="${LR:-5e-5}"
WARMUP_STEPS="${WARMUP_STEPS:-20}"
MAX_LENGTH="${MAX_LENGTH:-4096}"
BATCH_SIZE="${BATCH_SIZE:-1}"
GRAD_ACCUM="${GRAD_ACCUM:-8}"
ATTN_IMPL="${ATTN_IMPL:-flash_attention_2}"
OPTIM="${OPTIM:-paged_adamw_8bit}"
WANDB_RUN_NAME="${WANDB_RUN_NAME:-v3-sft-fullft-q35-9b}"

# Default to GPU 0 when caller does not override. sbatch sets this automatically.
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"

# bitsandbytes occasionally picks the wrong CUDA runtime when multiple are
# present; force it to the same cu13 libs that torch 2.11 uses.
export BNB_CUDA_VERSION="${BNB_CUDA_VERSION:-130}"

TS="$(date +%Y%m%dT%H%M%S)"
LOG="logs/v3-sft-9b-${TS}.log"

echo "=== scripts/train/train_student_v3_9b.sh ==="
echo "host:       $(hostname)"
echo "gpu:        ${CUDA_VISIBLE_DEVICES}"
echo "python:     ${PYTHON}"
echo "base:       ${BASE_MODEL}"
echo "output:     ${OUTPUT}"
echo "merged:     ${MERGED_OUTPUT}"
echo "epochs:     ${EPOCHS}    lr: ${LR}    warmup: ${WARMUP_STEPS}"
echo "max_length: ${MAX_LENGTH}    bs: ${BATCH_SIZE}    grad_accum: ${GRAD_ACCUM}"
echo "attn_impl:  ${ATTN_IMPL}"
echo "optim:      ${OPTIM}"
echo "wandb:      ${WANDB_RUN_NAME}"
echo "log file:   ${LOG}"
echo

exec "${PYTHON}" scripts/train/train_student.py \
    --samples "${SAMPLES}" \
    --teacher "${TEACHER}" \
    --base "${BASE_MODEL}" \
    --output "${OUTPUT}" \
    --merged-output "${MERGED_OUTPUT}" \
    --no-lora \
    --epochs "${EPOCHS}" \
    --learning-rate "${LR}" \
    --warmup-steps "${WARMUP_STEPS}" \
    --max-length "${MAX_LENGTH}" \
    --batch-size "${BATCH_SIZE}" \
    --grad-accum "${GRAD_ACCUM}" \
    --attn-impl "${ATTN_IMPL}" \
    --optim "${OPTIM}" \
    --wandb-run-name "${WANDB_RUN_NAME}" \
    2>&1 | tee "${LOG}"

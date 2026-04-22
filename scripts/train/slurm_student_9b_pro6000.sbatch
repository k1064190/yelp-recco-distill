#!/bin/bash
#SBATCH --job-name=qwen35-9b-sft-pro6000
#SBATCH --partition=pro6000
#SBATCH --gres=gpu:pro6000:1
# Full-FT SFT is bf16 compute + paged-optim I/O heavy. More CPU cores help
# data collation, chat-template rendering, and bnb page-in/out under memory
# spikes. The teacher-serve script runs with --cpus-per-task=1 because vLLM
# steady-state is pure GPU, but training dataloader and optimizer paging do
# benefit from additional CPUs.
#SBATCH --cpus-per-task=8
# 64 GB host RAM: enough for 18 GB paged-optim working set + HF tokenizer
# buffers + model-loading peak (~20 GB for bf16 safetensors).
#SBATCH --mem=64G
# Wall-clock projection: 0.8B v2-sft took 4h 34m on a single 4090 (993 steps,
# 16.6 s/step). Qwen3.5-9B is ~11x parameters; Pro 6000 Blackwell is ~2.5-3x
# faster than 4090 at bf16 matmul. Net: ~4x slower per step → ~18 h for 3
# epochs. 24 h leaves margin for model-load + merge + safety.
#SBATCH --time=24:00:00
#SBATCH --output=logs/slurm_v3_sft_9b_%j.out
#SBATCH --error=logs/slurm_v3_sft_9b_%j.err

# ABOUTME: Slurm wrapper for Qwen3.5-9B student full-FT SFT on a single RTX
# ABOUTME: PRO 6000 Blackwell (96 GB). Calls train_student_v3_9b.sh.

# --- Why single Pro 6000 (same rationale as slurm_teacher_pro6000) ---
#
# Full-FT Qwen3.5-9B with fp32 AdamW overflows 4x24 GB = 96 GB (FSDP shards
# to 27 GB/GPU > 24 GB before activations). A single Pro 6000 holds
# weights + grads + paged_adamw_8bit optim (~57 GB) with ~40 GB headroom for
# activations and workspace, without FSDP complexity. See
# train_student_v3_9b.sh header for the full memory budget.
#
# --- Coordination with teacher serve ---
#
# slurm_teacher_pro6000.sbatch and this script both request a single Pro 6000.
# Qwen3.5-35B-A3B teacher (70 GB bf16) + Qwen3.5-9B SFT (~57 GB) cannot
# coexist on one card. Either:
#   (a) train v3-sft BEFORE starting the teacher serve, or
#   (b) request a second Pro 6000 allocation,
# depending on cluster partition availability.
#
# --- How to run ---
#
#   sbatch scripts/train/slurm_student_9b_pro6000.sbatch
#   scontrol show job <jobid>        # note the allocated NodeList
#   tail -f logs/slurm_v3_sft_9b_<jobid>.out
#
# Outputs:
#   ckpt/student-v3-sft-9b/            (training intermediates + tokenizer)
#   ckpt/student-v3-sft-9b-merged/     (full bf16 model, input to PTQ)
#   logs/v3-sft-9b-<timestamp>.log     (full training log w/ loss curves)

set -euo pipefail

cd "${SLURM_SUBMIT_DIR:-$(pwd)}"
mkdir -p logs

echo "=== slurm_student_9b_pro6000.sbatch ==="
echo "node:       $(hostname)"
echo "job id:     ${SLURM_JOB_ID:-(non-slurm)}"
echo "GPU:        $(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader -i 0 2>/dev/null || echo 'nvidia-smi unavailable')"
echo "env:        \$HOME/micromamba/envs/llm_gkd (transformers 5.5.3, torch 2.11+cu130, trl 1.1.0, bitsandbytes 0.49.2)"
echo "cmd:        bash scripts/train/train_student_v3_9b.sh"
echo

exec bash scripts/train/train_student_v3_9b.sh

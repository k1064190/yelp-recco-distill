#!/bin/bash
#SBATCH --job-name=qwen35-teacher-pro6000
#SBATCH --partition=pro6000
#SBATCH --gres=gpu:pro6000:1
# Serve is GPU-bound — TP=1 means uniproc_executor (single python process,
# no worker spawn), tokenization is Rust and batch=1 HTTP traffic fits in a
# single asyncio event loop. 1 CPU core works, at the cost of ~3-5 min model
# load (vs ~1-2 min with 8 cores); steady-state inference is unaffected.
#SBATCH --cpus-per-task=1
#SBATCH --mem=64G
#SBATCH --time=24:00:00
#SBATCH --output=logs/slurm_teacher_%j.out
#SBATCH --error=logs/slurm_teacher_%j.err

# ABOUTME: Slurm wrapper to serve Qwen3.5-35B-A3B as a vLLM OpenAI endpoint on a
# ABOUTME: single RTX PRO 6000 Blackwell (96 GB). Teacher for 2-node GKD setup.

# --- Why a single Pro 6000 ---
# The 4x RTX 4090 node needs TP=4 for the 35B-A3B (70 GB bf16) with
# gpu_memory_utilization capped at 0.80 to leave room for prompt_logprobs
# workspaces. A single Pro 6000 holds the full model in bf16 on one card
# with ~26 GB of headroom — no TP communication overhead, no cross-GPU
# fragmentation, prompt_logprobs=100 on 8k prompts fits comfortably.
#
# Division of labor in the 2-node setup this wrapper enables:
#   Pro 6000 (this job)  — teacher serve (TP=1, ~70 GB weights, 26 GB free)
#   4090 node (4x 4090)  — student training / PTQ / eval / bench
# The 3090 node previously used for student training is retired.
#
# How to run:
#   sbatch scripts/serve/slurm_teacher_pro6000.sbatch
# Endpoint lands on the allocated host; find it with
#   scontrol show job <jobid>   (look for 'NodeList')
# and set TEACHER_BASE_URL=http://<node>:8100/v1 on the 4090 node.
#
# Uses the same scripts/serve/serve_teacher_vllm.sh entry point as the 4090 setup —
# TP / DEVICES / GPU_MEM_UTIL are overridden via env so only one script of
# record.

set -euo pipefail

cd "${SLURM_SUBMIT_DIR:-$(pwd)}"
mkdir -p logs

export TP=1
export DEVICES=0
export GPU_MEM_UTIL=0.85
export MAX_MODEL_LEN=16384
export MAX_LOGPROBS=100

echo "=== slurm_teacher_pro6000.sbatch ==="
echo "node:       $(hostname)"
echo "job id:     ${SLURM_JOB_ID:-(non-slurm)}"
echo "GPU:        $(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader -i 0)"
echo "config:     TP=$TP, GPU_MEM_UTIL=$GPU_MEM_UTIL, MAX_MODEL_LEN=$MAX_MODEL_LEN"
echo "endpoint:   http://$(hostname):${PORT:-8100}/v1"
echo

exec bash scripts/serve/serve_teacher_vllm.sh

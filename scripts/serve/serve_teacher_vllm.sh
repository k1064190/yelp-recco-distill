#!/bin/bash
# ABOUTME: Serve the Qwen3.5-35B-A3B teacher on the 4x4090 node as an OpenAI-compatible
# ABOUTME: vLLM endpoint, so a separate node (the 3090 box) can run student GKD training.

# --- Why this architecture ---
# On-policy GKD with an in-process 35B teacher doesn't fit on 4x24GB during
# load: bf16 shards (~70 GB total) overflow even 3 GPUs of 20 GiB before bnb
# shrinks them to NF4. We resolve this by splitting work across two nodes:
#
#   Node A (this one, 4x RTX 4090):
#     - vLLM serves Qwen3.5-35B-A3B in bf16 with tensor_parallel_size=4,
#       one 17.5 GB weight shard per GPU. Exposes OpenAI-compatible HTTP
#       endpoint (--host 0.0.0.0 --port 8100 by default).
#     - Student training never runs here; the 4x 4090 box is dedicated teacher.
#
#   Node B (the 4x RTX 3090 box):
#     - Runs the student trainer (Qwen3.5-0.8B full FT + on-policy GKD).
#     - Queries this node's /v1/chat/completions with logprobs=top-K per token
#       to get teacher logits; computes token-level JSD.
#
# The student-side GKD loop needs a rewrite: TRL's GKDTrainer assumes an
# in-process teacher_model. See scripts/train/train_student_gkd.py TODO — we need
# an HTTPTeacherAdapter that implements .forward() via a vLLM OpenAI client
# with logprobs=K; K ≥ 20 is usually enough to approximate the full-vocab KL
# and vLLM exposes up to --max-logprobs.

set -euo pipefail

PORT=${PORT:-8100}
# MAX_MODEL_LEN tuning: default lowered from 8192 → 4096 on 2026-04-14 16:35
# KST after a second OOM. Qwen3.5's 248k vocab makes prompt_logprobs=50 × 2316
# tokens ≈ 1.75 GB workspace, which exceeded the free budget with 8192 KV
# cache. Halving max_model_len releases ~2 GB KV cache per GPU, enough to
# absorb the workspace spike. Pro 6000 will bump this back to 16384 via
# slurm_teacher_pro6000.sbatch.
MAX_MODEL_LEN=${MAX_MODEL_LEN:-4096}
MAX_LOGPROBS=${MAX_LOGPROBS:-100}
# max_num_seqs governs how many Mamba cache blocks vLLM reserves — Qwen3.5's
# Gated DeltaNet layers need one block per concurrent decode sequence. vLLM's
# default 256 overshoots the budget we have left after weights + KV cache;
# with GPU_MEM_UTIL=0.80 on 4× 4090 we can only fit ~165 Mamba blocks, so
# startup fails. Our workload is batch=1 HTTP requests (student trainer
# sequential forward pass), so 64 is plenty and gives margin to KV cache.
MAX_NUM_SEQS=${MAX_NUM_SEQS:-64}
# GPU_MEM_UTIL tuning history (4× RTX 4090, TP=4):
#   - 0.85 default: OOM under prompt_logprobs=50 × 2316-token prompt workload.
#     A 970 MiB workspace alloc failed with 637 MiB free; EngineCore died
#     (logs/teacher_vllm_serve.log, 2026-04-14 15:16 KST). See PROBLEMS P-X.
#   - 0.75: too low — "No available memory for the cache blocks" at startup
#     (weights 17.5 GB/GPU + activations > 18 GB budget; KV cache has nothing
#     to reserve).
#   - 0.80 (current): ~1.2 GB KV cache + ~3.6 GB out-of-budget room for
#     prompt_logprobs workspaces. Confirmed stable as of 2026-04-14 16:30 KST.
# On Pro 6000 (96 GB), overrideable to 0.85 via env: `GPU_MEM_UTIL=0.85 bash …`
GPU_MEM_UTIL=${GPU_MEM_UTIL:-0.80}
# TP default matches 4× 4090 layout. Single Pro 6000 (96 GB) → set TP=1 via
# env: `TP=1 DEVICES=0 bash scripts/serve/serve_teacher_vllm.sh`. Or use the
# Pro 6000 Slurm wrapper `scripts/serve/slurm_teacher_pro6000.sbatch`.
TP=${TP:-4}
DEVICES=${DEVICES:-0,1,2,3}
MODEL=${MODEL:-Qwen/Qwen3.5-35B-A3B}
SERVED_NAME=${SERVED_NAME:-qwen35-teacher}
HOST=${HOST:-0.0.0.0}

PY=python
LOGDIR=$(dirname "$0")/../logs
mkdir -p "$LOGDIR"
LOG="$LOGDIR/teacher_vllm_serve.log"

echo "=== serve_teacher_vllm.sh ==="
echo "model:     $MODEL"
echo "served-as: $SERVED_NAME"
echo "host:      $HOST:$PORT"
echo "max-len:   $MAX_MODEL_LEN"
echo "max-logprobs: $MAX_LOGPROBS"
echo "gpu-mem-util: $GPU_MEM_UTIL"
echo "log:       $LOG"
echo

# workaround (harmless on Ada SM 8.9 Lovelace, required on Ampere SM 8.6).
export VLLM_ATTENTION_BACKEND=FLASH_ATTN

# expandable_segments reduces PyTorch allocator fragmentation — matters when a
# request reserves a large contiguous workspace (e.g. prompt_logprobs=50 over a
# long prompt) and tripped over small unallocated fragments scattered by the
# KV cache pool. Empirically this combined with GPU_MEM_UTIL=0.75 kept the
# server stable against the 3090-node GKD workload that crashed the 0.85 run.
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"

# NOTE: NO --enforce-eager here. The original flag was a carry-over from
# vLLM 0.11 + Ampere + W4A16 flashinfer. On the current 0.19.1rc1 + Ada SM 8.9
# + bf16 teacher + FLASH_ATTN backend stack, the same model runs cleanly with
# CUDA graphs enabled — the offline benchmark (vllm-teacher, 4.9 ms/tok) was
# collected with graphs on, which is ~10x faster than the previous HTTP-mode
# measurement (51.4 ms/tok with eager). Keeping graphs on means ~45 s of
# one-time profile / compile at startup; per-request latency pays it back.
exec env CUDA_VISIBLE_DEVICES="$DEVICES" "$PY" -m vllm.entrypoints.openai.api_server \
    --model "$MODEL" \
    --served-model-name "$SERVED_NAME" \
    --dtype bfloat16 \
    --tensor-parallel-size "$TP" \
    --max-model-len "$MAX_MODEL_LEN" \
    --max-logprobs "$MAX_LOGPROBS" \
    --max-num-seqs "$MAX_NUM_SEQS" \
    --gpu-memory-utilization "$GPU_MEM_UTIL" \
    --host "$HOST" \
    --port "$PORT" \
    --trust-remote-code \
    2>&1 | tee "$LOG"

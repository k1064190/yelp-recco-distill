#!/bin/bash
# ABOUTME: Launch the teacher vLLM with `trl vllm-serve` (TRL's custom endpoints)
# ABOUTME: for DistillationTrainer's VLLMClient -- not OpenAI-compatible.

# Why this wrapper exists
# -----------------------
# TRL 1.1.0's DistillationTrainer talks to an *external* teacher through
# VLLMClient (trl/generation/vllm_client.py), which hits TRL-specific
# endpoints:
#
#     GET  /health/
#     POST /generate/
#     POST /get_sequence_logprobs/
#     POST /init_communicator/   (unused when teacher weights are fixed)
#
# These are served only by `trl vllm-serve`, NOT by
# `vllm.entrypoints.openai.api_server` (which is what
# scripts/serve/serve_teacher_vllm.sh and scripts/serve/launch_teacher_pro6000.sh use for
# eval / offline inference). Running DistillationTrainer against the
# OpenAI-compatible teacher will 404 on `/generate/`.
#
# Both launchers can coexist: OpenAI server on port 8100 for eval, this
# launcher on port 8200 for distillation training. Or stop one, start the
# other -- they share the same model weights and cache.
#
# Blackwell SM 12.0 notes are inherited from launch_teacher_pro6000.sh:
# flashinfer 0.6.7 requires CUDA >= 12.9 for its JIT MoE kernels, so we load
# cuda/13.0 and set FLASHINFER_CUDA_ARCH_LIST="12.0f".

set -euo pipefail

# Project root = two levels up from scripts/serve/ (post-2026-04-19 refactor).
cd "$(dirname "$0")/../.."

# --- CUDA 13.0 toolchain for flashinfer JIT builds ---
# shellcheck disable=SC1091
source /etc/profile.d/modules.sh
module unload cuda 2>/dev/null || true
module load cuda/13.0

export CUDA_HOME=/usr/local/cuda-13.0
export PATH="$CUDA_HOME/bin:$PATH"
export LD_LIBRARY_PATH="$CUDA_HOME/lib64:${LD_LIBRARY_PATH:-}"

# Belt-and-suspenders: force Blackwell sm_120f gencode regardless of
# autodetection path.
export FLASHINFER_CUDA_ARCH_LIST="12.0f"

# vLLM tuning carried over from serve_teacher_vllm.sh: expandable_segments
# reduces allocator fragmentation when the server reserves large contiguous
# workspaces for get_sequence_logprobs requests.
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"

# --- Serve config ---
export TP=${TP:-1}
export DEVICES=${DEVICES:-1}
export PORT=${PORT:-8200}                 # distinct from 8100 (OpenAI serve)
export GPU_MEM_UTIL=${GPU_MEM_UTIL:-0.85}
export MAX_MODEL_LEN=${MAX_MODEL_LEN:-16384}
export MODEL=${MODEL:-Qwen/Qwen3.5-35B-A3B}
export HOST=${HOST:-0.0.0.0}

# Uses the matching environment because `trl vllm-serve` needs BOTH trl (installed via
# --no-deps to avoid fsspec/dill conflicts with datasets 4.x) and a working
# vLLM. the matching environment is training-only (no vLLM); the matching environment is serve-side.
TRL_BIN=trl
LOGDIR="$(pwd)/logs"
mkdir -p "$LOGDIR"
LOG="$LOGDIR/teacher_trl_serve.log"

echo "=== launch_teacher_trl_serve.sh ==="
echo "host:           $(hostname)"
echo "nvcc:           $(which nvcc)"
nvcc --version | tail -1
echo "cuda home:      $CUDA_HOME"
echo "flashinfer arch: $FLASHINFER_CUDA_ARCH_LIST"
echo "serve device:   GPU $DEVICES"
echo "port:           $PORT"
echo "model:          $MODEL"
echo "log:            $LOG"
echo

# --enforce-eager is mandatory here: trl vllm-serve hardcodes max_num_seqs
# at vLLM's default 1024 (no CLI knob), which exceeds the ~505 Mamba cache
# blocks allocated for Qwen3.5 Gated DeltaNet at gpu_memory_utilization=0.85.
# Disabling CUDA graph capture sidesteps the Mamba block budget altogether.
# Throughput hit is acceptable for training-path HTTP logprob queries; we
# keep the graph-enabled OpenAI serve (serve_teacher_vllm.sh) for eval.
exec env CUDA_VISIBLE_DEVICES="$DEVICES" "$TRL_BIN" vllm-serve \
    --model "$MODEL" \
    --tensor-parallel-size "$TP" \
    --gpu-memory-utilization "$GPU_MEM_UTIL" \
    --dtype bfloat16 \
    --max-model-len "$MAX_MODEL_LEN" \
    --host "$HOST" \
    --port "$PORT" \
    --enforce-eager \
    --trust-remote-code \
    2>&1 | tee "$LOG"

#!/bin/bash
# ABOUTME: Launch the student (v3-sft) via `trl vllm-serve` as an external
# ABOUTME: vLLM process so DistillationTrainer can connect in server mode.

# Why this exists
# ---------------
# vllm_mode="colocate" (default) loads the student vLLM engine inside the
# training process. On vLLM 0.19 + torch 2.11+cu130 + Blackwell SM 12.0, this
# triggers all-NaN logits on the first forward pass -- suspected CUDA context
# pollution from the HF accelerate init that precedes LLM() construction.
# Standalone `trl vllm-serve` + the same ckpt + same regex works fine, so we
# split student generation into its own process via vllm_mode="server".
#
# GPU 0 holds this serve; training continues on GPU 2; teacher serve on GPU 1.
#
# Wait for /health/ before launching training:
#     until curl -sf http://localhost:8300/health/; do sleep 5; done

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

# Blackwell SM 12.0 override for flashinfer.
export FLASHINFER_CUDA_ARCH_LIST="12.0f"

# expandable_segments helps when the engine reserves large contiguous
# workspaces (inherits the logic from launch_teacher_trl_serve.sh).
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"

# --- Serve config ---
export TP=${TP:-1}
export DEVICES=${DEVICES:-0}                       # GPU 0 is the free slot
export PORT=${PORT:-8300}                          # distinct from teacher 8200
export GPU_MEM_UTIL=${GPU_MEM_UTIL:-0.35}          # student is small (0.8B); leave headroom for other GPU 0 workloads
export MAX_MODEL_LEN=${MAX_MODEL_LEN:-4096}
export MODEL=${MODEL:-/workspace/projects/LLM_distillation/ckpt/student-v3-sft-merged_vllm_vlm}
export HOST=${HOST:-0.0.0.0}

# the matching environment has `trl` (installed via --no-deps) + vllm 0.19 dev build. Same
# runtime as colocate but in its own process.
TRL_BIN=trl

LOGDIR="$(pwd)/logs"
mkdir -p "$LOGDIR"
LOG="$LOGDIR/student_trl_serve.log"

echo "=== launch_student_trl_serve.sh ==="
echo "host:           $(hostname)"
echo "cuda home:      $CUDA_HOME"
echo "flashinfer arch: $FLASHINFER_CUDA_ARCH_LIST"
echo "serve device:   GPU $DEVICES"
echo "port:           $PORT"
echo "model:          $MODEL"
echo "log:            $LOG"
echo

# --enforce-eager mirrors the teacher launcher; trl vllm-serve hardcodes
# max_num_seqs which can exceed Mamba cache blocks under CUDA graph capture.
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

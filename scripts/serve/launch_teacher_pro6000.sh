#!/bin/bash
# ABOUTME: Launch serve_teacher_vllm.sh on a single Pro 6000 Blackwell (SM 12.0)
# ABOUTME: with the CUDA 13.0 toolkit bindings needed by flashinfer 0.6.7.

# Why this wrapper exists
# -----------------------
# flashinfer 0.6.7 JIT-compiles SM 12.x MoE kernels via torch.utils.cpp_extension
# and hard-requires CUDA >= 12.9 (see
# flashinfer/compilation_context.py::_normalize_cuda_arch). The cluster default
# module is cuda/12.8, which makes flashinfer throw
#   RuntimeError: No supported CUDA architectures found for major versions [12]
# during first-use kernel build (fused_moe::flashinfer_cutlass_fused_moe), and
# the vLLM EngineCore dies mid-startup. cuda/13.0 is available as a module,
# matches torch 2.11.0+cu130's runtime, and lets flashinfer emit
# compute_120f/sm_120f gencode that nvcc 13.0 accepts.
#
# FLASHINFER_CUDA_ARCH_LIST="12.0f" is belt-and-suspenders: even with CUDA
# 13.0 loaded, some code paths autodetect via torch.cuda.get_device_capability
# which returns (12, 0); the explicit env var skips the CUDA-version gate.

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

# --- flashinfer override (Blackwell SM 12.0 → sm_120f) ---
export FLASHINFER_CUDA_ARCH_LIST="12.0f"

# --- Pro 6000 single-GPU serve config ---
# Defaults pick GPU 7 (PCI D8:00.0) — override by setting DEVICES=N before
# invoking this wrapper (e.g. DEVICES=3 bash scripts/serve/launch_teacher_pro6000.sh).
export TP=${TP:-1}
export DEVICES=${DEVICES:-7}
export GPU_MEM_UTIL=${GPU_MEM_UTIL:-0.85}
export MAX_MODEL_LEN=${MAX_MODEL_LEN:-16384}
export MAX_LOGPROBS=${MAX_LOGPROBS:-100}

echo "=== launch_teacher_pro6000.sh ==="
echo "host:         $(hostname)"
echo "nvcc:         $(which nvcc)"
nvcc --version | tail -1
echo "cuda home:    $CUDA_HOME"
echo "flashinfer arch: $FLASHINFER_CUDA_ARCH_LIST"
echo "serve device: GPU $DEVICES"
echo

exec bash scripts/serve/serve_teacher_vllm.sh

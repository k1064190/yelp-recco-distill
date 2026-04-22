#!/bin/bash
# ABOUTME: Create llm_blackwell micromamba env for PRO 6000 Blackwell (SM 12.0)
# ABOUTME: with torch 2.11+cu130, flash-attn built from source, and full training stack.

# This env replaces llm_gkd on Blackwell nodes. The llm_gkd env has a
# broken flash_attn (pre-built wheel compiled against a different torch
# ABI → undefined symbol on import). This env builds flash-attn from
# source against the actual torch 2.11+cu130 we're using.
#
# Key differences from llm_gkd:
#   - flash_attn built from source with TORCH_CUDA_ARCH_LIST="12.0"
#   - CUDA 13.0 toolkit used for compilation
#   - Verified on RTX PRO 6000 Blackwell (SM 12.0, 96 GB)
#
# Usage:
#   bash scripts/env/setup_blackwell_env.sh
#
# After setup, use:
#   python scripts/train/train_student.py \
#     --attn-impl flash_attention_2 ...

set -euo pipefail

# Source micromamba shell init (it's a shell function, not a binary)
export MAMBA_ROOT_PREFIX="$HOME/micromamba"
eval "$("$MAMBA_ROOT_PREFIX/bin/micromamba" shell hook -s bash)"

ENV_NAME="llm_blackwell"
PYTHON_VERSION="3.11"
CUDA_HOME="/usr/local/cuda-13.0"

echo "=== setup_blackwell_env.sh ==="
echo "env:    $ENV_NAME"
echo "python: $PYTHON_VERSION"
echo "cuda:   $CUDA_HOME"
echo ""

# ---- 0. Pre-checks ----
if [ ! -d "$CUDA_HOME" ]; then
    echo "ERROR: CUDA 13.0 not found at $CUDA_HOME"
    echo "Run: module load cuda/13.0"
    exit 1
fi

if ! command -v micromamba &>/dev/null; then
    echo "ERROR: micromamba not found"
    exit 1
fi

# ---- 1. Create env ----
if [ -d "$HOME/micromamba/envs/$ENV_NAME" ]; then
    echo "WARNING: env $ENV_NAME already exists. Remove it first with:"
    echo "  micromamba env remove -n $ENV_NAME"
    exit 1
fi

echo "--- creating micromamba env ---"
micromamba create -n "$ENV_NAME" python="$PYTHON_VERSION" -y -c conda-forge

PIP="$HOME/micromamba/envs/$ENV_NAME/bin/pip"
PYTHON="$HOME/micromamba/envs/$ENV_NAME/bin/python"

# ---- 2. Install torch 2.11+cu130 ----
echo "--- installing torch 2.11+cu130 ---"
$PIP install torch==2.11.0 torchvision torchaudio \
    --index-url https://download.pytorch.org/whl/cu130

# Verify torch CUDA
$PYTHON -c "
import torch
print(f'torch {torch.__version__}, CUDA {torch.version.cuda}')
assert torch.cuda.is_available(), 'CUDA not available'
cap = torch.cuda.get_device_capability()
print(f'GPU SM: {cap[0]}.{cap[1]}')
"

# ---- 3. Install training stack ----
echo "--- installing training stack ---"
$PIP install \
    "transformers>=5.5,<6" \
    "trl>=1.1,<2" \
    "datasets>=4.8" \
    "accelerate>=1.13" \
    "bitsandbytes>=0.49" \
    "deepspeed>=0.18" \
    "wandb>=0.25" \
    "einops" \
    "packaging" \
    "ninja"

# ---- 4. Build flash-attn from source ----
echo "--- building flash-attn from source (this takes 5-15 min) ---"
echo "Using CUDA_HOME=$CUDA_HOME, nvcc=$CUDA_HOME/bin/nvcc"

export CUDA_HOME="$CUDA_HOME"
export PATH="$CUDA_HOME/bin:$PATH"
export TORCH_CUDA_ARCH_LIST="12.0"
export MAX_JOBS=4

# flash-attn requires source build for non-standard SM targets
$PIP install flash-attn --no-build-isolation 2>&1 | tail -20

# Verify flash-attn import
echo "--- verifying flash-attn ---"
$PYTHON -c "
import flash_attn
print(f'flash_attn {flash_attn.__version__} OK')
from flash_attn import flash_attn_func
print('flash_attn_func imported OK')
" || {
    echo "WARNING: flash-attn import failed. Training will fall back to sdpa."
    echo "You can retry the build manually with:"
    echo "  CUDA_HOME=$CUDA_HOME TORCH_CUDA_ARCH_LIST=12.0 $PIP install flash-attn --no-build-isolation"
}

# ---- 5. Final verification ----
echo ""
echo "=== verification ==="
$PYTHON -c "
import torch
print(f'torch:          {torch.__version__} (CUDA {torch.version.cuda})')
print(f'GPU:            {torch.cuda.get_device_name()}')
print(f'SM:             {torch.cuda.get_device_capability()}')

import transformers
print(f'transformers:   {transformers.__version__}')

import trl
print(f'trl:            {trl.__version__}')

import bitsandbytes
print(f'bitsandbytes:   {bitsandbytes.__version__}')

import datasets
print(f'datasets:       {datasets.__version__}')

try:
    import flash_attn
    print(f'flash_attn:     {flash_attn.__version__} ✓')
except Exception as e:
    print(f'flash_attn:     FAILED ({e})')

print()
print(f'env ready: {\"$HOME/micromamba/envs/$ENV_NAME/bin/python\"}')
"

echo ""
echo "=== done ==="
echo "Use: \$HOME/micromamba/envs/$ENV_NAME/bin/python"

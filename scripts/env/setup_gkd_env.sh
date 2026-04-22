#!/bin/bash
# ABOUTME: Create the `llm_gkd` micromamba env for GKD (on-policy distillation).
# ABOUTME: Requires transformers 5.x for Qwen3.5 (qwen3_5) Gated DeltaNet architecture.

# --- Why a separate environment? ---
# The main llm_exp env pins transformers<5.0 (peft 0.18.1 / llmcompressor 0.9.0.2 break
# on 5.x lazy imports). Qwen3.5-0.8B and Qwen3.5-35B-A3B both use the qwen3_5 model
# type introduced after transformers 4.57. GKD training needs transformers 5.x with
# compatible trl (experimental GKDTrainer), peft, and bitsandbytes (for 4-bit teacher).
#
# This env is ONLY for GKD training. Do not use it for the rest of the pipeline
# (SFT w/ Qwen3-4B, PTQ, bench, eval, judge) — those stay in llm_exp.

set -euo pipefail

ENV_NAME=llm_gkd
PY_VERSION=3.11
MICROMAMBA=$HOME/micromamba/bin/micromamba

log() { printf '[%s] %s\n' "$(date +%H:%M:%S)" "$*"; }

# ---------- 1. Create env ----------
if [ -d "$HOME/micromamba/envs/$ENV_NAME" ]; then
    log "env $ENV_NAME already exists — remove it first if you want a fresh install:"
    log "  $MICROMAMBA env remove -n $ENV_NAME -y"
    exit 1
fi

log "creating micromamba env: $ENV_NAME (python=$PY_VERSION)"
$MICROMAMBA create -n "$ENV_NAME" python="$PY_VERSION" -c conda-forge -y

PIP=$HOME/micromamba/envs/$ENV_NAME/bin/pip
PY=$HOME/micromamba/envs/$ENV_NAME/bin/python

# ---------- 2. Install torch (match llm_qwen35: torch 2.11 + cu130) ----------
log "installing torch 2.11 + cu130"
$PIP install torch==2.11.0 torchvision torchaudio \
    --index-url https://download.pytorch.org/whl/cu130

# ---------- 3. Install transformers from main (for qwen3_5 support) ----------
# Pin to a known-working commit or use latest release that includes qwen3_5.
# If a stable 5.x release exists with qwen3_5, prefer that over git main.
log "installing transformers (main branch for qwen3_5 arch)"
$PIP install "transformers>=5.0" 2>/dev/null || \
    $PIP install git+https://github.com/huggingface/transformers.git

# ---------- 4. Install ML training stack ----------
# Note: peft is intentionally excluded — Qwen3.5-0.8B is small enough for full
# fine-tuning (~15 GB on single 24 GB GPU: 1.6 weights + 1.6 grads + 6.4 AdamW
# states + 5 activations). Full FT avoids LoRA target auto-detection on the
# Gated DeltaNet architecture and removes the transformers<5 vs peft 0.18 pin
# conflict seen in llm_exp. If a future GKD run wants LoRA instead, add
# `peft>=0.15` back here.
log "installing trl, bitsandbytes, datasets, accelerate, wandb (no peft — full FT path)"
$PIP install \
    "trl>=1.0" \
    "bitsandbytes>=0.43" \
    "datasets>=4.0" \
    "accelerate>=1.0" \
    "wandb" \
    "scipy"

# ---------- 5. Install flash-attn (optional, best-effort) ----------
log "attempting flash-attn install (pre-built wheel, best effort)"
$PIP install flash-attn --no-build-isolation 2>/dev/null || \
    log "flash-attn install failed — GKD will fall back to sdpa or eager attention"

# ---------- 6. Verify ----------
log "=== verification ==="
$PY -c "
import torch, transformers, trl, bitsandbytes, datasets
print(f'torch:          {torch.__version__}')
print(f'transformers:   {transformers.__version__}')
print(f'trl:            {trl.__version__}')
print(f'bitsandbytes:   {bitsandbytes.__version__}')
print(f'datasets:       {datasets.__version__}')
print(f'CUDA available: {torch.cuda.is_available()}')
print(f'GPU count:      {torch.cuda.device_count()}')

# Verify qwen3_5 model type is recognized
from transformers import AutoConfig
cfg = AutoConfig.from_pretrained('Qwen/Qwen3.5-0.8B')
print(f'Qwen3.5-0.8B:  OK (model_type={cfg.model_type})')

# Verify GKDTrainer is importable
import os
os.environ['TRL_EXPERIMENTAL_SILENCE'] = '1'
from trl.experimental.gkd import GKDTrainer, GKDConfig
print(f'GKDTrainer:     OK (from trl.experimental.gkd)')
"

log "=== done ==="
log "activate with: micromamba activate $ENV_NAME"
log "or use directly: \$HOME/micromamba/envs/$ENV_NAME/bin/python"

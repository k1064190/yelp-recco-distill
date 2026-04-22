#!/bin/bash
# ABOUTME: Create the `llm_ptq_qwen35` micromamba env for Qwen3.5 quantization.
# ABOUTME: Separates PTQ tooling (llmcompressor, GPTQ) from the distillation env (llm_gkd).

# --- Why a separate environment from llm_gkd? ---
# Design note: "I think you should separate quantization and
# distillation env". Reasoning: when llmcompressor was attempted in llm_gkd, its
# pip resolver downgraded transformers 5.5.3 → 4.57.6 (matching llmcompressor's
# Hugging Face deps), which broke Qwen3.5 (qwen3_5 model_type) loading and
# corrupted torch (~orch leftover dir). Cleanest fix: separate envs so neither
# overwrites the other.
#
# Env split:
#   llm_exp           — legacy v1 (Qwen3-4B) train + PTQ + serve. transformers<5.
#   llm_gkd           — Qwen3.5 distillation training (SFT + GKD). transformers≥5, no llmcompressor, no peft.
#   llm_ptq_qwen35    — Qwen3.5 PTQ (W4A16). transformers≥5 + llmcompressor.

set -euo pipefail

ENV_NAME=llm_ptq_qwen35
PY_VERSION=3.11
MICROMAMBA=$HOME/micromamba/bin/micromamba

log() { printf '[%s] %s\n' "$(date +%H:%M:%S)" "$*"; }

if [ -d "$HOME/micromamba/envs/$ENV_NAME" ]; then
    log "env $ENV_NAME already exists — remove it first if you want a fresh install:"
    log "  $MICROMAMBA env remove -n $ENV_NAME -y"
    exit 1
fi

log "creating micromamba env: $ENV_NAME (python=$PY_VERSION)"
$MICROMAMBA create -n "$ENV_NAME" python="$PY_VERSION" -c conda-forge -y

PIP=$HOME/micromamba/envs/$ENV_NAME/bin/pip
PY=$HOME/micromamba/envs/$ENV_NAME/bin/python

# torch matched to llm_gkd (2.11 + cu130) so saved checkpoints from training
# load identically here.
log "installing torch 2.11.0 + cu130"
$PIP install torch==2.11.0 torchvision torchaudio \
    --index-url https://download.pytorch.org/whl/cu130

log "installing transformers ≥5 (qwen3_5 arch support)"
$PIP install "transformers>=5.0" 2>/dev/null || \
    $PIP install git+https://github.com/huggingface/transformers.git

# llmcompressor 0.10.0.1 release pins transformers<=4.57.6 (incompatible with
# Qwen3.5 — qwen3_5 model_type needs transformers 5.x). The git main branch
# already supports transformers 5.x but requires an unreleased compressed-tensors
# (>=0.15.1a2). Both must come from git main as a matched pair.
log "installing llmcompressor + compressed-tensors from git main (transformers 5.x compatible)"
$PIP install "datasets>=4.0"
$PIP install --no-deps "git+https://github.com/vllm-project/llm-compressor.git"
$PIP install --no-deps "git+https://github.com/neuralmagic/compressed-tensors.git"
# Re-pin transformers + huggingface_hub after llmcompressor pip resolver may have downgraded
$PIP install --force-reinstall --no-deps "transformers==5.5.3" "huggingface_hub>=1.10"

# bitsandbytes required for NF4 calibration paths in llmcompressor.
log "installing bitsandbytes + accelerate"
$PIP install "bitsandbytes>=0.43" "accelerate>=1.0"

# vllm (optional) for sanity-check generation after quant.
log "installing vllm (sanity-check generation post-quant)"
$PIP install "vllm>=0.11" || log "vllm install failed — quant still produces a usable checkpoint, only sanity-check disabled"

log "=== verification ==="
$PY -c "
import torch, transformers, llmcompressor, bitsandbytes
print(f'torch:           {torch.__version__}')
print(f'transformers:    {transformers.__version__}')
print(f'llmcompressor:   {llmcompressor.__version__}')
print(f'bitsandbytes:    {bitsandbytes.__version__}')
from transformers import AutoConfig
cfg = AutoConfig.from_pretrained('Qwen/Qwen3.5-0.8B')
print(f'Qwen3.5-0.8B:    OK (model_type={cfg.model_type})')
"

log "=== done ==="
log "use directly: \$HOME/micromamba/envs/$ENV_NAME/bin/python"

#!/bin/bash
# ABOUTME: Fix the failed downgrade: pin torch==2.8.0 and transformers<5.0 via
# ABOUTME: a pip constraint file so --upgrade cannot drag torch back up to 2.10.

PIP=pip
PY=python

export CUDA_HOME=/usr/local/cuda-12.8
export PATH=/usr/local/cuda-12.8/bin:$PATH
export LD_LIBRARY_PATH=/usr/local/cuda-12.8/lib64:$LD_LIBRARY_PATH
export TORCH_CUDA_ARCH_LIST="8.6"

log() { printf '[%s] %s\n' "$(date +%H:%M:%S)" "$*"; }

CONSTRAINTS=/tmp/downgrade-constraints.txt
cat > "$CONSTRAINTS" << 'EOF'
# Pin torch family to the 2.8 line so nothing can upgrade it transitively.
torch==2.8.0
torchvision==0.23.0
torchaudio==2.8.0
# Keep transformers on 4.x — 5.x requires newer torch and breaks peft/llmcompressor
# lazy-import resolution.
transformers<5.0
EOF

log "constraint file:"
cat "$CONSTRAINTS"
echo

log "=== step 1: uninstall torch + vllm + flash-attn + transformers (force clean) ==="
$PIP uninstall -y torch torchvision torchaudio vllm flash-attn transformers 2>&1 || true

log "=== step 2: install vllm 0.11.0 with constraint pinning torch==2.8.0 ==="
$PIP install -c "$CONSTRAINTS" "vllm==0.11.0"

log "=== step 3: verify torch is 2.8.x ==="
$PY -c "
import torch
print(f'torch: {torch.__version__}')
print(f'cuda runtime: {torch.version.cuda}')
print(f'cuda available: {torch.cuda.is_available()}')
print(f'device_count: {torch.cuda.device_count()}')
assert torch.__version__.startswith('2.8.'), f'expected torch 2.8.x, got {torch.__version__}'
print('torch version check: OK')
"

log "=== step 4: install pre-built flash-attn 2.8.3 wheel (torch 2.8, cxx11abiTRUE) ==="
WHEEL_URL="https://github.com/Dao-AILab/flash-attention/releases/download/v2.8.3/flash_attn-2.8.3+cu12torch2.8cxx11abiTRUE-cp311-cp311-linux_x86_64.whl"
$PIP install --no-build-isolation --no-deps "$WHEEL_URL"

log "=== step 5: install transformers 4.57.6 explicitly (safe for peft + llmcompressor) ==="
$PIP install -c "$CONSTRAINTS" "transformers==4.57.6"

log "=== step 6: install llmcompressor (constraint-guarded) ==="
$PIP install -c "$CONSTRAINTS" llmcompressor

log "=== step 7: install trl / peft / accelerate / datasets / bitsandbytes (no --upgrade) ==="
# Not using --upgrade: only install missing. Constraint keeps torch/transformers pinned.
$PIP install -c "$CONSTRAINTS" \
    "trl" "peft" "accelerate" \
    "datasets>=4.7.0,<5.0" \
    "bitsandbytes" "sentencepiece"

log "=== step 8: ensure wandb/rouge/nltk present ==="
$PIP install -c "$CONSTRAINTS" wandb rouge-score nltk

log "=== final verification ==="
$PY << 'PYEOF'
import importlib, traceback

mods = [
    "torch", "transformers", "accelerate", "peft", "trl",
    "datasets", "bitsandbytes", "vllm", "flash_attn",
    "llmcompressor", "gemini_parallel",
]
ok_lines, fail_lines = [], []
for name in mods:
    try:
        m = importlib.import_module(name)
        ver = getattr(m, "__version__", "?")
        ok_lines.append(f"  OK   {name:<18} {ver}")
    except Exception as e:
        fail_lines.append(f"  FAIL {name:<18} {type(e).__name__}: {str(e)[:120]}")

print("--- import OK ---")
for line in ok_lines: print(line)
print("--- import FAIL ---")
if not fail_lines:
    print("  (none)")
else:
    for line in fail_lines: print(line)

print()
import torch
print("--- torch cuda ---")
print(f"torch version:   {torch.__version__}")
print(f"cuda runtime:    {torch.version.cuda}")
print(f"cuda available:  {torch.cuda.is_available()}")
print(f"device_count:    {torch.cuda.device_count()}")
for i in range(torch.cuda.device_count()):
    p = torch.cuda.get_device_properties(i)
    mem_gb = p.total_memory / 1024**3
    print(f"  gpu {i}: {p.name} (SM {p.major}.{p.minor}, {mem_gb:.1f} GB)")
print(f"bf16 supported:  {torch.cuda.is_bf16_supported()}")

print()
print("--- flash-attn functional test ---")
try:
    from flash_attn import flash_attn_func
    q = torch.randn(1, 8, 4, 64, dtype=torch.bfloat16, device='cuda')
    k = torch.randn(1, 8, 4, 64, dtype=torch.bfloat16, device='cuda')
    v = torch.randn(1, 8, 4, 64, dtype=torch.bfloat16, device='cuda')
    out = flash_attn_func(q, k, v)
    print(f"  flash_attn_func  OK — output shape {tuple(out.shape)} dtype {out.dtype}")
    import flash_attn
    print(f"  version          {flash_attn.__version__}")
except Exception as e:
    print(f"  FAIL {type(e).__name__}: {e}")
PYEOF

log "=== DONE ==="

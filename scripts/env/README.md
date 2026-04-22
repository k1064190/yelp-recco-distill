# `scripts/env/` — micromamba environment bootstrap

Each script creates one named micromamba env from scratch. They are **not**
idempotent restarts — re-running rebuilds. Pin rationale lives in
[`../../ENV_VERSION.md`](../../ENV_VERSION.md).

| Script | Env created | Used by |
|---|---|---|
| `setup_blackwell_env.sh` | **`llm_blackwell`** (current default) | v3 9B SFT, GKD, eval on PRO 6000 (SM 12.0). torch 2.11+cu130, transformers 5.5.4, trl 1.1.0; `flash_attn` / `fla` / `causal-conv1d` **built from source** for SM 12.0. |
| `setup_gkd_env.sh` | `llm_gkd` | 0.8B SFT on non-Blackwell GPUs. torch 2.11+cu130, transformers 5.x, trl 1.1.0; **no pre-built flash_attn wheel** for this combo → `--attn-impl sdpa`. |
| `setup_ptq_qwen35_env.sh` | `llm_ptq_qwen35` | W4A16 PTQ via `llmcompressor` git main + `compressed-tensors` git main (only pair that supports transformers ≥ 5). |
| `downgrade_env_fix.sh` | `llm_exp` | Legacy path — original PTQ, GGUF eval. torch 2.8.0+cu128, transformers 4.57.3, vllm 0.11.0, flash_attn 2.8.3 (pre-built wheel). **Critical**: uses `/tmp/downgrade-constraints.txt` so transitive `--upgrade` cannot drag torch above 2.8. |

## Usage

```bash
bash scripts/env/setup_blackwell_env.sh   # ~10 min, builds 3 CUDA libs from source
```

After install:

```bash
$HOME/micromamba/envs/llm_blackwell/bin/python \
  -c "import torch, transformers, trl; print(torch.__version__, transformers.__version__, trl.__version__)"
```

## Hard constraints

- **NEVER `pip install --upgrade` without a `-c` constraint file.** A single
  unconstrained upgrade can pull `torch` from 2.8 → 2.10 transitively and
  silently break `flash-attn`.
- `transformers < 5.0` is **`llm_exp`-only**. The other three envs run
  transformers 5.x for Qwen3.5 (`qwen3_5` arch).
- `flash_attn`, `causal-conv1d`, `fla` on Blackwell SM 12.0 require **source
  builds** with `CUDA_HOME=/usr/local/cuda-13.0 TORCH_CUDA_ARCH_LIST="12.0"`.
  Pre-built PyPI wheels were compiled against incompatible torch ABI / lower
  SMs.
- `causal-conv1d` CUDA kernel does not work in DDP / multi-process — single-GPU
  only on Blackwell.

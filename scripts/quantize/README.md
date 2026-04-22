# `scripts/quantize/` — Post-Training Quantization (PTQ)

Three formats, three runtimes. All operate on a **merged FP16/BF16 student
checkpoint** as input.

| Script | Format | Runtime | Calibration | Output size | Active env |
|---|---|---|---|---|---|
| `quantize_w4a16.py` | W4A16 (GPTQ via llmcompressor / compressed-tensors) | vLLM (Marlin kernel), HF transformers | 128 in-domain teacher samples | ~0.73 GB (0.8B) / ~3.4 GB (4B) | (see `ENV_VERSION.md`) (or (see `ENV_VERSION.md`) for legacy v0/v1) |
| `quantize_nf4.py` | NF4 (bitsandbytes 4-bit NormalFloat, double-quant) | HF transformers, vLLM (`quantization=bitsandbytes`) | none — calibration-free | ~0.73 GB (0.8B) | (see `ENV_VERSION.md`) |
| `quantize_gguf.py` | Q4_K_M (llama.cpp k-quant) | llama.cpp / llama-cpp-python / Ollama / LM Studio | none | ~0.49 GB (0.8B) | (see `ENV_VERSION.md`) (conversion) |

## Pipeline

```
ckpt/student-vX-sft-merged/  ──┬─→ quantize_w4a16.py  →  ckpt/student-vX-sft-w4a16/
                               ├─→ quantize_nf4.py    →  ckpt/student-vX-sft-nf4/
                               └─→ quantize_gguf.py   →  ckpt/student-vX-sft-gguf-q4km/student-q4-k-m.gguf
```

Each output is then evaluated with the matching runtime via `../eval/`:
- W4A16 / NF4 → `../eval/eval_metrics.py` (HF transformers)
- W4A16 → `../eval/eval_metrics_vllm.py` after `../vllm_compat/merge_vlm_w4a16_for_vllm.py`
- GGUF → `../eval/eval_gguf.py` (llama-cpp-python, **must be CUDA-built**)

## W4A16 specifics

- Calibration draws from the SFT train split (in-domain), not generic. Avoids drift bias.
- vLLM `compressed-tensors` requires `quantization_config.ignore` regex extensions for VLM-shell re-hydration (`re:.*visual.*` + `re:.*mtp.*`); see `../vllm_compat/merge_vlm_w4a16_for_vllm.py`.
- **vLLM 0.11 + W4A16 + Ampere SM 8.6 + flashinfer**: assertion crash inside `flashinfer.py:972`. Fix: force `VLLM_ATTENTION_BACKEND=FLASH_ATTN` before the `vllm` import. The flag survived an env upgrade and quietly cost 10× latency until a defensive-flag audit caught it — re-check legacy workaround flags on every stack upgrade.

## GGUF specifics

- `quantize_gguf.py` shells out to `convert_hf_to_gguf.py` + `llama-quantize` from the project's local llama.cpp checkout at `/workspace/projects/llama.cpp` (built for SM 8.6 originally; rebuild for the deployment target SM).
- Default pip wheel of `llama-cpp-python` is **CPU-only**. Reinstall with `CMAKE_ARGS="-DGGML_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES=<sm>"` so `n_gpu_layers=-1` actually offloads.

## NF4 specifics

- Calibration-free, ~30 s on a single GPU. Cheapest path to "is this model quantizable at all?".
- Loads via HF native (`load_in_4bit=True`) without a packed checkpoint format.

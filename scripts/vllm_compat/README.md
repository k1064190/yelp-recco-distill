# `scripts/vllm_compat/` — Make Qwen3.5 fine-tunes loadable by vLLM

These three scripts exist because of : vLLM's `Qwen3_5ForCausalLM`
text-only path is **skeletal** (not in default ModelRegistry, missing
`IsHybrid`, no Transformers fallback). The only fully-working code path is
`Qwen3_5ForConditionalGeneration` — the VLM. So we re-hydrate our text-only
fine-tune into a VLM shell by re-injecting the base vision encoder + MTP
weights before vLLM loads it.

| Script | Input | Output | When |
|---|---|---|---|
| `rename_ckpt_for_vllm.py` | `ckpt/student-X-merged/` (HF text-only, weights prefixed `model.language_model.`) | `ckpt/student-X-merged_vllm_renamed/` (weights prefixed bare `model.`) | First attempt at. Passes vLLM's weight loader but still dies in the KV-cache planner (`page size of the layer is not divisible by the maximum page size`) because the text-only class lacks `IsHybrid`. **Kept for reference**; not the production path. |
| `merge_vlm_ckpt_for_vllm.py` | Fine-tuned text ckpt (320 tensors) + base Qwen3.5 VLM (vision 153 + MTP 15) | `ckpt/student-X-merged_vllm_vlm/` (488 tensors, ~+300 MB vs text-only) | **Production path for bf16 / NF4 vLLM serve.** Copies base's `config.json` so vLLM dispatches to `Qwen3_5ForConditionalGeneration`. Vision forward never fires without `pixel_values` → zero runtime cost, only disk overhead. |
| `merge_vlm_w4a16_for_vllm.py` | Already-W4A16 text ckpt (692 packed+scale tensors) + base vision + MTP (bf16) | `ckpt/student-X-merged_vllm_vlm_w4a16_shell/` | **Production path for W4A16 vLLM serve.** Extends `quantization_config.ignore` with `re:.*visual.*` + `re:.*mtp.*` so compressed-tensors does not try to interpret the bf16 vision tensors as packed int4. |

## Verified

5 eval prompts × 3 backends = 15 generations parse correctly. Latency on Pro
6000 / Ada SM 8.9:
- vLLM bf16 (VLM shell) — 2.3 ms/tok
- vLLM NF4 (bnb on-the-fly on the bf16 shell) — 2.0 ms/tok
- vLLM W4A16 (Marlin kernel) — 1.9 ms/tok

## Follow-on

When a future vLLM stable lands `Qwen3_5ForCausalLM + IsHybrid` natively, these
re-hydration scripts become unnecessary and we can ship text-only checkpoints
directly. Until then the VLM shell is the committed artifact.

# `scripts/eval/` — Retrieval metrics, latency, dashboard

Everything that measures a checkpoint. **Three eval entrypoints** by runtime,
**one latency bench**, **one inference-cache feeder**, **one dashboard**.

## Retrieval metrics (R@k / MRR / Kendall τ / parse rate)

| Script | Runtime | Input | Output | When to use |
|---|---|---|---|---|
| `eval_metrics.py` | HF transformers (free generation) | Merged ckpt + teacher JSONL | `data/results/eval_<tag>.json` | Default for any HF-loadable checkpoint (bf16, NF4 via `bnb`, W4A16 via `compressed-tensors`). Auto-handles Qwen3.5 thinking-mode (`enable_thinking=False`). |
| `eval_metrics_vllm.py` | vLLM HTTP (OpenAI-compatible), optional `response_format=json_schema` | Running vLLM endpoint | Same JSON shape + `_raw.jsonl` + `token_distribution` + `position_bias` | **Fair base-vs-SFT comparison** under guided JSON decoding (default). `--no-guided-json` switches to raw free-gen (unconstrained) for measuring intrinsic schema capability. `--prompt-module configs.teacher_prompt_*` swaps SYSTEM_INSTRUCTION without editing files (A/B/B-v2/C variants). `--enable-thinking` toggles Qwen3.5 thinking-mode via `chat_template_kwargs`. `--concurrency N` (default 8) runs parallel HTTP in-flight. Emits per-sample teacher-JSONL-compatible raw output for downstream `analyze_position_bias.py`. |
| `eval_metrics_vllm_offline.py` | In-process `vllm.LLM.chat. (no HTTP, no serve process) | HF path or local checkpoint dir | Same JSON shape as HTTP version | Same metrics as HTTP variant; offline = simpler deployment, one command loads the model and runs eval. Uses the same `--prompt-module` / `--enable-thinking` (default off) / raw JSONL / position bias machinery. Trade-off vs HTTP: offline blocks a whole GPU, can't share with training, one process per variant. |
| `eval_gguf.py` | llama-cpp-python (CUDA build) | GGUF Q4_K_M file | Same JSON shape | GGUF Q4_K_M only. Requires the CUDA-built `llama-cpp-python`, not the default CPU wheel. |

All three emit the same schema so `compare_results.py` can dashboard them.

## Latency

| Script | What it measures |
|---|---|
| `bench_latency.py` | ms/output-token (p50, p95), tok/s, e2e wall, output token mean ± std, fail rate. Across teacher Gemini API, student FP16, student W4A16. Used to produce the README latency table. |

## Inference-cache feeder (for the judge pipeline)

| Script | Output |
|---|---|
| `generate_inference_samples.py` | `data/inference_samples/<backend>.json` (one record per sample with prompt + raw output + parsed ranking + recovered business_ids). Spaced sample selection is deterministic via `pick_samples` so the judge and the cache always pick the same `sample_id`s. Backends: `teacher`, `fp16`, `w4a16`, `nf4`, `gguf`, plus vLLM variants. The consolidated `all_backends_merged.json` is what `judge/judge_listwise.py` reads by default. |

## Dashboard

| Script | Output |
|---|---|
| `compare_results.py` | `data/results/COMPARISON.md` (Markdown table for README), `comparison.csv` (Notion-importable), `comparison.html` (plotly Pareto scatter: size vs R@1, colored by method). Auto-discovers every `eval_*.json` + `latency_summary_*.json`. Includes an attribution section: ΔR@1 per stage (SFT, GKD-warm vs SFT, PTQ vs FP16 parent). |

## Conventions

- Tags follow `<phase>-<scale>-<format>`: e.g. `v3-9b-sft` or `v2-sft-w4a16`. The dashboard groups by tag prefix.
- Each eval JSON has top-level `tag`, `model_path`, `dataset`, `metrics`, `samples` (per-sample for residual analysis).
- Slot-stratified R@1 (residual position-bias audit) is on the wishlist — see the "Future work" section of the portfolio README "Add a slot-stratified R@1 column" item.

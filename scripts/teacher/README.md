# `scripts/teacher/` — Teacher generation, validation, debiasing

Everything that produces, validates, or audits the **teacher side** of the
distillation pipeline (Gemini API and local Qwen3.5-35B-A3B). Position-bias
analysis and counterfactual perturbations also live here because they all
operate on the same teacher JSONL contract.

## Generation

| Script | Backend | Output | Notes |
|---|---|---|---|
| `generate_teacher.py` | Gemini 3 Flash Preview API via `gemini_parallel` | `data/teacher/philly_teacher.jsonl` | T-3 baseline. 25 independent project keys (~500 RPD free tier). Resume-safe; `read_declared_key_names. defends against `gemini_parallel`'s `load_dotenv. env pollution. |
| `generate_teacher_qwen.py` | Local Qwen3.5-35B-A3B via vLLM **offline batch** | `data/teacher/philly_teacher_qwen35.jsonl` | T-1. TP=4, `StructuredOutputsParams(json=schema)`, `enforce_eager=True`. **This file is bias-contaminated** (, slot-1 = 20.25 %) — use HTTP path below for new training. Kept for v1/v2 reproducibility only. |
| `generate_teacher_permutation.py` | Local Qwen3.5-35B-A3B via vLLM **HTTP serve** | `data/teacher/philly_teacher_qwen35_<perm>.jsonl` | . Resume-safe, 8-wide concurrent. `--permutation {identity,reverse}`. Run twice (identity + reverse), then `merge_teacher_permutations.py`. |

## Aggregation

| Script | Input | Output |
|---|---|---|
| `merge_teacher_permutations.py` | Two `permutation` JSONLs (identity + reverse) | `data/teacher/philly_teacher_qwen35_borda_http.jsonl` (Borda-merged ranking + per-sample `permutation_consistency` block). **Production teacher dataset for v3+.** Slot-1 = 11.09 %, χ² p = 0.193 → uniform at α = 0.05. |

## Validation

| Script | Input | Output |
|---|---|---|
| `validate_teacher.py` | Teacher JSONL + source samples JSONL | Per-record verdicts (clean / `ranking_id_mismatch` / `rationale_id_mismatch` / etc.). `--rewrite` filters in-place; otherwise reports. Used at training-load time too (`train_student.load_and_filter`). |

## Position-bias audit (position-bias audit toolkit)

| Script | Output |
|---|---|
| `analyze_position_bias.py` | `data/results/position_bias*.json` — χ² goodness-of-fit over 10 slots for GT positive position, teacher top-1, per-GT-slot recall@1. |
| `visualize_position_bias.py` | 3 PNGs (10×10 deviation heatmaps, slot-1 across ranks, diagonal conflation profile) + `position_bias_full.json` 10×10 matrices + per-row χ². |

## Counterfactual perturbation (Stage 6.0.1 — judge sanity check)

| Script | Input | Output |
|---|---|---|
| `perturb_teacher_outputs.py` | Inference-cache JSON (the `teacher.json` from `eval/generate_inference_samples.py`) | Three perturbation tags: P1 ranking_shuffled, P2 rationale_swapped (derangement), P3 persona_replaced. Deterministic seeded by `sha256(kind:sample_id)`. Consumed directly by `judge/judge_listwise.py`. |

## Stress / load testing

| Script | What it does |
|---|---|
| `stress_teacher_vllm.py` | Replays GKD-like traffic (`prompt_logprobs=50`, long real prompts, sequential ×20 + concurrent ×8) against the running teacher serve. Used to validate the teacher logprob-workspace OOM fixes and to capture the 90.9 GB peak on PRO 6000. |

## Contract reminders

- **1-based `candidate_index`**, never `business_id`. See root README §1.2 for the schema-switch rationale.
- Pydantic schema in `configs/teacher_prompt.py`. `Literal[1..10]` + length 10 exactly. No `uniqueItems` (xgrammar).
- Always validate that the **inference path** matches between two teacher passes before attributing a "bias signature" to the model . Run identity through the same pipeline as the test condition.

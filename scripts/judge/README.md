# `scripts/judge/` — LLM-as-a-Judge (listwise, Stage 6)

Judges the **rationale + persona + ranking** quality of a model's recommendation
output against the same Yelp history shown to the model. **Rubric v3.1**
(`RUBRIC_VERSION = "v3.1"`): three independent 1-10 axes — Groundedness,
Personalization, Ranking Coherence — with quality gates at tiers 9-10 plus a
mandatory per-rationale verification scratchpad on the Groundedness axis (root
README §5.5). Gemini 3 Flash Preview via the `google-genai` SDK (paid-tier
`GOOGLE_API_KEY` for sequential runs **or** rotation across multiple
`GEMINI_API_KEY_*` keys for the parallel runner).

| Script | Reads | Writes | Notes |
|---|---|---|---|
| `judge_listwise.py` | `data/inference_samples/<backend>.json` (from `eval/generate_inference_samples.py` **or** `judge/adapt_raw_to_cache.py`) and/or perturbation caches (from `teacher/perturb_teacher_outputs.py`) | `data/results/judge_listwise_raw_v3.1.jsonl` (one verdict per sample × model) | Rubric v3.1: three mandatory VERIFICATION STEPs — Groundedness now begins with a **per-candidate scratchpad** (`Rationale #k: GT=[...] vs Cited=[...] -> Leak=[Yes/No]`) before counting, the **CAP** that locks G at 2 on any leak is hoisted ABOVE the Compute block, and `G_wrong` must list failing indices (`[2, 5]` or `[None]`) — bare integer is invalid. Personalization (`P_specific / P_nontrivial / R_link`; generic persona caps at 4) and Ranking Coherence (per-rationale tone `+/~/-`, `R_reverse`, `R_top3_positive`, `R_bottom3_negative`) unchanged. Resume-safe; only re-attempts records with `error != null`. New `--thinking-level` flag (default `MINIMAL` preserves v3-iter1 behaviour; production v3.1 runs use `MEDIUM`). `--max-output-tokens` default still 3072 — bump to 32768 with MEDIUM/HIGH so thinking does not crowd the JSON verdict. |
| `judge_listwise_parallel.py` | Same as above | `data/results/judge_listwise_raw_<RUBRIC>_parallel.jsonl` | ThreadPool over `gemini_parallel.AdvancedApiKeyManager` rotating across multiple `GEMINI_API_KEY_*` keys (from a parent `.env`; the project `.env` is loaded second without override). Each worker holds its own `GeminiSequentialProcessor` but they share one key manager so exhaustion / IP-ban detection stays coherent. Reuses prompt template + Pydantic schema + aggregator from `judge_listwise.py`. Used for the 50-sample × 3-thinking-level ablation matrix that resolved the P2×G blindspot. |
| `adapt_raw_to_cache.py` | `data/results/eval_<tag>_raw.jsonl` (from `eval/eval_metrics_vllm.py`) | `data/inference_samples/<tag>.json` | Converts streaming eval raw JSONL into the inference-cache JSON shape the judge consumes. Renames `raw_response` → `output_text`; re-derives `parsed_ranking` / `recovered_business_ids` / `json_parse_ok` via `eval.generate_inference_samples.summarize_output`. Lets the judge reuse the 287-record full eval split without re-running inference. |
| `analyze_judge_listwise.py` | `judge_listwise_raw_v3.1.jsonl` + matching inference caches | `judge_listwise_summary_v3.1.json` + `judge_listwise_report_v3.1.md` | Per-model bootstrap 95 % CI on all 3 axes; per-sample paired retrieval reconstruction (R@1 / R@5 / MRR from cache `recovered_business_ids`); point-biserial / Spearman for each axis × hit1/MRR pair; output-length descriptives (verbosity-bias pre-check, 0 extra calls). |
| `analyze_judge_validation.py` | Same raw JSONL but filters by perturbation tag (`P1` / `P2` / `P3`) | `judge_listwise_validation_v3.1.json` + `judge_listwise_validation_report_v3.1.md` | Paired Wilcoxon signed-rank (one-sided `less`) + bootstrap 95 % CI on mean Δ + discrimination rate over **all 3 axes**. Auto-verdict labels (`pass` / `weak` / `null` / `inverted` / `consistent` / `unexpected-drop`) against v3 expected-drop thresholds (10-scale: Δ ≤ −1.0 / −2.0 / etc.). |

## Validation gate (Stage 6.0.1, rubric v3.1 with MEDIUM thinking)

Expected verdict per (probe × axis):

| Probe | Groundedness | Personalization | Ranking Coherence |
|---|---|---|---|
| P1 ranking_shuffled | consistent | consistent | **pass** |
| P2 rationale_swapped | **pass** | **pass** | **pass** |
| P3 persona_replaced | consistent | **pass** | consistent |

**Observed v3.1 + ThinkingLevel.MEDIUM (parallel runner; P2 N=50, P1/P3 partial 42/50 due to free-tier daily quota cap)**:

| Probe | G Δ | P Δ | RC Δ |
|---|---:|---:|---:|
| P1 | −0.21 (drop 10/42) | −0.26 (drop 9/42) | **−5.62 pass** ✓ (disc 42/42 = 100 %) |
| P2 | **−6.43 pass** ✓ (drop 42/42 = 100 %) | −0.33 | **−4.24 pass** ✓ (drop 30/42) |
| P3 | −0.60 (drop 12/42) | **−5.33 pass** ✓ (drop 41/42 = 98 %) | −0.24 (drop 7/42) |

All three perturbation families now `pass` on their primary axis with no
regressions on secondary axes vs v3 iter1.

**Observed v3 iter1 (legacy baseline, MINIMAL, sequential, full N=50; retained for comparison)**:

| Probe | G Δ | P Δ | RC Δ |
|---|---:|---:|---:|
| P1 | −0.06 consistent | −0.18 noise | **−7.24 pass** ✓ (disc 98 %) |
| P2 | **+0.30 null ⚠️** | −0.18 weak ⚠️ | −1.84 pass |
| P3 | −0.02 consistent | **−1.70 pass** ✓ (disc 90 %) | +0.00 consistent |

The original v3-iter1 P2×G null was filed as a "Gemini Flash
limitation". The 2026-04-29 forensic deep-dive (root README §5.5) showed
this label was directionally correct (the failure is capacity-bound,
H5 under-thinking) but defeatist: a Gemini-Pro probe scored G=1 on
cases Flash MINIMAL gave G=10 *under the unchanged v3 prompt*, so the
prompt was executable with sufficient deliberation, and Flash with a
larger ThinkingLevel reaches the same verdict at a fraction of the
Pro cost. **v3.1's load-bearing change is `ThinkingLevel.MINIMAL →
MEDIUM`**; the three minor prompt edits ride along as scaffolding that
helps borderline budgets stay on track but do not carry the fix on
their own (v3.1 + LOW only moves Δ from +0.30 to −0.08).

## Rubric history (summary)

- **v1** (2026-04-15) — original 2-axis 1-5 rubric; failed P2-G and P3-P
  counterfactual cells (null Δ on both).
- **v2** (2026-04-15 evening) — added VERIFICATION STEPs 1 + 2; 6/6 probe
  cells pass but teacher saturates G = 4.94 / P = 5.00 on the 1-5 scale.
- **v3 iter0** (2026-04-22) — 3-axis 1-10 draft, cap = 4 for
  cross-candidate leak; P2 × G inverted (Δ = +0.44) because the gentle
  cap let subtle category/rating overlaps pass.
- **v3 iter1** (2026-04-22) — cap = 2 restored for cross-candidate leak;
  P1×RC and P3×P pass, P2×G null (Δ = +0.30, originally filed as Flash
  limitation).
- **v3.1** (2026-04-29, current) — three-edit prompt patch (E1: forced
  per-candidate scratchpad; E2: CAP hoisted above Compute; E3: G_wrong
  must list failing indices) plus ThinkingLevel raised MINIMAL → MEDIUM
  in production. Drives P2 × G from Δ = +0.30 to Δ = −6.80 with 0/50
  failures; no regressions on P1 or P3.

The root README §5.1 narrates the full iteration sequence; §5.4.3 covers
the iter0 → iter1 cap change; §5.5 details the v3.1 forensic and fix.

## Quota / cost

- **Production single-run** (paid `GOOGLE_API_KEY`, sequential):
  validation 150 + student 200 = ~350 calls, ~25 min at
  `--api-call-interval 0.5`.
- **Parallel ablation runs** (`judge_listwise_parallel.py`, multi-key
  rotation, free tier): the v3.1 fix matrix (P2 × {LOW, MEDIUM, HIGH} ×
  50) was 150 calls in ~45 min wall ($0). Free-tier daily quota caps
  successful calls before 429s start; resume on the same `--raw` path or
  fall back to the paid key for the tail.

## Bias-probe matrix (Stage 6.1, partially done)

- **Verbosity bias**: pre-checked observationally in Student Pass
  (`judge_listwise_report_v3.md` output-length table): v4-gkd-guided-B is
  longest AND has lowest Groundedness → rubric not rewarding verbosity.
  No extra Gemini calls consumed.
- **Position / self-enhancement / rubric order / score-ID bias**:
  probe scripts are **not yet written** — see the "Future work" section
  of the root README (§6.3 limitations).

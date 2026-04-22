# `scripts/judge/` — LLM-as-a-Judge (listwise, Stage 6)

Judges the **rationale + persona + ranking** quality of a model's recommendation
output against the same Yelp history shown to the model. **Rubric v3**
(`RUBRIC_VERSION = "v3"`): three independent 1-10 axes — Groundedness,
Personalization, Ranking Coherence — with quality gates at tiers 9-10.
Gemini 3 Flash Preview via the `google-genai` SDK (paid-tier
`GOOGLE_API_KEY` from `.env`).

| Script | Reads | Writes | Notes |
|---|---|---|---|
| `judge_listwise.py` | `data/inference_samples/<backend>.json` (from `eval/generate_inference_samples.py` **or** `judge/adapt_raw_to_cache.py`) and/or perturbation caches (from `teacher/perturb_teacher_outputs.py`) | `data/results/judge_listwise_raw_v3.jsonl` (one verdict per sample × model) | Rubric v3: three mandatory VERIFICATION STEPs — Groundedness (`G_exact / G_wrong / G_hallu / G_vague / avg_fields / G_attr_N` counts; any `G_wrong ≥ 1` caps at 2), Personalization (`P_specific / P_nontrivial / R_link`; generic persona caps at 4), Ranking Coherence (per-rationale tone `+/~/-`, `R_reverse`, `R_top3_positive`, `R_bottom3_negative`). Resume-safe; only re-attempts records with `error != null`. Backoff on 429/5xx. `--max-output-tokens` default = 3072. |
| `adapt_raw_to_cache.py` | `data/results/eval_<tag>_raw.jsonl` (from `eval/eval_metrics_vllm.py`) | `data/inference_samples/<tag>.json` | Converts streaming eval raw JSONL into the inference-cache JSON shape the judge consumes. Renames `raw_response` → `output_text`; re-derives `parsed_ranking` / `recovered_business_ids` / `json_parse_ok` via `eval.generate_inference_samples.summarize_output`. Lets the judge reuse the 287-record full eval split without re-running inference. |
| `analyze_judge_listwise.py` | `judge_listwise_raw_v3.jsonl` + matching inference caches | `judge_listwise_summary_v3.json` + `judge_listwise_report_v3.md` | Per-model bootstrap 95 % CI on all 3 axes; per-sample paired retrieval reconstruction (R@1 / R@5 / MRR from cache `recovered_business_ids`); point-biserial / Spearman for each axis × hit1/MRR pair; output-length descriptives (verbosity-bias pre-check, 0 extra calls). |
| `analyze_judge_validation.py` | Same raw JSONL but filters by perturbation tag (`P1` / `P2` / `P3`) | `judge_listwise_validation_v3.json` + `judge_listwise_validation_report_v3.md` | Paired Wilcoxon signed-rank (one-sided `less`) + bootstrap 95 % CI on mean Δ + discrimination rate over **all 3 axes**. Auto-verdict labels (`pass` / `weak` / `null` / `inverted` / `consistent` / `unexpected-drop`) against v3 expected-drop thresholds (10-scale: Δ ≤ −1.0 / −2.0 / etc.). |

## Validation gate (Stage 6.0.1, rubric v3 iter1)

Expected verdict per (probe × axis):

| Probe | Groundedness | Personalization | Ranking Coherence |
|---|---|---|---|
| P1 ranking_shuffled | consistent | consistent | **pass** |
| P2 rationale_swapped | **pass** | **pass** | **pass** |
| P3 persona_replaced | consistent | **pass** | consistent |

**Observed v3 iter1 (cap = 2, 200 Gemini calls)**:

| Probe | G Δ | P Δ | RC Δ |
|---|---:|---:|---:|
| P1 | −0.06 consistent ✓ | −0.18 unexpected-drop ⚠️ (noise) | **−7.24 pass** ✓ (disc 98 %) |
| P2 | +0.30 null ⚠️ | −0.18 weak ⚠️ | **−1.84 pass** ✓ |
| P3 | −0.02 consistent ✓ | **−1.70 pass** ✓ (disc 90 %) | +0.00 consistent ✓ |

**P2 × G/P weakness** is a Gemini 3 Flash limitation on semantically-similar
candidate swaps (it catches obvious name mismatches → G=4 cap, but misses
category/rating-only overlaps). RC is the universal back-stop; each
perturbation family hits ≥ 1 axis strongly. Documented as a known limitation
rather than a rubric defect.

## Rubric history (summary)

- **v1** (2026-04-15) — original 2-axis 1-5 rubric; failed P2-G and P3-P
  counterfactual cells (null Δ on both).
- **v2** (2026-04-15 evening) — added VERIFICATION STEPs 1 + 2; 6/6 probe
  cells pass but teacher saturates G = 4.94 / P = 5.00 on the 1-5 scale.
- **v3 iter0** (2026-04-22) — 3-axis 1-10 draft, cap = 4 for
  cross-candidate leak; P2 × G inverted (Δ = +0.44) because the gentle
  cap let subtle category/rating overlaps pass.
- **v3 iter1** (current) — cap = 2 restored for cross-candidate leak;
  improves P2 on average but hits the judge-model ceiling described
  above on truly subtle swaps.

Pre-release snapshots of earlier rubric versions are not included in the
public tree; the root README §5.1 narrates the full iteration sequence and
§5.4.3 describes the iter0 → iter1 cap change in detail.

## Quota / cost

Paid `GOOGLE_API_KEY` in `.env`. v3 iter1 run cost: 200 validation + 150
student = **350 calls** total. Wall ~25 min at `--api-call-interval 0.5`.

## Bias-probe matrix (Stage 6.1, partially done)

- **Verbosity bias**: pre-checked observationally in Student Pass
  (`judge_listwise_report_v3.md` output-length table): v4-gkd-guided-B is
  longest AND has lowest Groundedness → rubric not rewarding verbosity.
  No extra Gemini calls consumed.
- **Position / self-enhancement / rubric order / score-ID bias**:
  probe scripts are **not yet written** — see the "Future work" section
  of the root README (§6.3 limitations).

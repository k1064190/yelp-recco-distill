# Listwise LLM-as-a-Judge -- validation report

The baseline teacher pass saturated the 1-5 scale, so this run checks whether the judge is calibrated or merely stamping 5. Three controlled perturbations of the teacher output are judged under the same conditions; per-sample paired deltas tell us whether the judge actually reacts to quality defects.

## Baseline `teacher` (50 scored records)

| axis | mean | 95% CI |
|---|---:|---|
| groundedness | 4.94 | [4.82, 5.00] |
| personalization | 5.00 | [5.00, 5.00] |

## Per-perturbation paired deltas (perturbed − baseline)

`discrimination_rate` is the share of paired samples where the perturbed score is strictly less than the baseline score. A well-calibrated judge should drive this above 0.5 on the axes where the perturbation was designed to hurt. Wilcoxon alternative is `less` (one-sided test that Δ < 0).

| probe | axis | n_paired | mean Δ [95% CI] | discrimination | Wilcoxon | expected Δ ≤ | verdict |
|---|---|---:|---|---:|---|---:|---|
| `teacher-p1-ranking-shuffled` | groundedness | 50 | -0.02 [-0.06, +0.00] | 2% | (n_eff=1; fewer than 2 non-zero deltas) | +0.00 | **consistent** |
| `teacher-p1-ranking-shuffled` | personalization | 50 | -3.66 [-3.88, -3.40] | 98% | p=2.38e-11 (n_eff=49) | -0.50 | **pass** |
| `teacher-p2-rationale-swapped` | groundedness | 50 | -2.16 [-2.58, -1.72] | 66% | p=6.71e-08 (n_eff=33) | -1.00 | **pass** |
| `teacher-p2-rationale-swapped` | personalization | 50 | -2.08 [-2.50, -1.64] | 66% | p=5.77e-08 (n_eff=33) | -1.00 | **pass** |
| `teacher-p3-persona-replaced` | groundedness | 50 | +0.00 [+0.00, +0.00] | 0% | (n_eff=0; all deltas are zero (no signed ranks)) | +0.00 | **consistent** |
| `teacher-p3-persona-replaced` | personalization | 50 | -2.84 [-2.96, -2.68] | 100% | p=4.14e-12 (n_eff=50) | -0.50 | **pass** |

## Verdict legend

- **pass**: 95% CI lies entirely below 0 and mean Δ meets the expected-drop threshold for that axis.
- **weak**: 95% CI lies below 0 but mean Δ is smaller than the expected threshold (judge reacts, but weakly).
- **null**: 95% CI straddles 0 — perturbation produced no detectable change.
- **inverted**: perturbed score went UP (CI above 0) — judge is responding to the wrong signal.
- **consistent**: axis where no drop was expected, and no drop observed (CI contains 0 or above).
- **unexpected-drop**: axis where no drop was expected, but one showed up — warrants investigation.
- **insufficient**: too few paired samples to report.

## Interpretation guide

- If P2 (rationale_swapped) returns `pass` on both axes and P1 returns `pass` on Personalization, the judge is producing a quality signal on the teacher input distribution and the Stage 6.0 ceiling effect reflects genuine teacher quality rather than a broken judge.
- If P2 returns `null` on Groundedness, the judge is not reading the rationale<->candidate alignment we designed the rubric to check. That would be a real failure of the current prompt and would warrant rewriting the Groundedness rubric before judging students.
- If all three probes return `null`, the judge is uninformative at the teacher quality level and the Stage 6.0 result cannot be trusted.


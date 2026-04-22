# Listwise LLM-as-a-Judge -- validation report

The baseline teacher pass saturated the 1-5 scale, so this run checks whether the judge is calibrated or merely stamping 5. Three controlled perturbations of the teacher output are judged under the same conditions; per-sample paired deltas tell us whether the judge actually reacts to quality defects.

## Baseline `teacher` (50 scored records)

| axis | mean | 95% CI |
|---|---:|---|
| groundedness | 8.58 | [8.44, 8.72] |
| personalization | 9.28 | [9.16, 9.40] |
| ranking_coherence | 9.98 | [9.94, 10.00] |

## Per-perturbation paired deltas (perturbed − baseline)

`discrimination_rate` is the share of paired samples where the perturbed score is strictly less than the baseline score. A well-calibrated judge should drive this above 0.5 on the axes where the perturbation was designed to hurt. Wilcoxon alternative is `less` (one-sided test that Δ < 0).

| probe | axis | n_paired | mean Δ [95% CI] | discrimination | Wilcoxon | expected Δ ≤ | verdict |
|---|---|---:|---|---:|---|---:|---|
| `teacher-p1-ranking-shuffled` | groundedness | 50 | -0.04 [-0.16, +0.08] | 12% | p=0.377 (n_eff=10) | +0.00 | **consistent** |
| `teacher-p1-ranking-shuffled` | personalization | 50 | -0.20 [-0.32, -0.08] | 22% | p=0.00317 (n_eff=12) | +0.00 | **unexpected-drop** |
| `teacher-p1-ranking-shuffled` | ranking_coherence | 50 | -7.24 [-7.82, -6.62] | 98% | p=2.83e-10 (n_eff=49) | -1.50 | **pass** |
| `teacher-p2-rationale-swapped` | groundedness | 50 | +0.44 [+0.14, +0.72] | 14% | p=0.999 (n_eff=34) | -2.00 | **inverted** |
| `teacher-p2-rationale-swapped` | personalization | 50 | -0.12 [-0.26, +0.00] | 18% | p=0.073 (n_eff=12) | -2.00 | **null** |
| `teacher-p2-rationale-swapped` | ranking_coherence | 50 | -1.40 [-2.16, -0.70] | 24% | p=0.000244 (n_eff=13) | -1.00 | **pass** |
| `teacher-p3-persona-replaced` | groundedness | 50 | -0.04 [-0.14, +0.06] | 8% | p=0.344 (n_eff=6) | +0.00 | **consistent** |
| `teacher-p3-persona-replaced` | personalization | 50 | -2.08 [-2.54, -1.66] | 90% | p=1.71e-09 (n_eff=45) | -1.00 | **pass** |
| `teacher-p3-persona-replaced` | ranking_coherence | 50 | -0.02 [-0.06, +0.00] | 2% | (n_eff=1; fewer than 2 non-zero deltas) | +0.00 | **consistent** |

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


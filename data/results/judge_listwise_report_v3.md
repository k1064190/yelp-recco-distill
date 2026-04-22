# Listwise LLM-as-a-Judge -- summary report

Total verdicts on disk: **400** (includes errored attempts).

## Per-model judge scores (1-10 scale, bootstrap 95% CI)

| model | n_scored | n_errors | Groundedness | Personalization | Ranking Coherence |
|---|---:|---:|---|---|---|
| `teacher` | 50 | 0 | 8.54 [8.38, 8.70] | 9.32 [9.20, 9.44] | 9.96 [9.90, 10.00] |
| `teacher-p1-ranking-shuffled` | 50 | 0 | 8.48 [8.32, 8.64] | 9.14 [9.04, 9.26] | 2.72 [2.14, 3.36] |
| `teacher-p2-rationale-swapped` | 50 | 0 | 8.84 [8.46, 9.14] | 9.14 [9.04, 9.26] | 8.12 [7.24, 8.90] |
| `teacher-p3-persona-replaced` | 50 | 0 | 8.52 [8.36, 8.68] | 7.62 [7.20, 8.00] | 9.96 [9.90, 10.00] |
| `qwen35-teacher-guided-varB-example-v2` | 50 | 0 | 8.74 [8.56, 8.94] | 9.60 [9.46, 9.74] | 10.00 [10.00, 10.00] |
| `v4-sft-B-opt-vllm-guided` | 50 | 0 | 8.64 [8.46, 8.84] | 9.42 [9.28, 9.56] | 9.90 [9.70, 10.00] |
| `v4-gkd-guided-B-vllm-guided` | 50 | 0 | 8.28 [8.08, 8.46] | 9.28 [9.16, 9.40] | 9.98 [9.94, 10.00] |
| `base-0.8B-varB-vllm-guided` | 50 | 0 | 3.70 [3.06, 4.40] | 5.36 [4.94, 5.80] | 6.68 [5.98, 7.40] |

## Retrieval-metric agreement (per-sample paired with judge scores)

Sanity check: a calibrated judge should show **positive** correlation between its scores and the model's R@1 / MRR on the same sample. Negative or near-zero correlations suggest the judge is reacting to surface features (style, length) rather than recommendation quality -- decompose via the formal bias probes listed in Future work below.

| model | n_paired | R@1 | R@5 | MRR | Grnd vs R@1 | Pers vs R@1 | Rank vs R@1 | Grnd vs MRR | Pers vs MRR | Rank vs MRR |
|---|---:|---:|---:|---:|---|---|---|---|---|---|
| `teacher` | 0 | -- | -- | -- | -- | -- | -- | -- | -- | -- |
| `teacher-p1-ranking-shuffled` | 0 | -- | -- | -- | -- | -- | -- | -- | -- | -- |
| `teacher-p2-rationale-swapped` | 0 | -- | -- | -- | -- | -- | -- | -- | -- | -- |
| `teacher-p3-persona-replaced` | 0 | -- | -- | -- | -- | -- | -- | -- | -- | -- |
| `qwen35-teacher-guided-varB-example-v2` | 50 | 0.260 | 0.680 | 0.464 | +0.22 (p=0.117, n=50) | +0.02 (p=0.898, n=50) | -- | +0.25 (p=0.0832, n=50) | -0.00 (p=0.992, n=50) | -- |
| `v4-sft-B-opt-vllm-guided` | 50 | 0.240 | 0.720 | 0.440 | +0.09 (p=0.534, n=50) | -0.10 (p=0.495, n=50) | -0.25 (p=0.0748, n=50) | +0.07 (p=0.614, n=50) | -0.16 (p=0.269, n=50) | -0.19 (p=0.185, n=50) |
| `v4-gkd-guided-B-vllm-guided` | 50 | 0.300 | 0.700 | 0.462 | +0.23 (p=0.109, n=50) | +0.17 (p=0.224, n=50) | +0.09 (p=0.518, n=50) | +0.20 (p=0.166, n=50) | +0.03 (p=0.854, n=50) | +0.03 (p=0.862, n=50) |
| `base-0.8B-varB-vllm-guided` | 50 | 0.080 | 0.480 | 0.283 | +0.16 (p=0.277, n=50) | +0.26 (p=0.0697, n=50) | +0.21 (p=0.151, n=50) | +0.00 (p=0.982, n=50) | -0.02 (p=0.867, n=50) | +0.05 (p=0.754, n=50) |

## Output-length descriptives

Reported here for context only. A formal verbosity-bias test (length-controlled pair test, partial correlation) is scoped under Future work below.

| model | n | chars mean | chars p95 | tokens mean | tokens p95 |
|---|---:|---:|---:|---:|---:|
| `teacher` | 0 | -- | -- | -- | -- |
| `teacher-p1-ranking-shuffled` | 0 | -- | -- | -- | -- |
| `teacher-p2-rationale-swapped` | 0 | -- | -- | -- | -- |
| `teacher-p3-persona-replaced` | 0 | -- | -- | -- | -- |
| `qwen35-teacher-guided-varB-example-v2` | 50 | 2471 | 2720 | 589 | 638 |
| `v4-sft-B-opt-vllm-guided` | 50 | 2711 | 2976 | 685 | 739 |
| `v4-gkd-guided-B-vllm-guided` | 50 | 2807 | 2972 | 704 | 743 |
| `base-0.8B-varB-vllm-guided` | 50 | 2051 | 2408 | 514 | 581 |

---

## Future work (deferred bias probes)

- **Position bias** (pairwise-listwise hybrid swap test)
- **Verbosity bias** (length-controlled pair test + partial correlation)
- **Self-enhancement / preference leakage** (Gemini judge vs Gemini teacher vs Qwen3.5 teacher matched-sample test)
- **Rubric order bias** (criterion order reversal)
- **Score ID bias** (1-5 vs 1-10 vs Likert scale comparison)


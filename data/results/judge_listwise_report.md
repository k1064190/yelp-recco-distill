# Listwise LLM-as-a-Judge -- summary report

Total verdicts on disk: **50** (includes errored attempts).

## Per-model judge scores (1-5 scale, bootstrap 95% CI)

| model | n_scored | n_errors | Groundedness | Logicality & Personalization |
|---|---:|---:|---|---|
| `teacher` | 48 | 2 | 4.96 [4.90, 5.00] | 5.00 [5.00, 5.00] |

## Retrieval-metric agreement (per-sample paired with judge scores)

Sanity check: a calibrated judge should show **positive** correlation between its scores and the model's R@1 / MRR on the same sample. Negative or near-zero correlations suggest the judge is reacting to surface features (style, length) rather than recommendation quality -- decompose via the formal bias probes listed in Future work below.

| model | n_paired | R@1 | R@5 | MRR | Grnd vs R@1 | Pers vs R@1 | Grnd vs MRR (Spearman) | Pers vs MRR (Spearman) |
|---|---:|---:|---:|---:|---|---|---|---|
| `teacher` | 48 | 0.250 | 0.688 | 0.465 | +0.12 (p=0.415, n=48) | -- | +0.00 (p=0.979, n=48) | -- |

## Output-length descriptives

Reported here for context only. A formal verbosity-bias test (length-controlled pair test, partial correlation) is scoped under Future work below.

| model | n | chars mean | chars p95 | tokens mean | tokens p95 |
|---|---:|---:|---:|---:|---:|
| `teacher` | 48 | 2467 | 2888 | 606 | 739 |

---

## Future work (deferred bias probes)

- **Position bias** (pairwise-listwise hybrid swap test)
- **Verbosity bias** (length-controlled pair test + partial correlation)
- **Self-enhancement / preference leakage** (Gemini judge vs Gemini teacher vs Qwen3.5 teacher matched-sample test)
- **Rubric order bias** (criterion order reversal)
- **Score ID bias** (1-5 vs 1-10 vs Likert scale comparison)


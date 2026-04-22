# `scripts/data/` — Dataset preprocessing & migration

Anything that produces or transforms the **input side** of the SFT pipeline
(per-user samples, candidate sets) before the teacher is queried.

| Script | Input | Output | When |
|---|---|---|---|
| `preprocess_yelp.py` | Raw Yelp Open Dataset JSON files (business / review / user / tip / checkin) | `data/processed/philly_samples.jsonl` (history + candidates + positive target, one record per user) | T-3. `--max-history 20` cap derived from token-distribution profile; see ENV_VERSION § for the math. |
| `profile_token_distribution.py` | Teacher JSONL + Qwen3.5 tokenizer | `data/results/token_distribution.json` (p50/p95/p99/p99.9/max chat-template tokens) | Stage 2.5. Used to pick `--max-length` so 0 records are silently truncated at training time. |
| `migrate_teacher_format.py` | Legacy teacher JSONL (string `business_id` based) | New teacher JSONL (1-based `candidate_index`) | T-1. One-shot migration for the schema switch documented in root README §1.2 (Qwen3.5-35B-A3B couldn't faithfully copy 22-char hashes; switched contract to positional indices). Backs up the input before rewriting. |

## Notes

- `preprocess_yelp.py` is **deterministic** — same `--seed 42` reproduces the same
  3000 sample_ids regardless of the `--max-history` cap (T-3 verification).
- `migrate_teacher_format.py` uses `validate_teacher.py` (in `../teacher/`) to
  flag in-output duplicate indices that the schema-rewrite reveals.
- The 1-based candidate_index contract is enforced by the strict
  Pydantic schema at `configs/teacher_prompt.py` — `Literal[1..10]` plus
  `min_length=10, max_length=10`. `uniqueItems` is intentionally absent
  (xgrammar incompatible).

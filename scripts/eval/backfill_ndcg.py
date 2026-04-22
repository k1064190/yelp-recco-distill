#!/usr/bin/env python
# ABOUTME: Post-hoc NDCG@5/@10 backfill for eval_*.json that have _raw.jsonl sidecars.
# ABOUTME: CPU-only. Re-reads raw student rankings and re-runs metrics_against_positive.

"""
Backfill NDCG@5 and NDCG@10 into existing ``data/results/eval_<tag>.json``
files by re-running ``metrics_against_positive`` on the rankings stored in
the sidecar ``eval_<tag>_raw.jsonl``. No GPU, no model load — just I/O and
arithmetic.

Only works for tags that shipped a raw JSONL. Evals that ran via
``eval_metrics.py`` (HF transformers) without ``--raw-out`` are not
recoverable here; they would need a re-run.

Usage:
    $ python scripts/eval/backfill_ndcg.py            # update every pair found
    $ python scripts/eval/backfill_ndcg.py --dry-run  # report without writing
    $ python scripts/eval/backfill_ndcg.py --tag v4-sft-B-opt-vllm-guided
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.eval.eval_metrics import metrics_against_positive  # noqa: E402
from scripts.train.train_student import load_and_filter, split_examples  # noqa: E402

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("backfill_ndcg")


def _samples_by_id(samples_path: Path) -> dict:
    """Index processed samples by sample_id for candidate lookup."""
    out = {}
    with samples_path.open() as f:
        for line in f:
            s = json.loads(line)
            out[s["sample_id"]] = s
    return out


def _rank_indices_to_business_ids(
    ranking_indices: list[int] | None, candidates: list[dict]
) -> list[str]:
    """Convert 1-based ranking indices into the business_id sequence.

    Returns [] on invalid ranking (wrong length, not a permutation of 1..N).
    """
    if not isinstance(ranking_indices, list):
        return []
    n = len(candidates)
    if len(ranking_indices) != n:
        return []
    if not all(isinstance(x, int) for x in ranking_indices):
        return []
    if set(ranking_indices) != set(range(1, n + 1)):
        return []
    return [candidates[idx - 1]["business_id"] for idx in ranking_indices]


def backfill_one(eval_json: Path, samples_by_id: dict, dry_run: bool = False) -> dict:
    """Recompute NDCG for a single eval_<tag>.json using its _raw.jsonl.

    Returns a small summary dict with before/after metrics.
    """
    raw_path = eval_json.parent / f"{eval_json.stem}_raw.jsonl"
    if not raw_path.exists():
        return {"status": "skipped_no_raw", "json": str(eval_json)}

    eval_blob = json.loads(eval_json.read_text())
    pos = eval_blob.get("positive_metrics") or {}
    student_before = pos.get("student") or {}
    # If the student side already has ndcg@5, skip unless forced.
    if "ndcg@5" in student_before and "ndcg@10" in student_before:
        return {"status": "already_has_ndcg", "json": str(eval_json)}

    # Rebuild student rankings (business_ids) from raw JSONL.
    student_rankings: list[list[str]] = []
    positives: list[str] = []
    missing_samples = 0
    for line in raw_path.open():
        rec = json.loads(line)
        sid = rec.get("sample_id")
        sample = samples_by_id.get(sid)
        if sample is None:
            missing_samples += 1
            student_rankings.append([])
            positives.append(rec.get("positive_business_id") or "")
            continue
        cands = sample["candidates"]
        t_out = rec.get("teacher_output") or {}
        ranking_idx = t_out.get("ranking")
        business_id_ranking = _rank_indices_to_business_ids(ranking_idx, cands)
        student_rankings.append(business_id_ranking)
        positives.append(sample.get("positive_business_id") or rec.get("positive_business_id") or "")

    if missing_samples:
        log.warning(
            "%s: %d raw records had no matching sample in processed data (skipped)",
            eval_json.name, missing_samples,
        )

    # Recompute student side (new NDCG fields included via updated metrics func).
    student_new = metrics_against_positive(student_rankings, positives)

    # Teacher baseline: the JSON already stored it; just re-derive NDCG from
    # the teacher rankings. Easiest: read teacher_data field (if present)
    # else leave teacher as-is and only stamp NDCG on student.
    # In practice the teacher baseline is computed from the teacher JSONL at
    # run-time and we don't store that path in the JSON. So we skip the
    # teacher-side refresh; downstream compare_results.py already handles
    # missing NDCG by falling back to R@1/MRR.
    teacher_before = pos.get("teacher") or {}

    pos["student"] = {**student_before, **student_new}
    eval_blob["positive_metrics"] = pos

    before_summary = {
        k: round(student_before.get(k, 0), 4)
        for k in ("recall@1", "recall@5", "mrr@10")
    }
    after_summary = {
        k: round(student_new.get(k, 0), 4)
        for k in ("recall@1", "recall@5", "mrr@10", "ndcg@5", "ndcg@10")
    }

    if not dry_run:
        eval_json.write_text(json.dumps(eval_blob, indent=2, ensure_ascii=False))

    return {
        "status": "updated" if not dry_run else "dry_run",
        "json": str(eval_json),
        "before": before_summary,
        "after": after_summary,
    }


def main() -> int:
    """Process every ``eval_*.json`` with a matching ``_raw.jsonl`` sidecar."""
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--results-dir", type=Path,
                    default=PROJECT_ROOT / "data/results")
    p.add_argument("--samples", type=Path,
                    default=PROJECT_ROOT / "data/processed/philly_samples.jsonl")
    p.add_argument("--tag", type=str, default=None,
                    help="process only eval_<tag>.json")
    p.add_argument("--dry-run", action="store_true",
                    help="print what would change but don't write")
    args = p.parse_args()

    samples_by_id = _samples_by_id(args.samples)
    log.info("loaded %d processed samples", len(samples_by_id))

    if args.tag:
        targets = [args.results_dir / f"eval_{args.tag}.json"]
    else:
        targets = sorted(args.results_dir.glob("eval_*.json"))
        # skip sidecar files named eval_*_raw.jsonl (they're .jsonl not .json)
        targets = [t for t in targets if t.suffix == ".json"]

    n_updated = n_skipped = n_already = 0
    for target in targets:
        res = backfill_one(target, samples_by_id, dry_run=args.dry_run)
        status = res["status"]
        if status == "updated":
            n_updated += 1
            log.info(
                "%s: R@1 %.3f→%.3f | NDCG@5 %.3f | NDCG@10 %.3f",
                target.name,
                res["before"].get("recall@1", 0),
                res["after"].get("recall@1", 0),
                res["after"].get("ndcg@5", 0),
                res["after"].get("ndcg@10", 0),
            )
        elif status == "already_has_ndcg":
            n_already += 1
        elif status == "skipped_no_raw":
            n_skipped += 1
        elif status == "dry_run":
            log.info(
                "DRY %s: NDCG@5 %.3f NDCG@10 %.3f (would write)",
                target.name,
                res["after"].get("ndcg@5", 0),
                res["after"].get("ndcg@10", 0),
            )

    log.info(
        "done — updated=%d, already_had_ndcg=%d, skipped_no_raw=%d",
        n_updated, n_already, n_skipped,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())

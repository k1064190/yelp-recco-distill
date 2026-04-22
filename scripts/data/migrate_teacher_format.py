#!/usr/bin/env python
# ABOUTME: Migrate existing teacher JSONL from business_id-based format to
# ABOUTME: 1-based candidate_index format. Backs up input before rewriting.

"""
Migrate teacher JSONL files from the legacy business_id format to the
1-based candidate_index format.

Legacy format (what generate_teacher.py historically wrote):
    rationales: [{"business_id": "qcn...", "reason": "..."}, ...]
    ranking:    ["qcn...", "abc...", ...]

New format (what the updated pipeline expects):
    rationales: [{"candidate_index": 3, "reason": "..."}, ...]
    ranking:    [3, 1, 7, ...]

Per-record behaviour:

  * error is not None → passed through unchanged.
  * error is None AND teacher_output maps cleanly (every business_id in
    rationales + ranking appears exactly once in the source sample's
    candidate list) → teacher_output is replaced with the new-format
    equivalent and the record stays ok.
  * error is None but a business_id does not map → the record's error
    field is set to ``migration_failed:<reason>`` and teacher_output is
    left untouched. These are records that would have been rejected by
    validate_teacher.py anyway; a future teacher re-run can retry them
    because resume logic only counts ``error is None`` rows as done.
  * teacher_output already uses the new format (has candidate_index or
    int ranking) → passed through unchanged (idempotent re-run safe).

The source file is backed up with a ``.bak-<timestamp>`` suffix before
rewriting.

Example:
    $ python scripts/data/migrate_teacher_format.py \\
        --samples data/processed/philly_samples.jsonl \\
        --teacher data/teacher/philly_teacher.jsonl
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from pathlib import Path
from typing import Any, Iterator

# Project root so we can import scripts.*
PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.teacher.validate_teacher import load_samples_by_id  # noqa: E402

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("migrate_teacher_format")


# Canonical error strings emitted by migration failures. Kept distinct from
# validate_teacher.py error codes so downstream filtering can distinguish
# migration-dropped records from genuine generation failures.
ERR_MIGRATION_NO_SAMPLE = "migration_failed:no_sample"
ERR_MIGRATION_NO_CANDIDATES = "migration_failed:no_candidates"
ERR_MIGRATION_DUPLICATE_CAND = "migration_failed:duplicate_candidate_id"
ERR_MIGRATION_BAD_RATIONALE = "migration_failed:rationale_id_not_in_candidates"
ERR_MIGRATION_BAD_RANKING = "migration_failed:ranking_id_not_in_candidates"
ERR_MIGRATION_BAD_SHAPE = "migration_failed:teacher_output_shape"


def iter_jsonl(path: Path) -> Iterator[dict[str, Any]]:
    """Stream JSONL records from a file.

    Args:
        path (Path): path to a JSONL file.

    Yields:
        dict: one parsed record per line. Blank and unparseable lines are
            skipped with a warning so a partially corrupt file does not
            abort the whole migration.
    """
    with path.open("r", encoding="utf-8") as f:
        for lineno, raw in enumerate(f, start=1):
            line = raw.strip()
            if not line:
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError as e:
                log.warning("skipping unparseable line %d: %s", lineno, e)


def _is_already_new_format(teacher_output: dict[str, Any]) -> bool:
    """Detect whether a teacher_output already uses the new index format.

    A record is considered already-migrated if either:
      - any rationale entry has a ``candidate_index`` key, OR
      - the ranking list contains any int.

    This keeps the migration idempotent: running it twice is safe.

    Args:
        teacher_output (dict): candidate teacher_output payload.

    Returns:
        bool: True if already in new format (migration should skip).
    """
    rationales = teacher_output.get("rationales")
    if isinstance(rationales, list) and rationales:
        first = rationales[0]
        if isinstance(first, dict) and "candidate_index" in first:
            return True
    ranking = teacher_output.get("ranking")
    if isinstance(ranking, list) and ranking and isinstance(ranking[0], int):
        return True
    return False


def _build_id_to_index(
    candidates: list[dict[str, Any]],
) -> tuple[dict[str, int], str | None]:
    """Build a ``business_id -> 1-based index`` lookup from a candidate list.

    Args:
        candidates (list[dict]): ordered candidate entries from the source
            sample, each with a ``business_id`` key.

    Returns:
        tuple: (lookup_dict, error_string). ``lookup_dict`` is empty and
            ``error_string`` is non-None when the candidate list is empty
            or contains duplicate business_ids (which would make the
            mapping ambiguous).
    """
    if not candidates:
        return {}, ERR_MIGRATION_NO_CANDIDATES
    mapping: dict[str, int] = {}
    for i, c in enumerate(candidates, start=1):
        bid = c.get("business_id") if isinstance(c, dict) else None
        if not isinstance(bid, str) or not bid:
            return {}, ERR_MIGRATION_NO_CANDIDATES
        if bid in mapping:
            return {}, ERR_MIGRATION_DUPLICATE_CAND
        mapping[bid] = i
    return mapping, None


def migrate_teacher_output(
    teacher_output: dict[str, Any],
    candidates: list[dict[str, Any]],
) -> tuple[dict[str, Any] | None, str | None]:
    """Convert a legacy teacher_output to the new index-based format.

    Args:
        teacher_output (dict): legacy payload with keys persona (str),
            rationales (list of {business_id, reason}), ranking (list of
            business_id strings).
        candidates (list[dict]): ordered candidate list from the source
            sample. Each entry must have ``business_id``.

    Returns:
        tuple: (new_teacher_output, error). On success, new_teacher_output
            carries candidate_index + int-list ranking and error is None.
            On failure, new_teacher_output is None and error is a
            canonical ``migration_failed:*`` string.
    """
    if not isinstance(teacher_output, dict):
        return None, ERR_MIGRATION_BAD_SHAPE

    rationales = teacher_output.get("rationales")
    ranking = teacher_output.get("ranking")
    persona = teacher_output.get("persona")
    if not isinstance(rationales, list) or not isinstance(ranking, list):
        return None, ERR_MIGRATION_BAD_SHAPE

    lookup, err = _build_id_to_index(candidates)
    if err is not None:
        return None, err

    # Convert rationales: each business_id -> 1-based index.
    new_rationales: list[dict[str, Any]] = []
    for r in rationales:
        if not isinstance(r, dict):
            return None, ERR_MIGRATION_BAD_SHAPE
        bid = r.get("business_id")
        reason = r.get("reason")
        if not isinstance(bid, str) or bid not in lookup:
            return None, ERR_MIGRATION_BAD_RATIONALE
        new_rationales.append(
            {
                "candidate_index": lookup[bid],
                "reason": reason,
            }
        )

    # Convert ranking: list of business_ids -> list of 1-based indices.
    new_ranking: list[int] = []
    for bid in ranking:
        if not isinstance(bid, str) or bid not in lookup:
            return None, ERR_MIGRATION_BAD_RANKING
        new_ranking.append(lookup[bid])

    new_teacher_output = {
        "persona": persona,
        "rationales": new_rationales,
        "ranking": new_ranking,
    }
    return new_teacher_output, None


def migrate_record(
    rec: dict[str, Any],
    samples_by_id: dict[str, dict[str, Any]],
) -> tuple[dict[str, Any], str]:
    """Apply migration to one teacher record.

    Args:
        rec (dict): one teacher record as stored in the legacy file.
        samples_by_id (dict): output of ``load_samples_by_id`` used to look
            up the ordered candidate list for this record's sample_id.

    Returns:
        tuple: (updated_record, outcome). ``outcome`` is one of
            ``passthrough_error`` / ``already_new`` / ``migrated`` /
            ``migration_failed``. The updated_record is a shallow copy of
            ``rec`` with possibly-updated ``teacher_output`` and ``error``.
    """
    out = dict(rec)

    # Pre-existing errors are passed through untouched.
    if out.get("error") is not None:
        return out, "passthrough_error"

    teacher_output = out.get("teacher_output")
    if not isinstance(teacher_output, dict):
        out["error"] = ERR_MIGRATION_BAD_SHAPE
        return out, "migration_failed"

    # Idempotent: skip records already in the new format.
    if _is_already_new_format(teacher_output):
        return out, "already_new"

    sid = out.get("sample_id")
    sample = samples_by_id.get(sid) if sid else None
    if sample is None:
        out["error"] = ERR_MIGRATION_NO_SAMPLE
        return out, "migration_failed"

    candidates = sample.get("candidates") or []
    new_to, err = migrate_teacher_output(teacher_output, candidates)
    if err is not None:
        out["error"] = err
        return out, "migration_failed"

    out["teacher_output"] = new_to
    return out, "migrated"


def migrate_file(
    teacher_path: Path,
    samples_by_id: dict[str, dict[str, Any]],
) -> tuple[list[dict[str, Any]], dict[str, int]]:
    """Migrate every record in a teacher JSONL file.

    Args:
        teacher_path (Path): path to the teacher JSONL to migrate.
        samples_by_id (dict): output of ``load_samples_by_id``.

    Returns:
        tuple: (updated_records, stats). Records are returned in original
            file order. Stats counts each outcome bucket plus ``total``.
    """
    updated: list[dict[str, Any]] = []
    stats: dict[str, int] = {
        "total": 0,
        "passthrough_error": 0,
        "already_new": 0,
        "migrated": 0,
        "migration_failed": 0,
    }
    for rec in iter_jsonl(teacher_path):
        stats["total"] += 1
        new_rec, outcome = migrate_record(rec, samples_by_id)
        stats[outcome] = stats.get(outcome, 0) + 1
        updated.append(new_rec)
    return updated, stats


def rewrite_teacher_file(
    teacher_path: Path, updated: list[dict[str, Any]]
) -> Path:
    """Back up the teacher JSONL and write the migrated version in place.

    Args:
        teacher_path (Path): existing teacher file to replace.
        updated (list[dict]): records to write, in the desired output order.

    Returns:
        Path: path to the backup file that was created.
    """
    ts = time.strftime("%Y%m%dT%H%M%S")
    backup = teacher_path.with_suffix(teacher_path.suffix + f".bak-{ts}")
    teacher_path.rename(backup)
    with teacher_path.open("w", encoding="utf-8") as f:
        for rec in updated:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
    return backup


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments.

    Returns:
        argparse.Namespace: parsed arguments.
    """
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument(
        "--samples",
        type=Path,
        default=Path("data/processed/philly_samples.jsonl"),
        help="preprocessed samples JSONL (provides candidate order per sample)",
    )
    p.add_argument(
        "--teacher",
        type=Path,
        default=Path("data/teacher/philly_teacher.jsonl"),
        help="teacher output JSONL to migrate in place",
    )
    p.add_argument(
        "--dry-run",
        action="store_true",
        help="compute the migration plan and print stats, but do not write.",
    )
    return p.parse_args()


def main() -> int:
    args = parse_args()

    if not args.samples.exists():
        log.error("samples file not found: %s", args.samples)
        return 2
    if not args.teacher.exists():
        log.error("teacher file not found: %s", args.teacher)
        return 2

    log.info("loading samples from %s", args.samples)
    samples_by_id = load_samples_by_id(args.samples)
    log.info("loaded %d samples", len(samples_by_id))

    log.info("migrating teacher file %s", args.teacher)
    updated, stats = migrate_file(args.teacher, samples_by_id)

    log.info("=== migration summary ===")
    log.info("total records         : %d", stats["total"])
    log.info("passthrough (error)   : %d", stats["passthrough_error"])
    log.info("already new format    : %d", stats["already_new"])
    log.info("migrated successfully : %d", stats["migrated"])
    log.info("migration failed      : %d", stats["migration_failed"])

    if args.dry_run:
        log.info("dry-run mode; teacher file untouched")
        return 0

    if stats["migrated"] == 0 and stats["migration_failed"] == 0:
        log.info("nothing to change (all records already in new format or error)")
        return 0

    backup = rewrite_teacher_file(args.teacher, updated)
    log.info("rewrote %s (backup at %s)", args.teacher, backup)
    return 0


if __name__ == "__main__":
    sys.exit(main())

#!/usr/bin/env python
# ABOUTME: Post-hoc validator for Teacher outputs against their source samples.
# ABOUTME: Detects ranking/rationale business_id mismatches and missing fields.

"""
Validate a Teacher JSONL output file against its source samples JSONL.

The Teacher LLM is asked to return, for each input sample:
  - persona           : str
  - rationales        : list[{"candidate_index": int, "reason": str}]
  - ranking           : list[int]   (1-based candidate indices, best -> worst)

Because the Teacher is black-box, it can occasionally return well-formed JSON
that nevertheless disagrees with the candidate set — wrong indices, missing
candidates, or empty persona/rationales. Such records silently corrupt
downstream LoRA SFT if not caught, so this script joins teacher outputs with
their source samples on sample_id and enforces:

  1. teacher_output is a dict with keys persona / rationales / ranking
  2. persona is a non-empty string
  3. rationales is a list, each entry has candidate_index (int) + non-empty reason
  4. ranking is a list of integers
  5. set(rationale candidate_indices) == {1, 2, ..., N}
  6. set(ranking)                     == {1, 2, ..., N}
  7. len(ranking) == N (no duplicates)

Failure modes are reported individually and aggregated counts printed. With
``--rewrite`` the script creates a ``.bak-<timestamp>`` backup of the teacher
file and updates in place: each failing record's ``error`` field is set to a
specific reason string (e.g. ``ranking_mismatch``) so that the next run of
``generate_teacher.py`` will re-try those samples (its resume logic counts
only records whose ``error`` is ``None``).

Example:
    $ python scripts/teacher/validate_teacher.py \\
        --samples data/processed/philly_samples.jsonl \\
        --teacher data/teacher/philly_teacher.jsonl

    $ python scripts/teacher/validate_teacher.py \\
        --samples data/processed/philly_samples.jsonl \\
        --teacher data/teacher/philly_teacher.jsonl \\
        --rewrite
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from pathlib import Path
from typing import Any, Iterator

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("validate_teacher")


# ---------- JSONL helpers ----------


def iter_jsonl(path: Path) -> Iterator[dict[str, Any]]:
    """Stream JSONL records from a file.

    Args:
        path (Path): path to a JSONL file.

    Yields:
        dict: one parsed record per line. Blank lines and unparseable lines
            are skipped with a warning.
    """
    with path.open("r", encoding="utf-8") as f:
        for lineno, raw in enumerate(f, start=1):
            line = raw.strip()
            if not line:
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError as e:
                log.warning("skipping unparseable line %d in %s: %s", lineno, path, e)


def load_samples_by_id(path: Path) -> dict[str, dict[str, Any]]:
    """Load all preprocessed samples indexed by sample_id.

    Args:
        path (Path): samples.jsonl produced by preprocess_yelp.py.

    Returns:
        dict[str, dict]: mapping sample_id -> full sample record.
    """
    out: dict[str, dict[str, Any]] = {}
    for rec in iter_jsonl(path):
        sid = rec.get("sample_id")
        if sid:
            out[sid] = rec
    return out


# ---------- Per-record validation ----------


# Canonical error strings. Keep stable so downstream tools can filter on them.
ERR_NO_TEACHER_OUTPUT = "no_teacher_output"
ERR_MISSING_FIELD = "missing_field"
ERR_EMPTY_PERSONA = "empty_persona"
ERR_EMPTY_RATIONALE = "empty_rationale"
ERR_RATIONALE_MISMATCH = "rationale_id_mismatch"
ERR_RANKING_MISMATCH = "ranking_id_mismatch"
ERR_RANKING_DUPLICATE = "ranking_duplicate"
ERR_NO_SAMPLE = "sample_not_found"


def validate_record(
    teacher_rec: dict[str, Any],
    sample_rec: dict[str, Any] | None,
) -> str | None:
    """Check one teacher record against its source sample.

    Applies rules 1-7 described in the module docstring in order, returning
    the first failing rule's canonical error string. Returns None on success.

    Args:
        teacher_rec (dict): one record from the teacher JSONL file. Assumed
            to already have ``error is None`` (caller filters).
        sample_rec (dict | None): matching record from samples.jsonl, looked
            up by sample_id. None means the sample was not found at all.

    Returns:
        str | None: canonical error string (see ERR_* constants) on failure,
            or None if the record passes every check.
    """
    if sample_rec is None:
        return ERR_NO_SAMPLE

    teacher_out = teacher_rec.get("teacher_output")
    if not isinstance(teacher_out, dict):
        return ERR_NO_TEACHER_OUTPUT

    # Rule 1: required top-level keys
    for key in ("persona", "rationales", "ranking"):
        if key not in teacher_out:
            return f"{ERR_MISSING_FIELD}:{key}"

    # Rule 2: persona non-empty string
    persona = teacher_out["persona"]
    if not isinstance(persona, str) or not persona.strip():
        return ERR_EMPTY_PERSONA

    # Rule 3: rationales shape + non-empty reasons (now with candidate_index)
    rationales = teacher_out["rationales"]
    if not isinstance(rationales, list) or not rationales:
        return f"{ERR_MISSING_FIELD}:rationales"
    rationale_indices: list[int] = []
    for r in rationales:
        if not isinstance(r, dict):
            return f"{ERR_MISSING_FIELD}:rationale_entry"
        idx = r.get("candidate_index")
        reason = r.get("reason")
        if not isinstance(idx, int):
            return f"{ERR_MISSING_FIELD}:rationale.candidate_index"
        if not isinstance(reason, str) or not reason.strip():
            return ERR_EMPTY_RATIONALE
        rationale_indices.append(idx)

    # Rule 4: ranking shape (now list of ints)
    ranking = teacher_out["ranking"]
    if not isinstance(ranking, list) or not ranking:
        return f"{ERR_MISSING_FIELD}:ranking"
    if not all(isinstance(x, int) for x in ranking):
        return f"{ERR_MISSING_FIELD}:ranking.entry"

    # Build expected index set from candidate count.
    candidates = sample_rec.get("candidates") or []
    n_cands = len(candidates)
    expected_indices = set(range(1, n_cands + 1))

    # Rule 5: rationale indices must be exactly {1, ..., N}
    if set(rationale_indices) != expected_indices:
        return ERR_RATIONALE_MISMATCH

    # Rule 6: ranking indices must be exactly {1, ..., N}
    if set(ranking) != expected_indices:
        return ERR_RANKING_MISMATCH

    # Rule 7: ranking length must equal candidate count (no dupes)
    if len(ranking) != n_cands:
        return ERR_RANKING_DUPLICATE

    return None


# ---------- Top-level validation pass ----------


def validate_file(
    teacher_path: Path,
    samples_by_id: dict[str, dict[str, Any]],
) -> tuple[list[dict[str, Any]], dict[str, int]]:
    """Run validation over every record in a teacher JSONL file.

    Iterates teacher records in their original file order, runs ``validate_record``
    on each record whose ``error`` field is currently None, and attaches a new
    ``error`` value to any failing record. Records that were already failed
    (e.g. ``error="quota exhausted"`` from generate_teacher.py) are kept
    untouched — this validator only demotes previously-ok records, never
    promotes failed ones.

    Args:
        teacher_path (Path): path to teacher JSONL.
        samples_by_id (dict): output of ``load_samples_by_id``.

    Returns:
        tuple: (updated_records, stats). ``updated_records`` is a list of dicts
            in original file order with possibly-updated ``error`` fields.
            ``stats`` is a dict mapping status label -> count, always including
            the keys ``total``, ``ok``, ``preexisting_error`` and the per-
            error-string counters for any failures found.
    """
    updated: list[dict[str, Any]] = []
    stats: dict[str, int] = {
        "total": 0,
        "ok": 0,
        "preexisting_error": 0,
    }

    for rec in iter_jsonl(teacher_path):
        stats["total"] += 1

        if rec.get("error") is not None:
            stats["preexisting_error"] += 1
            updated.append(rec)
            continue

        sid = rec.get("sample_id")
        sample = samples_by_id.get(sid) if sid else None
        err = validate_record(rec, sample)

        if err is None:
            stats["ok"] += 1
        else:
            # Bucket per high-level error prefix for stats, but store the full
            # string (including ":field" detail) on the record itself.
            bucket = err.split(":", 1)[0]
            stats[bucket] = stats.get(bucket, 0) + 1
            rec = dict(rec)  # do not mutate caller's dict
            rec["error"] = err

        updated.append(rec)

    return updated, stats


def rewrite_teacher_file(
    teacher_path: Path, updated: list[dict[str, Any]]
) -> Path:
    """Back up the teacher JSONL file and write an updated version in place.

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


# ---------- CLI ----------


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--samples",
        type=Path,
        default=Path("data/processed/philly_samples.jsonl"),
        help="preprocessed samples JSONL",
    )
    p.add_argument(
        "--teacher",
        type=Path,
        default=Path("data/teacher/philly_teacher.jsonl"),
        help="teacher output JSONL to validate",
    )
    p.add_argument(
        "--rewrite",
        action="store_true",
        help=(
            "after validation, back up the teacher file and rewrite it in "
            "place with updated error fields. Without this flag the script "
            "only prints a report."
        ),
    )
    p.add_argument(
        "--max-examples",
        type=int,
        default=5,
        help="how many failing sample_ids to print per error bucket",
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

    log.info("validating teacher file %s", args.teacher)
    updated, stats = validate_file(args.teacher, samples_by_id)

    # Report.
    log.info("=== validation summary ===")
    log.info("total records         : %d", stats["total"])
    log.info("ok                    : %d", stats["ok"])
    log.info("pre-existing error    : %d", stats["preexisting_error"])
    new_failures = {
        k: v
        for k, v in stats.items()
        if k not in {"total", "ok", "preexisting_error"}
    }
    if new_failures:
        log.info("newly detected failures:")
        for bucket, count in sorted(new_failures.items()):
            log.info("  %-22s: %d", bucket, count)
    else:
        log.info("newly detected failures: none")

    # Print a few example failing sample_ids per bucket.
    if new_failures and args.max_examples > 0:
        examples: dict[str, list[str]] = {}
        for rec in updated:
            err = rec.get("error")
            if not err or err == "preexisting_error":
                continue
            # Re-derive bucket for examples.
            bucket = err.split(":", 1)[0]
            if bucket not in new_failures:
                continue
            lst = examples.setdefault(bucket, [])
            if len(lst) < args.max_examples:
                lst.append(rec.get("sample_id", "?"))
        if examples:
            log.info("example failing sample_ids:")
            for bucket, sids in examples.items():
                log.info("  %s: %s", bucket, ", ".join(sids))

    if args.rewrite and new_failures:
        backup = rewrite_teacher_file(args.teacher, updated)
        log.info("rewrote %s (backup at %s)", args.teacher, backup)
    elif args.rewrite:
        log.info("rewrite requested but nothing to update; teacher file untouched")

    # Exit non-zero if there are any failures (pre-existing OR new), so CI /
    # orchestration can gate on it.
    return 0 if stats["ok"] == stats["total"] - stats["preexisting_error"] else 1


if __name__ == "__main__":
    sys.exit(main())

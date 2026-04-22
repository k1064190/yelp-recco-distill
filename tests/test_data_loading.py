# ABOUTME: Integration tests for Teacher<->sample join coverage and required
# ABOUTME: fields. Gates SFT readiness — failing means do not start training.

"""
End-to-end data checks over the real ``data/processed`` and ``data/teacher``
files. These tests are intentionally integration-level (they touch the real
files) because the single highest-impact failure mode in this project is a
silent corruption between teacher outputs and their source samples — a bug
that unit-level mocks would miss.

Gate semantics:

* The teacher file must contain at least ``MIN_OK_FOR_SFT`` records whose
  ``error`` is ``None``. Falling below this threshold means do not start SFT.
* Every ok record must have a matching sample in the processed samples file.
* Every ok record must have a non-empty ``teacher_output`` dict with the
  three required top-level keys.

These are cheap, run in <1s, and fail loudly with a clear message.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SAMPLES_FILE = PROJECT_ROOT / "data/processed/philly_samples.jsonl"
TEACHER_FILE = PROJECT_ROOT / "data/teacher/philly_teacher.jsonl"

# Minimum number of ok teacher records that must exist before we are willing
# to start LoRA SFT. 200 is the smallest dataset where r=16 LoRA on Qwen3-4B
# reliably converges with 3 epochs at effective batch 8 (~75 gradient steps).
MIN_OK_FOR_SFT = 200


def _load_jsonl(path: Path) -> list[dict]:
    out: list[dict] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            out.append(json.loads(line))
    return out


@pytest.fixture(scope="module")
def samples():
    if not SAMPLES_FILE.exists():
        pytest.skip(f"samples file not found: {SAMPLES_FILE}")
    return _load_jsonl(SAMPLES_FILE)


@pytest.fixture(scope="module")
def teacher_records():
    if not TEACHER_FILE.exists():
        pytest.skip(f"teacher file not found: {TEACHER_FILE}")
    return _load_jsonl(TEACHER_FILE)


def test_samples_file_has_positive_size(samples):
    assert len(samples) > 0, "preprocessed samples file is empty"


def test_samples_have_required_fields(samples):
    required = {"sample_id", "user_id", "history", "candidates", "positive_business_id"}
    first = samples[0]
    missing = required - set(first.keys())
    assert not missing, f"sample is missing fields: {missing}"


def test_teacher_file_meets_sft_minimum(teacher_records):
    ok = [r for r in teacher_records if r.get("error") is None]
    assert len(ok) >= MIN_OK_FOR_SFT, (
        f"only {len(ok)} ok teacher records; need ≥{MIN_OK_FOR_SFT} "
        "before starting SFT. Keep generate_teacher.py running."
    )


def test_teacher_ok_records_have_teacher_output(teacher_records):
    for r in teacher_records:
        if r.get("error") is not None:
            continue
        to = r.get("teacher_output")
        assert isinstance(to, dict), (
            f"sample_id={r.get('sample_id')} has error=None but teacher_output is not a dict"
        )
        for key in ("persona", "rationales", "ranking"):
            assert key in to, (
                f"sample_id={r.get('sample_id')} teacher_output missing {key!r}"
            )


def test_join_coverage_meets_minimum(samples, teacher_records):
    """Every ok teacher record must have a matching processed sample."""
    sample_ids = {s["sample_id"] for s in samples}
    ok_ids = {r["sample_id"] for r in teacher_records if r.get("error") is None}
    orphans = ok_ids - sample_ids
    assert not orphans, (
        f"{len(orphans)} ok teacher records have no matching sample; "
        f"first few: {sorted(orphans)[:5]}"
    )
    joinable = ok_ids & sample_ids
    assert len(joinable) >= MIN_OK_FOR_SFT, (
        f"only {len(joinable)} joinable (teacher ∩ samples) records; "
        f"need ≥{MIN_OK_FOR_SFT} for SFT"
    )


def test_no_duplicate_sample_ids_in_teacher(teacher_records):
    sids = [r.get("sample_id") for r in teacher_records if r.get("sample_id")]
    assert len(sids) == len(set(sids)), (
        f"teacher file has {len(sids) - len(set(sids))} duplicate sample_ids — "
        "resume logic should have skipped these"
    )

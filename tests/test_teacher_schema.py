# ABOUTME: Unit tests for parse_teacher_response and validate_record logic,
# ABOUTME: covering JSON-fence stripping, malformed payloads, and candidate index mismatches.

"""
Schema-level tests for the Teacher output pipeline.

Two code paths are exercised:

1. ``scripts.teacher.generate_teacher.parse_teacher_response`` — the defensive JSON
   parser that strips Markdown code fences and catches decode errors so
   ``generate_teacher.py`` never crashes mid-run on a malformed Gemini reply.

2. ``scripts.teacher.validate_teacher.validate_record`` — the post-hoc validator that
   compares teacher ``ranking``/``rationales`` against the source sample's
   candidate set. A silent mismatch here would poison LoRA SFT, so these
   tests lock the failure modes as part of the build contract.
"""

from __future__ import annotations

import pytest

from configs.teacher_prompt import (
    N_CANDIDATES,
    build_gemini_response_schema_dict,
)
from scripts.teacher.generate_teacher import (
    coerce_indices_to_int,
    parse_teacher_response,
)
from scripts.teacher.validate_teacher import (
    ERR_EMPTY_PERSONA,
    ERR_EMPTY_RATIONALE,
    ERR_MISSING_FIELD,
    ERR_NO_SAMPLE,
    ERR_NO_TEACHER_OUTPUT,
    ERR_RANKING_DUPLICATE,
    ERR_RANKING_MISMATCH,
    ERR_RATIONALE_MISMATCH,
    validate_record,
)


# ---------- parse_teacher_response ----------


def test_parse_plain_json_succeeds():
    raw = '{"persona": "p", "rationales": [], "ranking": []}'
    parsed, err = parse_teacher_response(raw)
    assert err is None
    assert parsed == {"persona": "p", "rationales": [], "ranking": []}


def test_parse_fenced_json_stripped():
    raw = '```json\n{"persona": "p", "rationales": [], "ranking": ["b1"]}\n```'
    parsed, err = parse_teacher_response(raw)
    assert err is None
    assert parsed is not None and parsed["ranking"] == ["b1"]


def test_parse_fenced_without_language_tag():
    raw = '```\n{"persona": "p", "rationales": [], "ranking": []}\n```'
    parsed, err = parse_teacher_response(raw)
    assert err is None
    assert parsed is not None


def test_parse_empty_returns_error():
    parsed, err = parse_teacher_response("")
    assert parsed is None
    assert err == "empty response"


def test_parse_invalid_json_returns_error():
    parsed, err = parse_teacher_response("{not valid json}")
    assert parsed is None
    assert err is not None and "json decode" in err.lower()


# ---------- validate_record ----------


def _make_sample(candidate_ids):
    """Build a minimal sample record with the given candidate business_ids."""
    return {
        "sample_id": "sX",
        "user_id": "uX",
        "city": "Philadelphia",
        "history": [],
        "candidates": [{"business_id": bid, "name": bid} for bid in candidate_ids],
        "positive_business_id": candidate_ids[0],
    }


def _make_teacher(ranking, rationale_indices=None, persona="the user likes food"):
    """Build a minimal teacher record. Defaults rationale indices to ranking."""
    if rationale_indices is None:
        rationale_indices = ranking
    return {
        "sample_id": "sX",
        "teacher_output": {
            "persona": persona,
            "rationales": [
                {"candidate_index": idx, "reason": f"reason for candidate {idx}"}
                for idx in rationale_indices
            ],
            "ranking": list(ranking),
        },
        "error": None,
    }


def test_validate_accepts_well_formed_record():
    sample = _make_sample(["b1", "b2", "b3"])
    teacher = _make_teacher([2, 1, 3])
    assert validate_record(teacher, sample) is None


def test_validate_rejects_missing_sample():
    teacher = _make_teacher([1])
    assert validate_record(teacher, None) == ERR_NO_SAMPLE


def test_validate_rejects_missing_teacher_output():
    sample = _make_sample(["b1"])
    teacher = {"sample_id": "sX", "teacher_output": None, "error": None}
    assert validate_record(teacher, sample) == ERR_NO_TEACHER_OUTPUT


def test_validate_rejects_empty_persona():
    sample = _make_sample(["b1"])
    teacher = _make_teacher([1], persona="   ")
    assert validate_record(teacher, sample) == ERR_EMPTY_PERSONA


def test_validate_rejects_empty_rationale_reason():
    sample = _make_sample(["b1"])
    teacher = _make_teacher([1])
    teacher["teacher_output"]["rationales"][0]["reason"] = ""
    assert validate_record(teacher, sample) == ERR_EMPTY_RATIONALE


def test_validate_rejects_missing_ranking_key():
    sample = _make_sample(["b1"])
    teacher = _make_teacher([1])
    del teacher["teacher_output"]["ranking"]
    err = validate_record(teacher, sample)
    assert err is not None and err.startswith(ERR_MISSING_FIELD)


def test_validate_rejects_ranking_mismatch():
    sample = _make_sample(["b1", "b2", "b3"])
    # Teacher used index 99 instead of 3
    teacher = _make_teacher([1, 2, 99], rationale_indices=[1, 2, 3])
    assert validate_record(teacher, sample) == ERR_RANKING_MISMATCH


def test_validate_rejects_rationale_mismatch():
    sample = _make_sample(["b1", "b2"])
    # Rationales cover index 1 + 3 (wrong), ranking is correct
    teacher = _make_teacher([1, 2], rationale_indices=[1, 3])
    assert validate_record(teacher, sample) == ERR_RATIONALE_MISMATCH


def test_validate_rejects_ranking_with_duplicate():
    sample = _make_sample(["b1", "b2"])
    teacher = _make_teacher([1, 1], rationale_indices=[1, 2])
    # Duplicate makes set(ranking) == {1} != {1, 2},
    # so the mismatch rule fires first. Either error is acceptable, as long
    # as the record is rejected.
    err = validate_record(teacher, sample)
    assert err in (ERR_RANKING_MISMATCH, ERR_RANKING_DUPLICATE)


# ---------- coerce_indices_to_int (Gemini string-enum bridge) ----------


def test_coerce_handles_none():
    assert coerce_indices_to_int(None) is None


def test_coerce_handles_non_dict():
    assert coerce_indices_to_int("not a dict") == "not a dict"  # type: ignore[arg-type]


def test_coerce_converts_string_indices_to_int():
    parsed = {
        "persona": "p",
        "rationales": [
            {"candidate_index": "1", "reason": "r1"},
            {"candidate_index": "10", "reason": "r10"},
        ],
        "ranking": ["3", "1", "2"],
    }
    out = coerce_indices_to_int(parsed)
    assert out is not None
    assert out["ranking"] == [3, 1, 2]
    assert out["rationales"][0]["candidate_index"] == 1
    assert out["rationales"][1]["candidate_index"] == 10


def test_coerce_passthrough_for_int_indices():
    parsed = {
        "rationales": [{"candidate_index": 5, "reason": "r"}],
        "ranking": [5, 1],
    }
    out = coerce_indices_to_int(parsed)
    assert out is not None
    assert out["ranking"] == [5, 1]
    assert out["rationales"][0]["candidate_index"] == 5


def test_coerce_passthrough_for_non_digit_strings():
    # Legacy fixture style with opaque string ids — must not be coerced.
    parsed = {
        "rationales": [{"candidate_index": "b1", "reason": "r"}],
        "ranking": ["b1", "b2"],
    }
    out = coerce_indices_to_int(parsed)
    assert out is not None
    assert out["ranking"] == ["b1", "b2"]
    assert out["rationales"][0]["candidate_index"] == "b1"


def test_coerce_idempotent():
    parsed = {
        "rationales": [{"candidate_index": "2", "reason": "r"}],
        "ranking": ["2", "1"],
    }
    once = coerce_indices_to_int(parsed)
    twice = coerce_indices_to_int(once)
    assert twice is not None
    assert twice["ranking"] == [2, 1]
    assert twice["rationales"][0]["candidate_index"] == 2


# ---------- build_gemini_response_schema_dict ----------


def test_gemini_schema_has_string_enums():
    """SDK 1.72.0+ rejects int enums; this dict must use string enums."""
    schema = build_gemini_response_schema_dict()
    expected_enum = [str(i) for i in range(1, N_CANDIDATES + 1)]
    rationale_item = schema["properties"]["rationales"]["items"]
    assert rationale_item["properties"]["candidate_index"]["enum"] == expected_enum
    assert schema["properties"]["ranking"]["items"]["enum"] == expected_enum


def test_gemini_schema_has_length_constraints():
    schema = build_gemini_response_schema_dict()
    assert schema["properties"]["rationales"]["minItems"] == N_CANDIDATES
    assert schema["properties"]["rationales"]["maxItems"] == N_CANDIDATES
    assert schema["properties"]["ranking"]["minItems"] == N_CANDIDATES
    assert schema["properties"]["ranking"]["maxItems"] == N_CANDIDATES


def test_gemini_schema_required_fields():
    schema = build_gemini_response_schema_dict()
    assert set(schema["required"]) == {"persona", "rationales", "ranking"}
    rationale_item = schema["properties"]["rationales"]["items"]
    assert set(rationale_item["required"]) == {"candidate_index", "reason"}


def test_gemini_schema_passes_genai_validation():
    """Hard regression test: the SDK's pydantic Schema must accept this dict."""
    pytest.importorskip("google.genai")
    from google.genai import types as genai_types

    schema = build_gemini_response_schema_dict()
    # If this raises ValidationError, the SDK changed enum typing again.
    genai_types.Schema.model_validate(schema)

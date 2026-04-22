# ABOUTME: Unit tests for training data format builders in scripts.train.train_student.
# ABOUTME: Ensures SFT prompt/completion boundary and index-based JSON serialization stay correct.

"""
Unit tests for the pure data-prep helpers in ``scripts.train.train_student``.

These functions are the "silent bug magnet" of the SFT pipeline — if the
prompt/completion boundary drifts, training will compute loss on the wrong
tokens and the student will learn to predict the user question instead of
the assistant answer, with no obvious runtime error. Lock the contract.
"""

from __future__ import annotations

import json

import pytest

from configs.teacher_prompt import SYSTEM_INSTRUCTION
from scripts.train.train_student import (
    _split_bucket,
    build_training_example,
    split_examples,
    teacher_output_to_assistant_text,
)


# ---------- Fixtures ----------


@pytest.fixture
def sample():
    """Minimal but realistic processed sample (2 history visits, 3 candidates)."""
    return {
        "sample_id": "s1",
        "user_id": "u1",
        "city": "Philadelphia",
        "history": [
            {
                "business_id": "h1",
                "name": "Pho Street",
                "categories": "Vietnamese, Noodles",
                "stars": 5,
                "review_snippet": "Best pho in town",
                "date": "2023-05-01",
            },
            {
                "business_id": "h2",
                "name": "Blue Bottle",
                "categories": "Coffee, Cafes",
                "stars": 4,
                "review_snippet": "Good espresso but crowded",
                "date": "2023-05-15",
            },
        ],
        "candidates": [
            {"business_id": "c1", "name": "Banh Mi Corner", "categories": "Vietnamese, Sandwiches", "avg_stars": 4.3, "attributes": {}},
            {"business_id": "c2", "name": "Stumptown Coffee", "categories": "Coffee, Cafes", "avg_stars": 4.5, "attributes": {}},
            {"business_id": "c3", "name": "Chipotle", "categories": "Mexican, Fast Food", "avg_stars": 3.2, "attributes": {}},
        ],
        "positive_business_id": "c1",
    }


@pytest.fixture
def teacher_record():
    """Teacher output that matches the sample fixture's candidate set exactly."""
    return {
        "sample_id": "s1",
        "error": None,
        "teacher_output": {
            "persona": "A Vietnamese food and specialty coffee enthusiast who enjoys casual cafes.",
            "rationales": [
                {"candidate_index": 1, "reason": "Direct Vietnamese match for user's pho history."},
                {"candidate_index": 2, "reason": "Specialty coffee aligns with Blue Bottle preference."},
                {"candidate_index": 3, "reason": "Fast food doesn't match user's higher-quality taste."},
            ],
            "ranking": [1, 2, 3],
        },
    }


# ---------- teacher_output_to_assistant_text ----------


def test_assistant_text_is_valid_json(teacher_record):
    text = teacher_output_to_assistant_text(teacher_record["teacher_output"])
    parsed = json.loads(text)
    assert parsed["persona"].startswith("A Vietnamese")
    assert [r["candidate_index"] for r in parsed["rationales"]] == [1, 2, 3]
    assert parsed["ranking"] == [1, 2, 3]


def test_assistant_text_preserves_non_ascii():
    to = {
        "persona": "Le persona français",
        "rationales": [{"candidate_index": 1, "reason": "très bon café"}],
        "ranking": [1],
    }
    text = teacher_output_to_assistant_text(to)
    # ensure_ascii=False — the literal accented character must survive
    assert "français" in text
    assert "très bon café" in text


def test_assistant_text_is_indented_for_readability():
    to = {"persona": "p", "rationales": [], "ranking": []}
    text = teacher_output_to_assistant_text(to)
    # indent=2 → newlines in the serialized output
    assert "\n" in text


# ---------- build_training_example ----------


def test_example_has_prompt_and_completion_keys(sample, teacher_record):
    ex = build_training_example(sample, teacher_record)
    assert set(ex.keys()) == {"prompt", "completion"}


def test_prompt_has_system_and_user_messages(sample, teacher_record):
    ex = build_training_example(sample, teacher_record)
    prompt = ex["prompt"]
    assert len(prompt) == 2
    assert prompt[0]["role"] == "system"
    assert prompt[0]["content"] == SYSTEM_INSTRUCTION
    assert prompt[1]["role"] == "user"
    assert "VISIT HISTORY" in prompt[1]["content"]
    assert "CANDIDATE PLACES" in prompt[1]["content"]


def test_user_prompt_contains_candidate_names(sample, teacher_record):
    ex = build_training_example(sample, teacher_record)
    user_content = ex["prompt"][1]["content"]
    for cand in sample["candidates"]:
        assert cand["name"] in user_content


def test_completion_is_single_assistant_message(sample, teacher_record):
    ex = build_training_example(sample, teacher_record)
    completion = ex["completion"]
    assert len(completion) == 1
    assert completion[0]["role"] == "assistant"


def test_completion_content_round_trips_teacher_output(sample, teacher_record):
    ex = build_training_example(sample, teacher_record)
    parsed = json.loads(ex["completion"][0]["content"])
    assert parsed == {
        k: teacher_record["teacher_output"][k]
        for k in ("persona", "rationales", "ranking")
    }


def test_prompt_does_not_leak_teacher_output(sample, teacher_record):
    """The prompt must never contain the persona/ranking — that is the target.

    If this test fails the student is effectively being given the answer in
    its input and LoRA SFT becomes degenerate.
    """
    ex = build_training_example(sample, teacher_record)
    prompt_text = (ex["prompt"][0]["content"] + "\n" + ex["prompt"][1]["content"])
    assert teacher_record["teacher_output"]["persona"] not in prompt_text
    # Rationales are free-text — we check that the distinctive phrase from
    # the fixture isn't present.
    assert "Direct Vietnamese match" not in prompt_text


# ---------- _split_bucket + split_examples ----------


def test_split_bucket_is_deterministic():
    """Same sample_id always maps to the same bucket."""
    a = _split_bucket("sample_abc", 0.9)
    b = _split_bucket("sample_abc", 0.9)
    assert a == b


def test_split_bucket_returns_valid_labels():
    for sid in ("s1", "s2", "s3", "s4", "s5"):
        assert _split_bucket(sid, 0.9) in {"train", "eval"}


def test_split_examples_ratio_approximately_90_10():
    examples = [{"sample_id": f"sample_{i}"} for i in range(1000)]
    train, ev = split_examples(examples, ratio=0.9)
    assert len(train) + len(ev) == 1000
    # ~90% train with hash-based splitting — allow generous ±3% tolerance
    train_frac = len(train) / 1000
    assert 0.87 <= train_frac <= 0.93, f"train fraction {train_frac:.3f} out of tolerance"


def test_split_examples_preserves_sample_ids():
    examples = [{"sample_id": f"s{i}"} for i in range(50)]
    train, ev = split_examples(examples, ratio=0.9)
    recovered = {e["sample_id"] for e in train} | {e["sample_id"] for e in ev}
    assert recovered == {f"s{i}" for i in range(50)}

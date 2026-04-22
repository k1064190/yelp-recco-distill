# ABOUTME: Unit tests for the pure metric functions in scripts/eval_metrics.py —
# ABOUTME: Recall@k, MRR, Kendall tau, teacher agreement, JSON parsing edges.

from __future__ import annotations

import math
from pathlib import Path
import sys

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.eval.eval_metrics import (  # noqa: E402
    extract_student_ranking,
    kendall_tau,
    metrics_against_positive,
    metrics_against_teacher,
    parse_json_ranking,
    _random_mrr10,
)


# ---------- metrics_against_positive ----------


class TestMetricsAgainstPositive:
    def test_all_top1(self):
        """Perfect top-1 across all samples gives max Recall and MRR."""
        rankings = [["a", "b", "c"], ["x", "y", "z"]]
        positives = ["a", "x"]
        m = metrics_against_positive(rankings, positives)
        assert m["recall@1"] == 1.0
        assert m["recall@5"] == 1.0
        assert m["recall@10"] == 1.0
        assert m["mrr@10"] == 1.0
        assert m["n_evaluated"] == 2

    def test_middle_rank(self):
        """Positive at rank 2 → Recall@1=0, Recall@5=1, MRR=0.5."""
        rankings = [["b", "a", "c"]]
        positives = ["a"]
        m = metrics_against_positive(rankings, positives)
        assert m["recall@1"] == 0.0
        assert m["recall@5"] == 1.0
        assert m["mrr@10"] == pytest.approx(0.5)

    def test_missing_positive(self):
        """Positive absent → 0 everywhere."""
        rankings = [["b", "c", "d"]]
        positives = ["a"]
        m = metrics_against_positive(rankings, positives)
        assert m["recall@1"] == 0.0
        assert m["recall@5"] == 0.0
        assert m["recall@10"] == 0.0
        assert m["mrr@10"] == 0.0

    def test_rank_10(self):
        """Positive at rank 10 → Recall@10=1, MRR=0.1."""
        rankings = [list("abcdefghij")]  # j is at index 9 → rank 10
        positives = ["j"]
        m = metrics_against_positive(rankings, positives)
        assert m["recall@1"] == 0.0
        assert m["recall@5"] == 0.0
        assert m["recall@10"] == 1.0
        assert m["mrr@10"] == pytest.approx(0.1)

    def test_empty_ranking_is_zero(self):
        """An empty ranking (e.g. from failed parse) contributes 0 to every metric."""
        rankings = [[], ["a", "b"]]
        positives = ["a", "a"]
        m = metrics_against_positive(rankings, positives)
        # sample 0: empty → 0
        # sample 1: rank 1 → 1
        assert m["recall@1"] == 0.5
        assert m["mrr@10"] == 0.5

    def test_length_mismatch_raises(self):
        with pytest.raises(ValueError):
            metrics_against_positive([["a"]], ["a", "b"])

    def test_empty_raises(self):
        with pytest.raises(ValueError):
            metrics_against_positive([], [])


# ---------- kendall_tau ----------


class TestKendallTau:
    def test_identical(self):
        assert kendall_tau(["a", "b", "c", "d"], ["a", "b", "c", "d"]) == pytest.approx(1.0)

    def test_reversed(self):
        assert kendall_tau(["a", "b", "c", "d"], ["d", "c", "b", "a"]) == pytest.approx(-1.0)

    def test_one_swap(self):
        # Swap rank of a,b in b: one inversion out of 6 pairs → tau = (5-1)/6 ≈ 0.667
        tau = kendall_tau(["a", "b", "c", "d"], ["b", "a", "c", "d"])
        assert tau == pytest.approx((5 - 1) / 6)

    def test_different_set_returns_none(self):
        assert kendall_tau(["a", "b"], ["x", "y"]) is None

    def test_too_short(self):
        assert kendall_tau(["a"], ["a"]) is None

    def test_empty(self):
        assert kendall_tau([], []) is None


# ---------- metrics_against_teacher ----------


class TestMetricsAgainstTeacher:
    def test_identical_rankings(self):
        student = [["a", "b", "c", "d"]]
        teacher = [["a", "b", "c", "d"]]
        m = metrics_against_teacher(student, teacher)
        assert m["top1_agreement"] == 1.0
        assert m["kendall_tau_mean"] == pytest.approx(1.0)
        assert m["kendall_tau_valid_n"] == 1

    def test_reversed_rankings(self):
        student = [["d", "c", "b", "a"]]
        teacher = [["a", "b", "c", "d"]]
        m = metrics_against_teacher(student, teacher)
        assert m["top1_agreement"] == 0.0
        assert m["kendall_tau_mean"] == pytest.approx(-1.0)
        assert m["kendall_tau_valid_n"] == 1

    def test_invalid_student_counts_as_top1_miss_and_tau_skip(self):
        """Empty student ranking is a miss for top-1 and excluded from tau average."""
        student = [[], ["a", "b"]]
        teacher = [["a", "b"], ["a", "b"]]
        m = metrics_against_teacher(student, teacher)
        # top-1: 0 (empty) + 1 (match) = 1 / 2 = 0.5
        assert m["top1_agreement"] == 0.5
        # tau: only second pair is valid, tau=1
        assert m["kendall_tau_valid_n"] == 1
        assert m["kendall_tau_mean"] == pytest.approx(1.0)

    def test_length_mismatch_raises(self):
        with pytest.raises(ValueError):
            metrics_against_teacher([["a"]], [["a"], ["b"]])

    def test_empty_lists(self):
        m = metrics_against_teacher([], [])
        assert m["top1_agreement"] == 0.0
        assert m["kendall_tau_valid_n"] == 0


# ---------- parse_json_ranking ----------


class TestParseJsonRanking:
    def test_plain_json(self):
        text = '{"persona": "foodie", "ranking": ["b1", "b2"]}'
        out = parse_json_ranking(text)
        assert out == {"persona": "foodie", "ranking": ["b1", "b2"]}

    def test_fenced_json(self):
        text = '```json\n{"ranking": ["a"]}\n```'
        out = parse_json_ranking(text)
        assert out is not None
        assert out["ranking"] == ["a"]

    def test_trailing_text(self):
        text = '{"ranking": ["a", "b"]}\nHere is my reasoning...'
        out = parse_json_ranking(text)
        assert out is not None
        assert out["ranking"] == ["a", "b"]

    def test_leading_text(self):
        text = 'Sure, here is the JSON: {"ranking": ["x"]}'
        out = parse_json_ranking(text)
        assert out is not None
        assert out["ranking"] == ["x"]

    def test_nested_objects(self):
        text = '{"rationales": [{"business_id": "b1", "reason": "good"}], "ranking": ["b1"]}'
        out = parse_json_ranking(text)
        assert out is not None
        assert out["ranking"] == ["b1"]

    def test_no_json(self):
        assert parse_json_ranking("just plain text with no braces") is None

    def test_truncated(self):
        """Unterminated JSON should not parse."""
        text = '{"ranking": ["a", "b", "c"'
        assert parse_json_ranking(text) is None

    def test_empty_string(self):
        assert parse_json_ranking("") is None


# ---------- extract_student_ranking ----------


class TestExtractStudentRanking:
    """Tests for index-based ranking extraction with index→business_id mapping."""

    def _cands(self, *names):
        """Build a candidate list with sequential business_ids."""
        return [{"business_id": f"b{i+1}", "name": n} for i, n in enumerate(names)]

    def test_valid_permutation(self):
        text = '{"persona": "x", "rationales": [], "ranking": [1, 2, 3]}'
        cands = self._cands("A", "B", "C")
        r = extract_student_ranking(text, cands)
        assert r == ["b1", "b2", "b3"]

    def test_reordered_indices(self):
        text = '{"ranking": [3, 1, 2]}'
        cands = self._cands("A", "B", "C")
        r = extract_student_ranking(text, cands)
        assert r == ["b3", "b1", "b2"]

    def test_extra_element(self):
        text = '{"ranking": [1, 2, 3, 4]}'
        cands = self._cands("A", "B", "C")
        assert extract_student_ranking(text, cands) is None

    def test_missing_element(self):
        text = '{"ranking": [1, 2]}'
        cands = self._cands("A", "B", "C")
        assert extract_student_ranking(text, cands) is None

    def test_out_of_range_index(self):
        text = '{"ranking": [1, 2, 99]}'
        cands = self._cands("A", "B", "C")
        assert extract_student_ranking(text, cands) is None

    def test_duplicate_in_ranking(self):
        text = '{"ranking": [1, 1, 3]}'
        cands = self._cands("A", "B", "C")
        assert extract_student_ranking(text, cands) is None

    def test_missing_ranking_key(self):
        text = '{"persona": "x"}'
        cands = self._cands("A")
        assert extract_student_ranking(text, cands) is None

    def test_ranking_not_list(self):
        text = '{"ranking": 1}'
        cands = self._cands("A")
        assert extract_student_ranking(text, cands) is None

    def test_ranking_with_non_int(self):
        text = '{"ranking": [1, "two", 3]}'
        cands = self._cands("A", "B", "C")
        assert extract_student_ranking(text, cands) is None

    def test_no_json(self):
        assert extract_student_ranking("just text", [{"business_id": "b1"}]) is None


# ---------- random baseline ----------


class TestRandomMrr10:
    def test_matches_harmonic_sum(self):
        """Random MRR@10 = (1 + 1/2 + ... + 1/10) / 10 ≈ 0.2929."""
        expected = sum(1 / r for r in range(1, 11)) / 10
        assert _random_mrr10() == pytest.approx(expected)
        assert 0.29 < _random_mrr10() < 0.30

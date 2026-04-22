# ABOUTME: Unit tests for the permutation/merge logic used in Option-1
# ABOUTME: (PRP) position-bias debiasing of the teacher dataset.

"""Tests for ``scripts.teacher.generate_teacher_permutation`` and
``scripts.teacher.merge_teacher_permutations``.

These tests cover the pure-Python pieces only (permutation math, Borda
count, rank translation) — HTTP calls to the teacher serve are not
exercised here.
"""

from __future__ import annotations

import pytest

from scripts.teacher.generate_teacher_permutation import (
    apply_permutation,
    make_permutation,
)
from scripts.teacher.merge_teacher_permutations import (
    borda_merge,
    is_strict_valid_ranking,
    kendall_tau,
    ranking_to_rank_vector,
    translate_pass2,
)


# ---------- permutation construction & application ----------


def test_make_permutation_identity() -> None:
    """Identity permutation is 1..K in order."""
    assert make_permutation("identity") == list(range(1, 11))


def test_make_permutation_reverse() -> None:
    """Reverse permutation is K..1."""
    assert make_permutation("reverse") == list(range(10, 0, -1))


def test_make_permutation_unknown_raises() -> None:
    """Unknown permutation kind raises ``ValueError``."""
    with pytest.raises(ValueError):
        make_permutation("spooky")


def test_apply_permutation_reverses_candidates() -> None:
    """Reverse permutation produces candidates in reverse order."""
    sample = {
        "sample_id": "s1",
        "candidates": [{"id": i} for i in range(1, 11)],
    }
    perm = make_permutation("reverse")
    new = apply_permutation(sample, perm)
    assert [c["id"] for c in new["candidates"]] == list(range(10, 0, -1))
    # Original not mutated.
    assert [c["id"] for c in sample["candidates"]] == list(range(1, 11))


def test_apply_permutation_identity_no_change() -> None:
    """Identity permutation leaves candidates untouched (new list though)."""
    sample = {"sample_id": "s", "candidates": [{"id": i} for i in range(1, 11)]}
    new = apply_permutation(sample, make_permutation("identity"))
    assert new["candidates"] == sample["candidates"]
    assert new["candidates"] is not sample["candidates"]


def test_apply_permutation_length_mismatch_raises() -> None:
    """Passing a permutation of the wrong length raises ``ValueError``."""
    sample = {"sample_id": "s", "candidates": [{"id": 1}, {"id": 2}]}
    with pytest.raises(ValueError):
        apply_permutation(sample, [1, 2, 3])


# ---------- rank-space translations ----------


def test_ranking_to_rank_vector_roundtrip() -> None:
    """``ranking_to_rank_vector`` inverts position ↔ rank consistently."""
    ranking = [9, 4, 3, 7, 1, 2, 6, 5, 10, 8]
    ranks = ranking_to_rank_vector(ranking)
    # slot 9 is rank 0, slot 4 is rank 1, ...
    assert ranks[9 - 1] == 0
    assert ranks[4 - 1] == 1
    assert ranks[8 - 1] == 9


def test_translate_pass2_reverse_permutation() -> None:
    """With reverse permutation, prompt slot i becomes original slot 11-i."""
    perm = list(range(10, 0, -1))  # [10, 9, ..., 1]
    prompt_ranking = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
    # prompt slot 1 = original 10, prompt slot 2 = original 9, ...
    assert translate_pass2(prompt_ranking, perm) == list(range(10, 0, -1))


def test_translate_pass2_identity_permutation() -> None:
    """Identity permutation is a no-op."""
    perm = list(range(1, 11))
    ranking = [3, 1, 7, 5, 2, 9, 4, 10, 6, 8]
    assert translate_pass2(ranking, perm) == ranking


# ---------- Borda merge ----------


def test_borda_merge_identical_rankings_preserves_order() -> None:
    """Two identical rankings → merged ranking equal to both."""
    r = [9, 4, 3, 7, 1, 2, 6, 5, 10, 8]
    assert borda_merge(r, r) == r


def test_borda_merge_opposite_rankings_uses_pass1_tiebreak() -> None:
    """When every slot ties at Borda = K-1, pass 1 order wins (stable)."""
    r1 = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
    r2 = list(reversed(r1))
    # Every slot earns (K-1-pos1) + (K-1-pos2) = 2*(K-1) - (K-1) = K-1.
    merged = borda_merge(r1, r2)
    assert merged == r1


def test_borda_merge_partial_agreement_top1_stable() -> None:
    """If both rankings put slot S first, merged top-1 must be S."""
    r1 = [7, 1, 2, 3, 4, 5, 6, 8, 9, 10]
    r2 = [7, 9, 3, 4, 1, 5, 8, 2, 6, 10]
    merged = borda_merge(r1, r2)
    assert merged[0] == 7


def test_borda_merge_content_consistent_example() -> None:
    """Real example from the pilot: top-4 stable across permutations."""
    r1 = [9, 4, 3, 7, 1, 2, 6, 5, 10, 8]
    r2 = [9, 4, 3, 7, 10, 1, 6, 2, 8, 5]
    merged = borda_merge(r1, r2)
    # Both runs agree on the first four slots; merged preserves that.
    assert merged[:4] == [9, 4, 3, 7]


def test_borda_merge_output_is_permutation() -> None:
    """Merged output is always a valid K-permutation of {1..K}."""
    r1 = [5, 2, 7, 1, 9, 10, 4, 6, 8, 3]
    r2 = [3, 8, 6, 4, 10, 9, 1, 7, 2, 5]
    merged = borda_merge(r1, r2)
    assert sorted(merged) == list(range(1, 11))


# ---------- validation ----------


def test_is_strict_valid_ranking_accepts_perm() -> None:
    """Any length-K permutation of {1..K} is strict-valid."""
    assert is_strict_valid_ranking([9, 4, 3, 7, 1, 2, 6, 5, 10, 8])


def test_is_strict_valid_ranking_rejects_wrong_length() -> None:
    """Length != K is rejected."""
    assert not is_strict_valid_ranking([1, 2, 3])


def test_is_strict_valid_ranking_rejects_duplicates() -> None:
    """Duplicate slot ids are rejected."""
    assert not is_strict_valid_ranking([1, 1, 2, 3, 4, 5, 6, 7, 8, 9])


def test_is_strict_valid_ranking_rejects_out_of_range() -> None:
    """Slot ids outside [1, K] are rejected."""
    assert not is_strict_valid_ranking([0, 2, 3, 4, 5, 6, 7, 8, 9, 10])
    assert not is_strict_valid_ranking([11, 2, 3, 4, 5, 6, 7, 8, 9, 10])


def test_is_strict_valid_ranking_rejects_non_list() -> None:
    """Non-list inputs are rejected."""
    assert not is_strict_valid_ranking(None)
    assert not is_strict_valid_ranking("not a list")


# ---------- kendall tau sanity ----------


def test_kendall_tau_identical_is_plus_one() -> None:
    """τ of a sequence with itself is +1."""
    a = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
    assert kendall_tau(a, a) == pytest.approx(1.0)


def test_kendall_tau_reverse_is_minus_one() -> None:
    """τ of a sequence with its reverse is -1."""
    a = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
    b = list(reversed(a))
    assert kendall_tau(a, b) == pytest.approx(-1.0)

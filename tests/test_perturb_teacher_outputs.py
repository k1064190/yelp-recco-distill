# ABOUTME: Unit tests for scripts/perturb_teacher_outputs.py -- derangement
# ABOUTME: invariants, deterministic seeding, field preservation per kind.

from __future__ import annotations

import json
import random
import sys
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.teacher.perturb_teacher_outputs import (  # noqa: E402
    GENERIC_PERSONA,
    P1_TAG,
    P2_TAG,
    P3_TAG,
    PERTURBATIONS,
    _derangement,
    _recovered_bids_from_ranking,
    _rng_for_sample,
    _shuffled_non_identity,
    build_perturbed_cache,
    perturb_p1_ranking_shuffled,
    perturb_p2_rationale_swapped,
    perturb_p3_persona_replaced,
    write_all_perturbations,
)


# ---------- Fixtures ----------


def _teacher_output_fixture() -> dict:
    """Minimal teacher output with N=5 rationales + ranking."""
    return {
        "persona": "Real persona that captures user's preferences.",
        "rationales": [
            {"candidate_index": 1, "reason": "reason about candidate 1"},
            {"candidate_index": 2, "reason": "reason about candidate 2"},
            {"candidate_index": 3, "reason": "reason about candidate 3"},
            {"candidate_index": 4, "reason": "reason about candidate 4"},
            {"candidate_index": 5, "reason": "reason about candidate 5"},
        ],
        "ranking": [3, 1, 5, 2, 4],
    }


def _source_cache_fixture() -> dict:
    """Minimal source cache dict with 2 samples in per-backend shape."""
    out = _teacher_output_fixture()
    return {
        "backend": "teacher",
        "samples": [
            {
                "sample_id": "sid-A",
                "positive_business_id": "bid-A3",
                "output_text": json.dumps(out),
                "parsed_ranking": out["ranking"],
                "recovered_business_ids": ["bid-A3", "bid-A1", "bid-A5", "bid-A2", "bid-A4"],
                "json_parse_ok": True,
                "output_tokens": 200,
            },
            {
                "sample_id": "sid-B",
                "positive_business_id": "bid-B2",
                "output_text": json.dumps(out),
                "parsed_ranking": out["ranking"],
                "recovered_business_ids": ["bid-B3", "bid-B1", "bid-B5", "bid-B2", "bid-B4"],
                "json_parse_ok": True,
                "output_tokens": 200,
            },
        ],
    }


# ---------- RNG determinism ----------


class TestRNGSeeding:
    def test_same_sid_kind_yields_same_rng_stream(self):
        r1 = _rng_for_sample("sid-X", "p1")
        r2 = _rng_for_sample("sid-X", "p1")
        assert [r1.random() for _ in range(5)] == [r2.random() for _ in range(5)]

    def test_different_kind_yields_different_stream(self):
        r_p1 = _rng_for_sample("sid-X", "p1")
        r_p2 = _rng_for_sample("sid-X", "p2")
        assert [r_p1.random() for _ in range(5)] != [r_p2.random() for _ in range(5)]

    def test_different_sid_yields_different_stream(self):
        r_a = _rng_for_sample("sid-A", "p1")
        r_b = _rng_for_sample("sid-B", "p1")
        assert [r_a.random() for _ in range(5)] != [r_b.random() for _ in range(5)]


# ---------- Permutation primitives ----------


class TestShuffledNonIdentity:
    def test_returns_non_identity_for_typical_case(self):
        rng = random.Random(42)
        out = _shuffled_non_identity([1, 2, 3, 4, 5], rng)
        assert sorted(out) == [1, 2, 3, 4, 5]  # same multiset
        assert out != [1, 2, 3, 4, 5]  # not identity

    def test_length_one_returns_input(self):
        rng = random.Random(0)
        assert _shuffled_non_identity([7], rng) == [7]

    def test_length_zero_returns_empty(self):
        rng = random.Random(0)
        assert _shuffled_non_identity([], rng) == []

    def test_length_two_swaps(self):
        rng = random.Random(0)
        assert _shuffled_non_identity([1, 2], rng) == [2, 1]

    def test_does_not_mutate_input(self):
        rng = random.Random(0)
        src = [1, 2, 3, 4]
        _ = _shuffled_non_identity(src, rng)
        assert src == [1, 2, 3, 4]


class TestDerangement:
    @pytest.mark.parametrize("seed", [0, 1, 2, 3, 7, 42, 1337])
    def test_no_fixed_points(self, seed):
        rng = random.Random(seed)
        src = list(range(10))
        out = _derangement(src, rng)
        assert len(out) == len(src)
        assert sorted(out) == sorted(src)
        assert all(a != b for a, b in zip(out, src))

    def test_length_one_returns_input(self):
        rng = random.Random(0)
        assert _derangement([42], rng) == [42]

    def test_length_two_derangement_is_swap(self):
        rng = random.Random(0)
        assert _derangement([1, 2], rng) == [2, 1]

    def test_does_not_mutate_input(self):
        rng = random.Random(0)
        src = ["a", "b", "c", "d"]
        _ = _derangement(src, rng)
        assert src == ["a", "b", "c", "d"]


# ---------- Per-output perturbations ----------


class TestPerturbP1:
    def test_ranking_is_permuted_non_identity(self):
        out = _teacher_output_fixture()
        rng = _rng_for_sample("sid-X", "p1")
        perturbed = perturb_p1_ranking_shuffled(out, rng)
        assert sorted(perturbed["ranking"]) == sorted(out["ranking"])
        assert perturbed["ranking"] != out["ranking"]

    def test_persona_and_rationales_preserved(self):
        out = _teacher_output_fixture()
        rng = _rng_for_sample("sid-X", "p1")
        perturbed = perturb_p1_ranking_shuffled(out, rng)
        assert perturbed["persona"] == out["persona"]
        assert perturbed["rationales"] == out["rationales"]


class TestPerturbP2:
    def test_derangement_reassigns_every_reason(self):
        out = _teacher_output_fixture()
        rng = _rng_for_sample("sid-X", "p2")
        perturbed = perturb_p2_rationale_swapped(out, rng)
        old = {r["candidate_index"]: r["reason"] for r in out["rationales"]}
        new = {r["candidate_index"]: r["reason"] for r in perturbed["rationales"]}
        for idx in old:
            assert new[idx] != old[idx], \
                f"rationale #{idx} kept its original reason (fixed point in derangement)"
        # Multiset of reasons preserved.
        assert sorted(new.values()) == sorted(old.values())

    def test_candidate_indices_preserved(self):
        out = _teacher_output_fixture()
        rng = _rng_for_sample("sid-X", "p2")
        perturbed = perturb_p2_rationale_swapped(out, rng)
        assert [r["candidate_index"] for r in perturbed["rationales"]] == \
               sorted([r["candidate_index"] for r in out["rationales"]])

    def test_persona_and_ranking_preserved(self):
        out = _teacher_output_fixture()
        rng = _rng_for_sample("sid-X", "p2")
        perturbed = perturb_p2_rationale_swapped(out, rng)
        assert perturbed["persona"] == out["persona"]
        assert perturbed["ranking"] == out["ranking"]

    def test_single_rationale_degenerate_returns_input(self):
        out = {"persona": "p", "rationales": [{"candidate_index": 1, "reason": "r"}], "ranking": [1]}
        perturbed = perturb_p2_rationale_swapped(out, random.Random(0))
        # With only 1 rationale there is no valid derangement; return unchanged.
        assert perturbed["rationales"] == out["rationales"]


class TestPerturbP3:
    def test_persona_replaced_with_generic(self):
        out = _teacher_output_fixture()
        perturbed = perturb_p3_persona_replaced(out, random.Random(0))
        assert perturbed["persona"] == GENERIC_PERSONA

    def test_rationales_and_ranking_preserved(self):
        out = _teacher_output_fixture()
        perturbed = perturb_p3_persona_replaced(out, random.Random(0))
        assert perturbed["rationales"] == out["rationales"]
        assert perturbed["ranking"] == out["ranking"]


# ---------- Determinism across invocations ----------


class TestDeterminism:
    def test_p1_deterministic_given_sid(self):
        out = _teacher_output_fixture()
        r1 = perturb_p1_ranking_shuffled(out, _rng_for_sample("sid-A", "p1"))
        r2 = perturb_p1_ranking_shuffled(out, _rng_for_sample("sid-A", "p1"))
        assert r1 == r2

    def test_p2_deterministic_given_sid(self):
        out = _teacher_output_fixture()
        r1 = perturb_p2_rationale_swapped(out, _rng_for_sample("sid-A", "p2"))
        r2 = perturb_p2_rationale_swapped(out, _rng_for_sample("sid-A", "p2"))
        assert r1 == r2


# ---------- recovered_business_ids rebuild ----------


class TestRecoveredBidsRebuild:
    def test_maps_ranking_to_candidate_bids(self):
        cand_bids = ["bid-1", "bid-2", "bid-3", "bid-4"]
        ranking = [3, 1, 4, 2]
        assert _recovered_bids_from_ranking(ranking, cand_bids) == \
               ["bid-3", "bid-1", "bid-4", "bid-2"]

    def test_out_of_range_index_yields_none(self):
        cand_bids = ["a", "b"]
        ranking = [1, 5, 2]  # 5 is out of range
        out = _recovered_bids_from_ranking(ranking, cand_bids)
        assert out == ["a", None, "b"]


# ---------- build_perturbed_cache integration ----------


class TestBuildPerturbedCache:
    def test_all_three_kinds_shape(self):
        source = _source_cache_fixture()
        cand_by_sid = {
            "sid-A": ["bid-A1", "bid-A2", "bid-A3", "bid-A4", "bid-A5"],
            "sid-B": ["bid-B1", "bid-B2", "bid-B3", "bid-B4", "bid-B5"],
        }
        for kind in ("p1", "p2", "p3"):
            tag, fn = PERTURBATIONS[kind]
            doc = build_perturbed_cache(source, tag, kind, fn, cand_by_sid)
            assert doc["backend"] == tag
            assert len(doc["samples"]) == 2
            for s in doc["samples"]:
                assert s["json_parse_ok"] is True
                parsed = json.loads(s["output_text"])
                assert "persona" in parsed
                assert "rationales" in parsed
                assert "ranking" in parsed
                assert s["parsed_ranking"] == parsed["ranking"]
                # recovered_business_ids must map ranking via candidate_bids
                expected = [cand_by_sid[s["sample_id"]][i - 1] for i in s["parsed_ranking"]]
                assert s["recovered_business_ids"] == expected

    def test_p1_changes_ranking_only(self):
        source = _source_cache_fixture()
        cand_by_sid = {sid: ["x"] * 5 for sid in ("sid-A", "sid-B")}
        tag, fn = PERTURBATIONS["p1"]
        doc = build_perturbed_cache(source, tag, "p1", fn, cand_by_sid)
        for s_in, s_out in zip(source["samples"], doc["samples"]):
            orig = json.loads(s_in["output_text"])
            new = json.loads(s_out["output_text"])
            assert new["persona"] == orig["persona"]
            assert new["rationales"] == orig["rationales"]
            assert new["ranking"] != orig["ranking"]

    def test_p3_only_replaces_persona(self):
        source = _source_cache_fixture()
        cand_by_sid = {sid: ["x"] * 5 for sid in ("sid-A", "sid-B")}
        tag, fn = PERTURBATIONS["p3"]
        doc = build_perturbed_cache(source, tag, "p3", fn, cand_by_sid)
        for s_in, s_out in zip(source["samples"], doc["samples"]):
            orig = json.loads(s_in["output_text"])
            new = json.loads(s_out["output_text"])
            assert new["persona"] == GENERIC_PERSONA
            assert new["rationales"] == orig["rationales"]
            assert new["ranking"] == orig["ranking"]


# ---------- End-to-end write ----------


class TestWriteAllPerturbations:
    def test_writes_three_files(self, tmp_path):
        src_path = tmp_path / "teacher.json"
        src_path.write_text(json.dumps(_source_cache_fixture()))
        samples_path = tmp_path / "philly_samples.jsonl"
        samples_path.write_text("\n".join([
            json.dumps({"sample_id": "sid-A", "candidates": [
                {"business_id": f"bid-A{i}"} for i in range(1, 6)
            ]}),
            json.dumps({"sample_id": "sid-B", "candidates": [
                {"business_id": f"bid-B{i}"} for i in range(1, 6)
            ]}),
        ]))
        written = write_all_perturbations(
            src_path, samples_path, tmp_path, ["p1", "p2", "p3"],
        )
        assert set(written) == {"p1", "p2", "p3"}
        # Each file must have the right backend tag and 2 samples.
        expected_tags = {"p1": P1_TAG, "p2": P2_TAG, "p3": P3_TAG}
        for k, p in written.items():
            assert p.exists()
            doc = json.loads(p.read_text())
            assert doc["backend"] == expected_tags[k]
            assert len(doc["samples"]) == 2

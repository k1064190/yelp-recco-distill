# ABOUTME: Unit tests for scripts/analyze_judge_validation.py --
# ABOUTME: pairing, delta computation, Wilcoxon guard, verdict labelling.

from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.judge.analyze_judge_validation import (  # noqa: E402
    analyze_validation,
    compute_deltas,
    discrimination_rate,
    pair_verdicts_by_sample_id,
    probe_verdict,
    render_validation_report,
    wilcoxon_signed_rank,
)
from scripts.teacher.perturb_teacher_outputs import P1_TAG, P2_TAG, P3_TAG  # noqa: E402


# ---------- Helpers ----------


def _verdict(
    sid: str,
    tag: str,
    g: int | None,
    p: int | None,
    rc: int | None = 8,
    err: str | None = None,
) -> dict:
    return {
        "sample_id": sid,
        "model_tag": tag,
        "groundedness": g,
        "personalization": p,
        "ranking_coherence": rc,
        "groundedness_evidence": "e",
        "personalization_evidence": "e",
        "ranking_coherence_evidence": "e",
        "error": err,
    }


# ---------- pair_verdicts_by_sample_id ----------


class TestPairing:
    def test_only_clean_pairs_returned(self):
        verdicts = [
            _verdict("s1", "teacher", 5, 5),
            _verdict("s1", P1_TAG, 4, 3),
            _verdict("s2", "teacher", 5, 5),
            _verdict("s2", P1_TAG, None, None, err="quota"),
            _verdict("s3", "teacher", 4, 4),
            # no P1 entry for s3
        ]
        pairs = pair_verdicts_by_sample_id(verdicts, "teacher", P1_TAG)
        assert [sid for sid, _, _ in pairs] == ["s1"]

    def test_empty_when_baseline_missing(self):
        verdicts = [_verdict("s1", P1_TAG, 4, 3)]
        assert pair_verdicts_by_sample_id(verdicts, "teacher", P1_TAG) == []


# ---------- compute_deltas ----------


class TestComputeDeltas:
    def test_per_axis_subtraction(self):
        pairs = [
            ("s1", _verdict("s1", "teacher", 5, 5), _verdict("s1", P1_TAG, 3, 4)),
            ("s2", _verdict("s2", "teacher", 4, 5), _verdict("s2", P1_TAG, 4, 2)),
        ]
        assert compute_deltas(pairs, "groundedness") == [-2.0, 0.0]
        assert compute_deltas(pairs, "personalization") == [-1.0, -3.0]

    def test_skip_non_int_sides(self):
        # s1: teacher.g=None -> groundedness delta skipped;
        #     teacher.p=5, P1.p=4 -> personalization delta = -1.
        # s2: teacher.g=5, P1.g=3 -> groundedness delta = -2;
        #     P1.p=None -> personalization delta skipped.
        pairs = [
            ("s1", _verdict("s1", "teacher", None, 5), _verdict("s1", P1_TAG, 3, 4)),
            ("s2", _verdict("s2", "teacher", 5, 5), _verdict("s2", P1_TAG, 3, None)),
        ]
        assert compute_deltas(pairs, "groundedness") == [-2.0]
        assert compute_deltas(pairs, "personalization") == [-1.0]


# ---------- discrimination_rate ----------


class TestDiscrimination:
    def test_basic(self):
        assert discrimination_rate([-1.0, -2.0, 0.0, 1.0]) == 0.5

    def test_all_negative(self):
        assert discrimination_rate([-1, -2, -3]) == 1.0

    def test_empty_returns_nan(self):
        assert math.isnan(discrimination_rate([]))


# ---------- wilcoxon_signed_rank ----------


class TestWilcoxon:
    def test_all_zero_deltas_no_p(self):
        res = wilcoxon_signed_rank([0.0, 0.0, 0.0])
        assert res["p_value"] is None
        assert "zero" in (res["error"] or "")

    def test_single_nonzero_no_p(self):
        res = wilcoxon_signed_rank([0.0, -1.0, 0.0])
        assert res["p_value"] is None
        assert res["n_effective"] == 1

    def test_strong_drop_gives_small_p(self):
        # 20 observations of -1, with alternative='less', p should be tiny.
        res = wilcoxon_signed_rank([-1.0] * 20)
        assert res["p_value"] is not None
        assert res["p_value"] < 0.001
        assert res["n_effective"] == 20


# ---------- probe_verdict ----------


class TestProbeVerdict:
    @pytest.mark.parametrize(
        "mean,lo,hi,expected,result",
        [
            # Axis where a drop is expected (expected_drop = -0.5)
            (-1.5, -2.0, -1.0, -0.5, "pass"),      # strong drop below threshold
            (-0.3, -0.6, -0.1, -0.5, "weak"),      # drop but below threshold strength
            (-0.1, -0.4, +0.2, -0.5, "null"),      # CI contains 0
            (+0.3, +0.1, +0.5, -0.5, "inverted"),  # CI above 0
            # Axis where NO drop is expected (expected_drop = 0.0)
            (-0.1, -0.3, +0.1, 0.0, "consistent"),  # CI contains 0
            (-0.8, -1.2, -0.4, 0.0, "unexpected-drop"),  # CI entirely below 0
            (+0.2, -0.1, +0.5, 0.0, "consistent"),
        ],
    )
    def test_verdict_cases(self, mean, lo, hi, expected, result):
        assert probe_verdict(mean, lo, hi, expected) == result

    def test_nan_inputs_return_insufficient(self):
        assert probe_verdict(float("nan"), 0, 0, -0.5) == "insufficient"


# ---------- analyze_validation + render end-to-end ----------


class TestEndToEnd:
    def test_analyze_validation_smoke(self, tmp_path):
        raw = tmp_path / "raw.jsonl"
        records = []
        # Baseline: 10 samples scored 8/9/10 (representative teacher v3).
        for i in range(10):
            records.append(_verdict(f"s{i}", "teacher", g=8, p=9, rc=10))
        # P1 rank_shuffled: ranking_coherence drops sharply, others stable.
        for i in range(10):
            records.append(_verdict(f"s{i}", P1_TAG, g=8, p=9, rc=4))
        # P2 rationale_swapped: all three axes drop meaningfully.
        for i in range(10):
            records.append(_verdict(f"s{i}", P2_TAG, g=4, p=5, rc=6))
        # P3 missing entirely.
        raw.write_text("\n".join(json.dumps(r) for r in records))

        summary = analyze_validation(raw)
        # Baseline block
        assert summary["baseline"]["axes"]["groundedness"]["mean"] == pytest.approx(8.0)
        assert summary["baseline"]["axes"]["ranking_coherence"]["mean"] == pytest.approx(10.0)

        # P1 ranking_coherence: expected -1.5, observed -6.0 -> pass.
        p1_rc = summary["probes"][P1_TAG]["axes"]["ranking_coherence"]
        assert p1_rc["mean_delta"] == pytest.approx(-6.0)
        assert p1_rc["verdict"] == "pass"
        # P1 groundedness / personalization: expected 0, observed 0 -> consistent.
        p1_grnd = summary["probes"][P1_TAG]["axes"]["groundedness"]
        assert p1_grnd["mean_delta"] == pytest.approx(0.0)
        assert p1_grnd["verdict"] == "consistent"
        p1_pers = summary["probes"][P1_TAG]["axes"]["personalization"]
        assert p1_pers["mean_delta"] == pytest.approx(0.0)
        assert p1_pers["verdict"] == "consistent"

        # P2 groundedness: expected -2.0, observed -4.0 -> pass.
        p2_grnd = summary["probes"][P2_TAG]["axes"]["groundedness"]
        assert p2_grnd["mean_delta"] == pytest.approx(-4.0)
        assert p2_grnd["verdict"] == "pass"
        # P2 ranking_coherence: expected -1.0, observed -4.0 -> pass.
        p2_rc = summary["probes"][P2_TAG]["axes"]["ranking_coherence"]
        assert p2_rc["mean_delta"] == pytest.approx(-4.0)
        assert p2_rc["verdict"] == "pass"

        # P3 missing
        assert summary["probes"][P3_TAG]["missing"] is True

    def test_render_report_smoke(self, tmp_path):
        raw = tmp_path / "raw.jsonl"
        records = [
            _verdict("s1", "teacher", g=8, p=9, rc=10),
            _verdict("s1", P1_TAG, g=8, p=9, rc=4),
            _verdict("s2", "teacher", g=8, p=9, rc=10),
            _verdict("s2", P1_TAG, g=8, p=9, rc=5),
        ]
        raw.write_text("\n".join(json.dumps(r) for r in records))
        summary = analyze_validation(raw)
        md = render_validation_report(summary)
        assert "validation report" in md
        assert P1_TAG in md
        assert "pass" in md or "weak" in md or "null" in md
        # ranking_coherence row must appear.
        assert "ranking_coherence" in md

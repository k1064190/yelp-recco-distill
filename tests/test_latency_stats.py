# ABOUTME: Unit tests for percentile math and aggregation helpers in
# ABOUTME: scripts.eval.bench_latency — the numbers that end up in the README.

"""
Lock the arithmetic correctness of the latency benchmark's stats pipeline.

These are cheap unit tests (no vLLM, no Gemini, no GPU) that exercise the
pure functions ``summarize``, ``aggregate``, and ``render_markdown`` in
``scripts.eval.bench_latency``. They exist because a bug in the percentile code
would silently turn the final portfolio numbers into nonsense — the kind
of failure a reviewer catches and nobody else does.
"""

from __future__ import annotations

import math

import pytest

from scripts.eval.bench_latency import (
    BackendResult,
    aggregate,
    render_markdown,
    summarize,
)


# ---------- summarize ----------


def test_summarize_empty_returns_zeros():
    result = summarize([])
    assert result == {"mean": 0.0, "std": 0.0, "p50": 0.0, "p95": 0.0, "p99": 0.0}


def test_summarize_single_value():
    result = summarize([42.0])
    assert result["mean"] == pytest.approx(42.0)
    assert result["std"] == 0.0  # population stdev of single element is 0
    assert result["p50"] == 42.0
    assert result["p95"] == 42.0
    assert result["p99"] == 42.0


def test_summarize_uniform_distribution():
    # 1..100 — p50 should be ~50, p95 ~95, p99 ~99
    values = list(range(1, 101))
    r = summarize([float(v) for v in values])
    assert r["mean"] == pytest.approx(50.5)
    # For the rounded-index percentile scheme used in summarize, p50 of
    # 1..100 lands on index round(0.5 * 99) = 50 which is the value 51.
    # Lock that behavior so an accidental change is caught.
    assert r["p50"] == 51
    assert r["p95"] == 95 or r["p95"] == 96
    assert r["p99"] == 99 or r["p99"] == 100


def test_summarize_preserves_order_independence():
    forward = [1.0, 2.0, 3.0, 4.0, 5.0]
    reverse = list(reversed(forward))
    assert summarize(forward) == summarize(reverse)


def test_summarize_std_is_population_stdev():
    # pstdev of [2, 4, 4, 4, 5, 5, 7, 9] is 2.0 exactly.
    r = summarize([2.0, 4.0, 4.0, 4.0, 5.0, 5.0, 7.0, 9.0])
    assert r["std"] == pytest.approx(2.0)


# ---------- aggregate ----------


def _row(backend, sid, latency, tokens, success=True, run_id=0, error=""):
    return BackendResult(
        backend=backend,
        sample_id=sid,
        run_id=run_id,
        latency_ms=latency,
        output_tokens=tokens,
        success=success,
        error=error,
    )


def test_aggregate_splits_by_backend():
    rows = [
        _row("a", "s1", 100.0, 50),
        _row("a", "s2", 200.0, 100),
        _row("b", "s1", 50.0, 25),
        _row("b", "s2", 150.0, 75),
    ]
    summary = aggregate(rows)
    assert set(summary.keys()) == {"a", "b"}
    assert summary["a"]["count_ok"] == 2
    assert summary["b"]["count_ok"] == 2


def test_aggregate_ms_per_tok_is_latency_over_tokens():
    # Two calls on backend x: 1000ms/100tok = 10ms/tok, 2000ms/200tok = 10ms/tok
    rows = [
        _row("x", "s1", 1000.0, 100),
        _row("x", "s2", 2000.0, 200),
    ]
    summary = aggregate(rows)
    assert summary["x"]["ms_per_tok"]["p50"] == pytest.approx(10.0)
    assert summary["x"]["ms_per_tok"]["mean"] == pytest.approx(10.0)


def test_aggregate_failure_rate():
    rows = [
        _row("x", "s1", 100.0, 50, success=True),
        _row("x", "s2", 0.0, 0, success=False, error="timeout"),
        _row("x", "s3", 100.0, 50, success=True),
        _row("x", "s4", 0.0, 0, success=False, error="429"),
    ]
    summary = aggregate(rows)
    assert summary["x"]["count_ok"] == 2
    assert summary["x"]["count_fail"] == 2
    assert summary["x"]["failure_rate"] == pytest.approx(0.5)


def test_aggregate_excludes_failed_calls_from_latency_stats():
    # A failed call with a wildly different latency must not drag ms/tok stats.
    rows = [
        _row("x", "s1", 100.0, 50, success=True),   # 2 ms/tok
        _row("x", "s2", 100.0, 50, success=True),   # 2 ms/tok
        _row("x", "s3", 99999.0, 0, success=False), # failure: excluded
    ]
    summary = aggregate(rows)
    assert summary["x"]["ms_per_tok"]["p50"] == pytest.approx(2.0)
    assert summary["x"]["latency_ms"]["p50"] == pytest.approx(100.0)


def test_aggregate_handles_zero_token_completion_safely():
    # A successful call with 0 output tokens must not divide-by-zero.
    rows = [_row("x", "s1", 100.0, 0, success=True)]
    summary = aggregate(rows)
    # 100 ms / max(1, 0) = 100 ms/tok
    assert summary["x"]["ms_per_tok"]["p50"] == pytest.approx(100.0)


def test_aggregate_output_tokens_mean_and_std():
    rows = [
        _row("x", "s1", 100.0, 100),
        _row("x", "s2", 100.0, 200),
        _row("x", "s3", 100.0, 300),
    ]
    summary = aggregate(rows)
    assert summary["x"]["output_tokens"]["mean"] == pytest.approx(200.0)
    # Population stdev of [100, 200, 300] = sqrt(6666.67) ≈ 81.65
    assert summary["x"]["output_tokens"]["std"] == pytest.approx(
        math.sqrt(20000 / 3), rel=1e-3
    )


# ---------- render_markdown ----------


def test_render_markdown_has_header_and_one_row_per_backend():
    rows = [
        _row("teacher", "s1", 1000.0, 500),
        _row("student", "s1", 500.0, 500),
    ]
    summary = aggregate(rows)
    md = render_markdown(summary)
    # Header present
    assert "Backend" in md and "p50 e2e (ms)" in md and "p50 ms/tok" in md
    # Separator row present
    assert "|---|" in md
    # Both backends present
    assert "teacher" in md
    assert "student" in md
    # Each row ends with newline and has 7 fields + two outer pipes = 8 pipes
    for backend in ("teacher", "student"):
        row_line = next(l for l in md.splitlines() if l.startswith(f"| {backend}"))
        assert row_line.count("|") == 8

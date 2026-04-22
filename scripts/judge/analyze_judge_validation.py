#!/usr/bin/env python
# ABOUTME: Paired-sample analysis: did the judge's scores drop on
# ABOUTME: controlled perturbations of the teacher output? (Sanity check.)

"""
Validate the listwise judge with controlled perturbations.

The judge's teacher-only pass saturated at Groundedness 4.96 /
Personalization 5.00, so we cannot tell from that run alone whether
the judge is calibrated or ceiling-stamping. ``perturb_teacher_outputs.py``
builds three counterfactual copies of the teacher output with targeted
defects. If the judge is calibrated, scores on the perturbed outputs
should drop in axis-appropriate ways:

  P1 ranking_shuffled      -> Personalization should drop; Groundedness
                              roughly unchanged (rationale text still
                              references real candidate fields).

  P2 rationale_swapped     -> Both axes should drop sharply; rationales
                              now cite the wrong candidate's fields.

  P3 persona_replaced      -> Personalization should drop; Groundedness
                              roughly unchanged.

This script loads ``judge_listwise_raw.jsonl`` after both the baseline
(``teacher``) and the three perturbed tags have been judged, pairs
verdicts by ``sample_id``, and reports per-(perturbation, axis) paired
statistics.

Outputs:
    - data/results/judge_listwise_validation.json
    - data/results/judge_listwise_validation_report.md

Example::

    $ python scripts/judge/analyze_judge_validation.py \\
        --raw data/results/judge_listwise_raw.jsonl \\
        --summary data/results/judge_listwise_validation.json \\
        --report data/results/judge_listwise_validation_report.md
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import sys
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.judge.judge_listwise import load_done_keys  # noqa: F401  (re-exported for tests)
from scripts.judge.analyze_judge_listwise import load_verdicts  # noqa: E402
from scripts.teacher.perturb_teacher_outputs import P1_TAG, P2_TAG, P3_TAG  # noqa: E402

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("analyze_judge_validation")


BASELINE_TAG = "teacher"
PERTURBATION_TAGS = (P1_TAG, P2_TAG, P3_TAG)
AXES = ("groundedness", "personalization", "ranking_coherence")


# ---------- Expected-drop thresholds (one source of truth) ----------


# Keyed (perturbation_tag, axis) -> minimum |mean_delta| we demand to
# call the judge "discriminative" on that probe. Calibrated for the v3
# rubric (1-10 scale, three independent axes). v2 thresholds were on a
# 1-5 scale with ranking_coherence folded into personalization.
#
# Rationale per cell:
#   P1 (ranking_shuffled)   -- only the ranking order is randomized.
#     groundedness:       rationale-candidate alignment unchanged -> consistent.
#     personalization:    persona/P_specific/R_link unchanged -> consistent.
#                         (v2 paid this in personalization because R_reverse
#                         lived there; v3 pulls it out.)
#     ranking_coherence:  random ranking destroys tone-rank correlation -> PASS.
#
#   P2 (rationale_swapped)  -- rationales deranged across candidates.
#     groundedness:       rationales now talk about wrong candidates -> PASS.
#     personalization:    R_link plummets (rationales no longer anchor to
#                         persona for the candidate they describe) -> PASS.
#     ranking_coherence:  swapped rationales randomise per-candidate tone,
#                         so ranking no longer tracks tone -> PASS (modest).
#
#   P3 (persona_replaced)   -- persona swapped for a generic template.
#     groundedness:       rationales still cite real fields -> consistent.
#     personalization:    P_specific=0 caps Personalization at 4 -> PASS.
#     ranking_coherence:  rationales + ranking unchanged -> consistent.
EXPECTED_DROP: dict[tuple[str, str], float] = {
    (P1_TAG, "groundedness"):       0.0,   # not expected to drop
    (P1_TAG, "personalization"):    0.0,   # migrated to ranking_coherence in v3
    (P1_TAG, "ranking_coherence"): -1.5,
    (P2_TAG, "groundedness"):      -2.0,
    (P2_TAG, "personalization"):   -2.0,
    (P2_TAG, "ranking_coherence"): -1.0,
    (P3_TAG, "groundedness"):       0.0,   # not expected to drop
    (P3_TAG, "personalization"):   -1.0,
    (P3_TAG, "ranking_coherence"):  0.0,   # not expected to drop
}


# ---------- Statistics ----------


def bootstrap_ci(
    values: list[float],
    n_resamples: int = 10000,
    alpha: float = 0.05,
    seed: int = 0,
) -> tuple[float, float, float]:
    """Bootstrap percentile CI for the mean of a 1-D numeric sample.

    Args:
        values (list[float]): observed Δ values (one per paired sample).
        n_resamples (int): bootstrap iterations.
        alpha (float): significance level; default 0.05 -> 95% CI.
        seed (int): RNG seed.

    Returns:
        tuple: (mean, lo, hi). (NaN, NaN, NaN) for empty input.
    """
    import numpy as np

    arr = np.asarray(values, dtype=float)
    if arr.size == 0:
        return float("nan"), float("nan"), float("nan")
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, arr.size, size=(n_resamples, arr.size))
    means = arr[idx].mean(axis=1)
    return (
        float(arr.mean()),
        float(np.quantile(means, alpha / 2)),
        float(np.quantile(means, 1 - alpha / 2)),
    )


def wilcoxon_signed_rank(
    deltas: list[float],
) -> dict[str, float | int | str | None]:
    """Run Wilcoxon signed-rank on paired deltas and package the result.

    Args:
        deltas (list[float]): one delta per paired sample (perturbed −
            baseline). Zero-diff entries are handled by scipy's default
            ``zero_method='wilcox'`` which drops them.

    Returns:
        dict: {"statistic", "p_value", "n_effective", "n_total",
            "method", "error" (if any)}. Degenerate inputs (all zeros
            or fewer than 2 non-zero entries) return a non-error record
            with p=None so the caller can still print a sensible row.
    """
    import numpy as np
    from scipy import stats

    n_total = len(deltas)
    arr = np.asarray(deltas, dtype=float)
    nonzero = arr[arr != 0.0]
    n_eff = int(nonzero.size)
    base = {
        "n_total": n_total,
        "n_effective": n_eff,
        "method": "wilcoxon-signed-rank",
        "error": None,
    }
    if n_eff == 0:
        return {**base, "statistic": 0.0, "p_value": None,
                "error": "all deltas are zero (no signed ranks)"}
    if n_eff < 2:
        return {**base, "statistic": float(nonzero.sum()), "p_value": None,
                "error": "fewer than 2 non-zero deltas"}
    try:
        res = stats.wilcoxon(
            nonzero,
            alternative="less",  # test that mean delta < 0 (score dropped)
            zero_method="wilcox",
        )
        return {**base, "statistic": float(res.statistic),
                "p_value": float(res.pvalue)}
    except ValueError as e:
        return {**base, "statistic": None, "p_value": None, "error": str(e)}


# ---------- Data shaping ----------


def pair_verdicts_by_sample_id(
    verdicts: list[dict[str, Any]],
    baseline_tag: str,
    perturbation_tag: str,
) -> list[tuple[str, dict[str, Any], dict[str, Any]]]:
    """Join baseline and perturbed verdicts on ``sample_id``.

    Only sample_ids with a clean (``error is None``) verdict under
    BOTH tags are returned. Errored records in either arm are dropped
    — we cannot form a paired sample without both sides.

    Args:
        verdicts (list[dict]): all rows from judge_listwise_raw.jsonl.
        baseline_tag (str): usually ``"teacher"``.
        perturbation_tag (str): ``P1_TAG`` / ``P2_TAG`` / ``P3_TAG``.

    Returns:
        list[tuple]: [(sample_id, baseline_rec, perturbed_rec), ...].
    """
    by_tag_sid: dict[tuple[str, str], dict[str, Any]] = {}
    for v in verdicts:
        if v.get("error") is not None:
            continue
        tag = v.get("model_tag")
        sid = v.get("sample_id")
        if tag and sid:
            by_tag_sid[(tag, sid)] = v

    out: list[tuple[str, dict[str, Any], dict[str, Any]]] = []
    for (tag, sid), rec in by_tag_sid.items():
        if tag != baseline_tag:
            continue
        pert = by_tag_sid.get((perturbation_tag, sid))
        if pert is None:
            continue
        out.append((sid, rec, pert))
    return out


def compute_deltas(
    pairs: list[tuple[str, dict[str, Any], dict[str, Any]]],
    axis: str,
) -> list[float]:
    """Per-sample Δ = perturbed − baseline on one axis.

    Args:
        pairs (list): output of ``pair_verdicts_by_sample_id``.
        axis (str): ``"groundedness"`` or ``"personalization"``.

    Returns:
        list[float]: one delta per pair, skipping pairs where either
            side has a non-int score.
    """
    out: list[float] = []
    for _sid, base, pert in pairs:
        b = base.get(axis)
        p = pert.get(axis)
        if not isinstance(b, int) or not isinstance(p, int):
            continue
        out.append(float(p - b))
    return out


def discrimination_rate(deltas: list[float]) -> float:
    """Share of pairs where the perturbed score is strictly below baseline.

    Args:
        deltas (list[float]): per-pair Δ.

    Returns:
        float: ``mean(Δ < 0)`` in [0, 1]; NaN if empty.
    """
    if not deltas:
        return float("nan")
    return sum(1 for d in deltas if d < 0) / len(deltas)


# ---------- Verdict on the probe ----------


def probe_verdict(
    mean_delta: float,
    ci_low: float,
    ci_high: float,
    expected_drop: float,
) -> str:
    """Map one probe's statistics onto a short pass/fail label.

    Rules of thumb:
      - If the 95 % CI lies entirely above 0 the perturbation made the
        judge score GO UP, which is nonsensical for a quality defect;
        label "inverted".
      - If the 95 % CI contains 0 the drop is not distinguishable from
        noise; label "null".
      - If the mean Δ beats the expected_drop threshold we call it a
        pass; otherwise "weak".
      - For axes where we do NOT expect a drop (expected_drop == 0 or
        positive), a CI containing 0 is actually the desired outcome;
        label "consistent" vs "unexpected drop".

    Args:
        mean_delta (float): observed mean Δ.
        ci_low (float): lower bound of 95% CI.
        ci_high (float): upper bound.
        expected_drop (float): signed threshold (typically ≤ 0).

    Returns:
        str: one of
            {"pass", "weak", "null", "inverted", "consistent",
             "unexpected-drop", "insufficient"}.
    """
    if math.isnan(mean_delta) or math.isnan(ci_low) or math.isnan(ci_high):
        return "insufficient"
    if expected_drop < 0.0:
        # Axis where we EXPECT a drop.
        if ci_low > 0:
            return "inverted"  # score went up
        if ci_high >= 0:
            return "null"  # not significantly different
        if mean_delta <= expected_drop:
            return "pass"
        return "weak"
    else:
        # Axis where we do NOT expect a drop.
        if ci_high < 0:
            return "unexpected-drop"
        return "consistent"


# ---------- Main analysis ----------


def analyze_validation(raw_path: Path) -> dict[str, Any]:
    """Produce the full validation summary from judge_listwise_raw.jsonl.

    Args:
        raw_path (Path): append-only verdict log.

    Returns:
        dict: {"baseline": {axis -> {n, mean, ...}},
               "probes": {tag -> {axis -> {...}}}}.
    """
    verdicts = load_verdicts(raw_path)
    by_tag: dict[str, list[dict[str, Any]]] = {}
    for v in verdicts:
        if v.get("error") is not None:
            continue
        by_tag.setdefault(v.get("model_tag") or "unknown", []).append(v)

    def _axis_stats(recs: list[dict[str, Any]], axis: str) -> dict[str, Any]:
        vals = [float(r[axis]) for r in recs if isinstance(r.get(axis), int)]
        mean, lo, hi = bootstrap_ci(vals)
        return {"n": len(vals), "mean": mean, "ci95_low": lo, "ci95_high": hi}

    baseline_recs = by_tag.get(BASELINE_TAG, [])
    baseline_block = {
        "n_records": len(baseline_recs),
        "axes": {axis: _axis_stats(baseline_recs, axis) for axis in AXES},
    }

    probes: dict[str, Any] = {}
    for tag in PERTURBATION_TAGS:
        if tag not in by_tag:
            probes[tag] = {"n_paired": 0, "missing": True}
            continue
        pairs = pair_verdicts_by_sample_id(verdicts, BASELINE_TAG, tag)
        per_axis: dict[str, Any] = {}
        for axis in AXES:
            deltas = compute_deltas(pairs, axis)
            mean, lo, hi = bootstrap_ci(deltas)
            wcx = wilcoxon_signed_rank(deltas)
            disc = discrimination_rate(deltas)
            expected = EXPECTED_DROP.get((tag, axis), 0.0)
            per_axis[axis] = {
                "n_paired": len(deltas),
                "mean_delta": mean,
                "delta_ci95_low": lo,
                "delta_ci95_high": hi,
                "perturbed_mean": _axis_stats(by_tag[tag], axis)["mean"],
                "baseline_mean_in_pair": (
                    float(sum(float(b.get(axis, 0)) for _, b, _ in pairs)) / len(pairs)
                    if pairs else float("nan")
                ),
                "discrimination_rate": disc,
                "wilcoxon": wcx,
                "expected_drop_threshold": expected,
                "verdict": probe_verdict(mean, lo, hi, expected),
            }
        probes[tag] = {"n_paired": len(pairs), "axes": per_axis, "missing": False}

    return {
        "baseline_tag": BASELINE_TAG,
        "baseline": baseline_block,
        "probes": probes,
    }


# ---------- Report rendering ----------


def _fmt_delta(mean: float, lo: float, hi: float) -> str:
    """Render a mean Δ cell with 95 % CI brackets (or ``--`` if NaN)."""
    if any(math.isnan(x) for x in (mean, lo, hi)):
        return "--"
    return f"{mean:+.2f} [{lo:+.2f}, {hi:+.2f}]"


def _fmt_pvalue(wcx: dict[str, Any]) -> str:
    """Render a Wilcoxon result cell."""
    p = wcx.get("p_value")
    n_eff = wcx.get("n_effective")
    if p is None:
        err = wcx.get("error")
        if err:
            return f"(n_eff={n_eff}; {err})"
        return f"(n_eff={n_eff}; no p)"
    return f"p={p:.3g} (n_eff={n_eff})"


def render_validation_report(summary: dict[str, Any]) -> str:
    """Render the markdown report. Sections: baseline, per-probe tables,
    verdict legend.
    """
    lines: list[str] = []
    lines.append("# Listwise LLM-as-a-Judge -- validation report")
    lines.append("")
    lines.append(
        "The baseline teacher pass saturated the 1-5 scale, so this run "
        "checks whether the judge is calibrated or merely stamping 5. "
        "Three controlled perturbations of the teacher output are judged "
        "under the same conditions; per-sample paired deltas tell us "
        "whether the judge actually reacts to quality defects."
    )
    lines.append("")

    # Baseline block.
    base = summary["baseline"]
    lines.append(f"## Baseline `{summary['baseline_tag']}` "
                 f"({base['n_records']} scored records)")
    lines.append("")
    lines.append("| axis | mean | 95% CI |")
    lines.append("|---|---:|---|")
    for axis in AXES:
        s = base["axes"][axis]
        lines.append(
            f"| {axis} | {s['mean']:.2f} | "
            f"[{s['ci95_low']:.2f}, {s['ci95_high']:.2f}] |"
        )
    lines.append("")

    # Per-probe tables.
    lines.append("## Per-perturbation paired deltas (perturbed − baseline)")
    lines.append("")
    lines.append(
        "`discrimination_rate` is the share of paired samples where the "
        "perturbed score is strictly less than the baseline score. A "
        "well-calibrated judge should drive this above 0.5 on the axes "
        "where the perturbation was designed to hurt. Wilcoxon alternative "
        "is `less` (one-sided test that Δ < 0)."
    )
    lines.append("")
    lines.append(
        "| probe | axis | n_paired | mean Δ [95% CI] | "
        "discrimination | Wilcoxon | expected Δ ≤ | verdict |"
    )
    lines.append("|---|---|---:|---|---:|---|---:|---|")
    for tag in PERTURBATION_TAGS:
        probe = summary["probes"].get(tag, {})
        if probe.get("missing"):
            lines.append(f"| `{tag}` | -- | 0 | -- | -- | -- | -- | missing |")
            continue
        for axis in AXES:
            a = probe["axes"][axis]
            lines.append(
                f"| `{tag}` | {axis} | {a['n_paired']} | "
                f"{_fmt_delta(a['mean_delta'], a['delta_ci95_low'], a['delta_ci95_high'])} | "
                f"{a['discrimination_rate']:.0%} | "
                f"{_fmt_pvalue(a['wilcoxon'])} | "
                f"{a['expected_drop_threshold']:+.2f} | "
                f"**{a['verdict']}** |"
            )
    lines.append("")

    # Legend.
    lines.append("## Verdict legend")
    lines.append("")
    lines.append("- **pass**: 95% CI lies entirely below 0 and mean Δ meets the expected-drop threshold for that axis.")
    lines.append("- **weak**: 95% CI lies below 0 but mean Δ is smaller than the expected threshold (judge reacts, but weakly).")
    lines.append("- **null**: 95% CI straddles 0 — perturbation produced no detectable change.")
    lines.append("- **inverted**: perturbed score went UP (CI above 0) — judge is responding to the wrong signal.")
    lines.append("- **consistent**: axis where no drop was expected, and no drop observed (CI contains 0 or above).")
    lines.append("- **unexpected-drop**: axis where no drop was expected, but one showed up — warrants investigation.")
    lines.append("- **insufficient**: too few paired samples to report.")
    lines.append("")
    lines.append("## Interpretation guide")
    lines.append("")
    lines.append("- If P2 (rationale_swapped) returns `pass` on both axes and P1 returns `pass` on Personalization, the judge is producing a quality signal on the teacher input distribution and the Stage 6.0 ceiling effect reflects genuine teacher quality rather than a broken judge.")
    lines.append("- If P2 returns `null` on Groundedness, the judge is not reading the rationale<->candidate alignment we designed the rubric to check. That would be a real failure of the current prompt and would warrant rewriting the Groundedness rubric before judging students.")
    lines.append("- If all three probes return `null`, the judge is uninformative at the teacher quality level and the Stage 6.0 result cannot be trusted.")
    lines.append("")
    return "\n".join(lines) + "\n"


# ---------- CLI ----------


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--raw",
        type=Path,
        default=PROJECT_ROOT / "data/results/judge_listwise_raw.jsonl",
    )
    p.add_argument(
        "--summary",
        type=Path,
        default=PROJECT_ROOT / "data/results/judge_listwise_validation.json",
    )
    p.add_argument(
        "--report",
        type=Path,
        default=PROJECT_ROOT / "data/results/judge_listwise_validation_report.md",
    )
    return p.parse_args()


def main() -> int:
    args = parse_args()
    args.summary.parent.mkdir(parents=True, exist_ok=True)
    args.report.parent.mkdir(parents=True, exist_ok=True)

    summary = analyze_validation(args.raw)
    args.summary.write_text(json.dumps(summary, indent=2, ensure_ascii=False))
    args.report.write_text(render_validation_report(summary))

    log.info("wrote summary -> %s", args.summary)
    log.info("wrote report -> %s", args.report)

    # Echo the verdict grid to stdout for quick inspection.
    for tag, probe in summary["probes"].items():
        if probe.get("missing"):
            log.info("%-35s MISSING", tag)
            continue
        for axis in AXES:
            a = probe["axes"][axis]
            log.info(
                "%-35s %-15s Δ=%+.2f [%+.2f, %+.2f]  disc=%.0f%%  verdict=%s",
                tag, axis,
                a["mean_delta"], a["delta_ci95_low"], a["delta_ci95_high"],
                a["discrimination_rate"] * 100,
                a["verdict"],
            )
    return 0


if __name__ == "__main__":
    sys.exit(main())

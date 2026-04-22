#!/usr/bin/env python
# ABOUTME: Sanity-check analyzer for listwise judge results -- per-model means
# ABOUTME: with bootstrap CIs plus correlation against retrieval metrics.

"""
Analyze ``judge_listwise_raw.jsonl`` and emit a sanity-check report.

This step is **descriptive only** -- no bias probes. (Full bias probes
-- position, verbosity, self-enhancement, rubric-order, score-ID --
were deferred in this project.) It answers two questions:

    1. What did the judge actually say per model? (mean Groundedness +
       mean Personalization with 95% bootstrap CIs.)
    2. Does the judge agree with the retrieval-metric ground truth?
       Per-sample correlations:
         - point-biserial( judge_score, R@1 hit )    -> binary outcome
         - Spearman      ( judge_score, MRR ranking )-> continuous

The retrieval metrics are reconstructed from the same inference cache
the judge consumed, so per-sample R@1 / MRR are computed without re-running
``eval_metrics.py``.

Outputs:
    - data/results/judge_listwise_summary.json   (machine-readable)
    - data/results/judge_listwise_report.md      (Notion-ready table)

Example::

    $ python scripts/judge/analyze_judge_listwise.py \\
        --raw data/results/judge_listwise_raw.jsonl \\
        --inference-cache data/inference_samples/all_backends_merged.json \\
        --summary data/results/judge_listwise_summary.json \\
        --report data/results/judge_listwise_report.md
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.judge.judge_listwise import (  # noqa: E402
    aggregate_per_model,
    bootstrap_mean_ci,
    load_inference_cache,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("analyze_judge_listwise")


# ---------- Per-sample retrieval metrics from the inference cache ----------


def per_sample_retrieval(
    cache_entry: dict[str, Any],
    model_tag: str,
) -> dict[str, float | int | None]:
    """Reconstruct R@1, R@5, R@10, MRR, and rank for one (sample, model) pair.

    Args:
        cache_entry (dict): one element of ``cache.values()`` (i.e. one
            sample with its ``by_backend`` map).
        model_tag (str): which backend's ranking to score.

    Returns:
        dict: {"hit_at_1", "hit_at_5", "hit_at_10", "mrr", "rank_of_pos",
            "output_chars", "output_tokens"}. Each metric is None if the
            backend's ranking is missing or if the positive business_id
            does not appear in ``recovered_business_ids``.
    """
    pos = cache_entry.get("positive_business_id")
    bk = (cache_entry.get("by_backend") or {}).get(model_tag) or {}
    ids = bk.get("recovered_business_ids") or []
    output_text = bk.get("output_text") or ""
    out: dict[str, float | int | None] = {
        "hit_at_1": None,
        "hit_at_5": None,
        "hit_at_10": None,
        "mrr": None,
        "rank_of_pos": None,
        "output_chars": len(output_text),
        "output_tokens": bk.get("output_tokens"),
    }
    if not pos or not ids:
        return out
    try:
        rank = ids.index(pos)  # 0-based
    except ValueError:
        # Positive not in the recovered list -> all retrieval metrics are 0.
        out.update({"hit_at_1": 0, "hit_at_5": 0, "hit_at_10": 0, "mrr": 0.0, "rank_of_pos": -1})
        return out
    out["rank_of_pos"] = rank
    out["hit_at_1"] = int(rank < 1)
    out["hit_at_5"] = int(rank < 5)
    out["hit_at_10"] = int(rank < 10)
    out["mrr"] = 1.0 / (rank + 1)
    return out


# ---------- Correlation helpers ----------


def _safe_pointbiserialr(
    scores: list[float],
    binary: list[int],
) -> dict[str, float | int]:
    """Run ``scipy.stats.pointbiserialr`` on (scores, binary), guarding
    against degenerate inputs.

    Args:
        scores (list[float]): continuous variable (judge scores).
        binary (list[int]): 0/1 outcome (e.g. R@1 hit).

    Returns:
        dict: {"r": float, "p": float, "n": int}. Returns NaNs if either
            sample is empty, lengths mismatch, or one of the inputs has
            zero variance.
    """
    from scipy import stats  # local import to keep top-level deps light
    import numpy as np

    n = min(len(scores), len(binary))
    if n < 3:
        return {"r": float("nan"), "p": float("nan"), "n": n}
    s_arr = np.asarray(scores[:n], dtype=float)
    b_arr = np.asarray(binary[:n], dtype=float)
    # pointbiserialr requires non-zero variance on both sides.
    if s_arr.std() == 0.0 or b_arr.std() == 0.0:
        return {"r": float("nan"), "p": float("nan"), "n": n}
    res = stats.pointbiserialr(b_arr, s_arr)
    return {"r": float(res.statistic), "p": float(res.pvalue), "n": n}


def _safe_spearmanr(xs: list[float], ys: list[float]) -> dict[str, float | int]:
    """Run ``scipy.stats.spearmanr`` on two equal-length numeric samples,
    guarding against degenerate inputs.

    Args:
        xs (list[float]): first variable.
        ys (list[float]): second variable.

    Returns:
        dict: {"rho": float, "p": float, "n": int}.
    """
    from scipy import stats
    import numpy as np

    n = min(len(xs), len(ys))
    if n < 3:
        return {"rho": float("nan"), "p": float("nan"), "n": n}
    x_arr = np.asarray(xs[:n], dtype=float)
    y_arr = np.asarray(ys[:n], dtype=float)
    if x_arr.std() == 0.0 or y_arr.std() == 0.0:
        return {"rho": float("nan"), "p": float("nan"), "n": n}
    res = stats.spearmanr(x_arr, y_arr)
    return {"rho": float(res.statistic), "p": float(res.pvalue), "n": n}


# ---------- Length descriptive stats ----------


def _length_stats(values: list[float]) -> dict[str, float | int]:
    """Return basic length descriptives for a per-model output sample.

    Args:
        values (list[float]): one number per (sample, model) -- typically
            character count of ``output_text`` or token count.

    Returns:
        dict: {"n", "mean", "p50", "p95", "max"}. Missing values are
            filtered before the stats are computed.
    """
    import numpy as np

    arr = np.asarray([v for v in values if v is not None], dtype=float)
    if arr.size == 0:
        return {"n": 0, "mean": float("nan"), "p50": float("nan"), "p95": float("nan"), "max": float("nan")}
    return {
        "n": int(arr.size),
        "mean": float(arr.mean()),
        "p50": float(np.quantile(arr, 0.5)),
        "p95": float(np.quantile(arr, 0.95)),
        "max": float(arr.max()),
    }


# ---------- Main analysis ----------


def load_verdicts(raw_path: Path) -> list[dict[str, Any]]:
    """Load every verdict record from the append-only JSONL.

    Args:
        raw_path (Path): the file written by ``judge_listwise.py``.

    Returns:
        list[dict]: one record per line; malformed lines are skipped.
    """
    out: list[dict[str, Any]] = []
    if not raw_path.exists():
        return out
    with raw_path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                out.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return out


def analyze(
    raw_path: Path,
    inference_cache: Path | list[Path],
) -> dict[str, Any]:
    """Compute the full per-model summary + retrieval-correlation table.

    Args:
        raw_path (Path): judge_listwise_raw.jsonl.
        inference_cache (Path | list[Path]): one inference cache file
            (consolidated or per-backend) or a list of per-backend files
            to stitch.

    Returns:
        dict: {"per_model": {...}, "n_verdicts_total": int}. Each
            ``per_model`` entry mirrors ``aggregate_per_model`` plus
            ``retrieval`` (R@k means + per-axis correlations) and
            ``length`` (output_chars / output_tokens descriptives).
    """
    verdicts = load_verdicts(raw_path)
    cache = load_inference_cache(inference_cache)

    per_model_scored = aggregate_per_model(verdicts)

    # Build per-model paired arrays for retrieval correlations.
    model_arrays: dict[str, dict[str, list[Any]]] = {}
    for v in verdicts:
        if v.get("error") is not None:
            continue
        sid = v.get("sample_id")
        tag = v.get("model_tag")
        g = v.get("groundedness")
        p = v.get("personalization")
        rc = v.get("ranking_coherence")  # None for v1/v2 records
        if not sid or not tag or g is None or p is None:
            continue
        entry = cache.get(sid)
        if entry is None:
            continue
        retr = per_sample_retrieval(entry, tag)
        if retr["hit_at_1"] is None or retr["mrr"] is None:
            continue
        ma = model_arrays.setdefault(tag, {
            "g": [], "p": [], "rc": [],
            "hit1": [], "hit5": [], "hit10": [], "mrr": [],
            "chars": [], "tokens": [],
            # Parallel arrays for correlations -- ranking_coherence is
            # missing on v1/v2 records, so track its own hit/mrr copies
            # to keep lengths in sync when some records carry rc and
            # others don't.
            "rc_hit1": [], "rc_mrr": [],
        })
        ma["g"].append(float(g))
        ma["p"].append(float(p))
        ma["hit1"].append(int(retr["hit_at_1"]))
        ma["hit5"].append(int(retr["hit_at_5"]))
        ma["hit10"].append(int(retr["hit_at_10"]))
        ma["mrr"].append(float(retr["mrr"]))
        if rc is not None:
            ma["rc"].append(float(rc))
            ma["rc_hit1"].append(int(retr["hit_at_1"]))
            ma["rc_mrr"].append(float(retr["mrr"]))
        if retr["output_chars"] is not None:
            ma["chars"].append(float(retr["output_chars"]))
        if retr["output_tokens"] is not None:
            ma["tokens"].append(float(retr["output_tokens"]))

    # Stitch retrieval + correlations + length stats into the summary.
    for tag, s in per_model_scored.items():
        ma = model_arrays.get(tag)
        if ma is None or not ma["g"]:
            s["retrieval"] = None
            s["length"] = None
            continue
        recall_at_1 = sum(ma["hit1"]) / len(ma["hit1"])
        recall_at_5 = sum(ma["hit5"]) / len(ma["hit5"])
        recall_at_10 = sum(ma["hit10"]) / len(ma["hit10"])
        mrr_mean = sum(ma["mrr"]) / len(ma["mrr"])

        s["retrieval"] = {
            "n_paired": len(ma["g"]),
            "recall_at_1": recall_at_1,
            "recall_at_5": recall_at_5,
            "recall_at_10": recall_at_10,
            "mrr_mean": mrr_mean,
            "groundedness_vs_hit1": _safe_pointbiserialr(ma["g"], ma["hit1"]),
            "personalization_vs_hit1": _safe_pointbiserialr(ma["p"], ma["hit1"]),
            "ranking_coherence_vs_hit1": (
                _safe_pointbiserialr(ma["rc"], ma["rc_hit1"]) if ma["rc"] else None
            ),
            "groundedness_vs_mrr": _safe_spearmanr(ma["g"], ma["mrr"]),
            "personalization_vs_mrr": _safe_spearmanr(ma["p"], ma["mrr"]),
            "ranking_coherence_vs_mrr": (
                _safe_spearmanr(ma["rc"], ma["rc_mrr"]) if ma["rc"] else None
            ),
        }
        s["length"] = {
            "output_chars": _length_stats(ma["chars"]),
            "output_tokens": _length_stats(ma["tokens"]),
        }

    return {
        "n_verdicts_total": len(verdicts),
        "per_model": per_model_scored,
    }


# ---------- Markdown report ----------


def _fmt_ci(mean: float, lo: float, hi: float, digits: int = 2) -> str:
    """Format a mean +/- CI cell for the markdown report.

    Args:
        mean (float): point estimate.
        lo (float): CI lower bound.
        hi (float): CI upper bound.
        digits (int): decimal places.

    Returns:
        str: ``"mean [lo, hi]"`` or ``"--"`` if any is NaN.
    """
    import math as _m

    if any(_m.isnan(x) for x in (mean, lo, hi)):
        return "--"
    return f"{mean:.{digits}f} [{lo:.{digits}f}, {hi:.{digits}f}]"


def _fmt_corr(d: dict[str, float | int] | None, key: str = "r") -> str:
    """Format a correlation cell as ``"r (p=..., n=...)"``.

    Args:
        d (dict | None): output of ``_safe_pointbiserialr`` / ``_safe_spearmanr``.
        key (str): which statistic key to render ("r" or "rho").

    Returns:
        str: short cell or ``"--"`` on missing/NaN.
    """
    import math as _m

    if d is None:
        return "--"
    val = d.get(key)
    p = d.get("p")
    n = d.get("n")
    if val is None or _m.isnan(float(val)):
        return "--"
    return f"{val:+.2f} (p={p:.3g}, n={n})"


def render_report_md(summary: dict[str, Any]) -> str:
    """Render a Notion-ready markdown report from the summary dict.

    Args:
        summary (dict): output of ``analyze``.

    Returns:
        str: full markdown document.
    """
    lines: list[str] = []
    lines.append("# Listwise LLM-as-a-Judge -- summary report")
    lines.append("")
    lines.append(
        f"Total verdicts on disk: **{summary['n_verdicts_total']}** "
        "(includes errored attempts)."
    )
    lines.append("")
    lines.append("## Per-model judge scores (1-10 scale, bootstrap 95% CI)")
    lines.append("")
    lines.append("| model | n_scored | n_errors | Groundedness | Personalization | Ranking Coherence |")
    lines.append("|---|---:|---:|---|---|---|")
    for tag, s in summary["per_model"].items():
        g = s["groundedness"]
        p = s["personalization"]
        rc = s.get("ranking_coherence") or {"mean": float("nan"), "ci95_low": float("nan"), "ci95_high": float("nan")}
        lines.append(
            f"| `{tag}` | {s['n_scored']} | {s['n_errors']} | "
            f"{_fmt_ci(g['mean'], g['ci95_low'], g['ci95_high'])} | "
            f"{_fmt_ci(p['mean'], p['ci95_low'], p['ci95_high'])} | "
            f"{_fmt_ci(rc['mean'], rc['ci95_low'], rc['ci95_high'])} |"
        )
    lines.append("")
    lines.append("## Retrieval-metric agreement (per-sample paired with judge scores)")
    lines.append("")
    lines.append("Sanity check: a calibrated judge should show **positive** "
                 "correlation between its scores and the model's R@1 / MRR on "
                 "the same sample. Negative or near-zero correlations suggest "
                 "the judge is reacting to surface features (style, length) "
                 "rather than recommendation quality -- decompose via the "
                 "formal bias probes listed in Future work below.")
    lines.append("")
    lines.append(
        "| model | n_paired | R@1 | R@5 | MRR | "
        "Grnd vs R@1 | Pers vs R@1 | Rank vs R@1 | "
        "Grnd vs MRR | Pers vs MRR | Rank vs MRR |"
    )
    lines.append("|---|---:|---:|---:|---:|---|---|---|---|---|---|")
    for tag, s in summary["per_model"].items():
        retr = s.get("retrieval")
        if retr is None:
            lines.append(f"| `{tag}` | 0 | -- | -- | -- | -- | -- | -- | -- | -- | -- |")
            continue
        lines.append(
            f"| `{tag}` | {retr['n_paired']} | "
            f"{retr['recall_at_1']:.3f} | "
            f"{retr['recall_at_5']:.3f} | "
            f"{retr['mrr_mean']:.3f} | "
            f"{_fmt_corr(retr['groundedness_vs_hit1'], 'r')} | "
            f"{_fmt_corr(retr['personalization_vs_hit1'], 'r')} | "
            f"{_fmt_corr(retr.get('ranking_coherence_vs_hit1'), 'r')} | "
            f"{_fmt_corr(retr['groundedness_vs_mrr'], 'rho')} | "
            f"{_fmt_corr(retr['personalization_vs_mrr'], 'rho')} | "
            f"{_fmt_corr(retr.get('ranking_coherence_vs_mrr'), 'rho')} |"
        )
    lines.append("")
    lines.append("## Output-length descriptives")
    lines.append("")
    lines.append("Reported here for context only. A formal verbosity-bias "
                 "test (length-controlled pair test, partial correlation) is "
                 "scoped under Future work below.")
    lines.append("")
    lines.append("| model | n | chars mean | chars p95 | tokens mean | tokens p95 |")
    lines.append("|---|---:|---:|---:|---:|---:|")
    for tag, s in summary["per_model"].items():
        ln = s.get("length")
        if ln is None:
            lines.append(f"| `{tag}` | 0 | -- | -- | -- | -- |")
            continue
        c = ln["output_chars"]
        t = ln["output_tokens"]
        lines.append(
            f"| `{tag}` | {c['n']} | "
            f"{c['mean']:.0f} | {c['p95']:.0f} | "
            f"{t['mean']:.0f} | {t['p95']:.0f} |"
        )
    lines.append("")
    lines.append("---")
    lines.append("")
    lines.append("## Future work (deferred bias probes)")
    lines.append("")
    lines.append("- **Position bias** (pairwise-listwise hybrid swap test)")
    lines.append("- **Verbosity bias** (length-controlled pair test + partial correlation)")
    lines.append("- **Self-enhancement / preference leakage** (Gemini judge vs Gemini teacher vs Qwen3.5 teacher matched-sample test)")
    lines.append("- **Rubric order bias** (criterion order reversal)")
    lines.append("- **Score ID bias** (1-5 vs 1-10 vs Likert scale comparison)")
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
        "--inference-cache",
        type=Path,
        action="append",
        default=None,
        help=(
            "inference cache JSON. May be passed multiple times "
            "(per-backend files are stitched). Defaults to "
            "all_backends_merged.json when omitted."
        ),
    )
    p.add_argument(
        "--summary",
        type=Path,
        default=PROJECT_ROOT / "data/results/judge_listwise_summary.json",
    )
    p.add_argument(
        "--report",
        type=Path,
        default=PROJECT_ROOT / "data/results/judge_listwise_report.md",
    )
    return p.parse_args()


def main() -> int:
    args = parse_args()
    args.summary.parent.mkdir(parents=True, exist_ok=True)
    args.report.parent.mkdir(parents=True, exist_ok=True)

    cache_paths: list[Path] = args.inference_cache or [
        PROJECT_ROOT / "data/inference_samples/all_backends_merged.json"
    ]
    summary = analyze(args.raw, cache_paths)
    args.summary.write_text(json.dumps(summary, indent=2, ensure_ascii=False))
    args.report.write_text(render_report_md(summary))

    log.info(
        "wrote summary -> %s (n_verdicts=%d, models=%d)",
        args.summary, summary["n_verdicts_total"], len(summary["per_model"]),
    )
    log.info("wrote report -> %s", args.report)

    for tag, s in summary["per_model"].items():
        g = s["groundedness"]
        p = s["personalization"]
        rc = s.get("ranking_coherence") or {"mean": float("nan"), "ci95_low": float("nan"), "ci95_high": float("nan")}
        log.info(
            "%-50s G=%.2f [%.2f, %.2f]  P=%.2f [%.2f, %.2f]  RC=%.2f [%.2f, %.2f]  n=%d/%d",
            tag,
            g["mean"], g["ci95_low"], g["ci95_high"],
            p["mean"], p["ci95_low"], p["ci95_high"],
            rc["mean"], rc["ci95_low"], rc["ci95_high"],
            s["n_scored"], s["n_total"],
        )
    return 0


if __name__ == "__main__":
    sys.exit(main())

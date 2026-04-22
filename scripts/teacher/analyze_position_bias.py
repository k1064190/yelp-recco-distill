#!/usr/bin/env python
# ABOUTME: Position bias analysis for Yelp KD dataset.
# ABOUTME: Checks GT positive position distribution and teacher top-1 ranking bias across candidate slots 1..10.
"""Position bias analysis for the Yelp KD dataset.

Three complementary analyses, all against the null hypothesis that a given
quantity is uniformly distributed across the K=10 candidate slots.

1. Dataset GT position distribution
   - For every sample in ``data/processed/philly_samples.jsonl`` locate the
     slot index (1-based) of ``positive_business_id`` inside ``candidates``.
   - Should be uniform because ``scripts/data/preprocess_yelp.py`` calls
     ``rng.shuffle(candidate_ids)`` with a seeded RNG. Serves as a sanity
     check on the shuffle.

2. Teacher top-1 position distribution
   - For every strict-valid record in a teacher jsonl read
     ``teacher_output.ranking[0]`` and tally the counts by slot.
   - Non-uniformity here is the classical LLM-as-judge "position bias": the
     teacher prefers certain slots regardless of content.

3. Teacher-accuracy-by-GT-position
   - For every sample where both the teacher and the dataset agree, compute
     ``ranking[0] == gt_position``. Report per-GT-position recall so we can
     see whether the teacher is systematically stronger when the truth sits
     at a particular slot (interaction between dataset and teacher biases).

For each distribution we report counts, percentages, the chi-square
goodness-of-fit statistic with 9 degrees of freedom, and its p-value.
Results are written to ``data/results/position_bias.json``.

Usage
-----
    python scripts/teacher/analyze_position_bias.py

"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any

from scipy import stats

K = 10  # candidate slots, 1-based in teacher contract

REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_SAMPLES = REPO_ROOT / "data" / "processed" / "philly_samples.jsonl"
DEFAULT_TEACHERS = [
    REPO_ROOT / "data" / "teacher" / "philly_teacher_qwen35.jsonl",
    REPO_ROOT / "data" / "teacher" / "philly_teacher.jsonl",
]
DEFAULT_OUTPUT = REPO_ROOT / "data" / "results" / "position_bias.json"


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    """Read a JSON-Lines file into a list of dicts.

    Args:
        path (Path): absolute path to a ``*.jsonl`` file.

    Returns:
        list[dict]: one dict per line, in file order.
    """
    with path.open() as fh:
        return [json.loads(line) for line in fh if line.strip()]


def gt_position(sample: dict[str, Any]) -> int | None:
    """Locate the 1-based slot of ``positive_business_id`` in ``candidates``.

    Args:
        sample (dict): one record from ``philly_samples.jsonl``; must contain
            keys ``candidates`` (list of dicts with ``business_id``) and
            ``positive_business_id`` (str).

    Returns:
        int | None: 1-based slot in ``[1, K]`` if found, otherwise ``None``.
    """
    pos_bid = sample["positive_business_id"]
    for idx, cand in enumerate(sample["candidates"], start=1):
        if cand["business_id"] == pos_bid:
            return idx
    return None


def chi2_uniform(counts: list[int]) -> tuple[float, float]:
    """Chi-square goodness-of-fit against a discrete uniform over K slots.

    Args:
        counts (list[int]): length-K observed counts (index i → slot i+1).

    Returns:
        tuple[float, float]: (chi-square statistic, two-sided p-value). The
            test has ``K - 1`` degrees of freedom.
    """
    total = sum(counts)
    expected = [total / K] * K
    result = stats.chisquare(f_obs=counts, f_exp=expected)
    return float(result.statistic), float(result.pvalue)


def counts_from_iter(slots: list[int]) -> list[int]:
    """Tally 1-based slot ids into a length-K count vector.

    Args:
        slots (list[int]): slot ids in ``[1, K]``.

    Returns:
        list[int]: ``out[i]`` is the number of observations with slot ``i+1``.
    """
    c = Counter(slots)
    return [c.get(i, 0) for i in range(1, K + 1)]


def describe(name: str, counts: list[int]) -> dict[str, Any]:
    """Summarize a slot distribution: counts, %, argmax/argmin, chi-square.

    Args:
        name (str): human-readable label for logging only.
        counts (list[int]): length-K observed counts produced by
            :func:`counts_from_iter`.

    Returns:
        dict: keys ``n``, ``counts``, ``percent``, ``uniform_percent``,
            ``max_slot``, ``max_percent``, ``min_slot``, ``min_percent``,
            ``chi2``, ``p_value``, ``df``.
    """
    total = sum(counts)
    pct = [100.0 * c / total for c in counts] if total else [0.0] * K
    chi2, pval = chi2_uniform(counts)
    max_slot = 1 + max(range(K), key=lambda i: counts[i])
    min_slot = 1 + min(range(K), key=lambda i: counts[i])
    return {
        "name": name,
        "n": total,
        "counts": counts,
        "percent": [round(p, 3) for p in pct],
        "uniform_percent": round(100.0 / K, 3),
        "max_slot": max_slot,
        "max_percent": round(pct[max_slot - 1], 3),
        "min_slot": min_slot,
        "min_percent": round(pct[min_slot - 1], 3),
        "chi2": round(chi2, 4),
        "p_value": float(f"{pval:.6g}"),
        "df": K - 1,
    }


def render_table(summary: dict[str, Any]) -> str:
    """Render a per-slot table for a single distribution summary.

    Args:
        summary (dict): output of :func:`describe`.

    Returns:
        str: multi-line fixed-width table, ending with chi-square line.
    """
    lines = [f"\n== {summary['name']} (N={summary['n']}) =="]
    lines.append(f"{'slot':>4}  {'count':>6}  {'pct':>7}  {'bar':<30}")
    pcts = summary["percent"]
    max_p = max(pcts) if pcts else 1.0
    for slot in range(1, K + 1):
        c = summary["counts"][slot - 1]
        p = pcts[slot - 1]
        bar_len = int(round(30 * (p / max_p))) if max_p else 0
        lines.append(f"{slot:>4d}  {c:>6d}  {p:>6.2f}%  {'#' * bar_len}")
    verdict = "UNIFORM" if summary["p_value"] >= 0.05 else "NON-UNIFORM"
    lines.append(
        f"chi2={summary['chi2']:.3f}, df={summary['df']}, "
        f"p={summary['p_value']:.4g} -> {verdict} (alpha=0.05)"
    )
    lines.append(
        f"argmax slot {summary['max_slot']} at {summary['max_percent']}% "
        f"(uniform expect {summary['uniform_percent']}%)"
    )
    return "\n".join(lines)


def analyze_teacher(
    teacher_path: Path,
    gt_map: dict[str, int],
) -> dict[str, Any] | None:
    """Analyze top-1 position bias and GT-conditional recall for one teacher.

    Args:
        teacher_path (Path): path to a teacher ``*.jsonl`` file with records
            having ``sample_id``, ``error``, and
            ``teacher_output.ranking`` (list of 10 ints, 1-based).
        gt_map (dict[str, int]): ``sample_id`` → GT slot in ``[1, K]``.

    Returns:
        dict | None: ``None`` if the file has no strict-valid records.
            Otherwise a dict with:
              - ``path``: str
              - ``top1_distribution``: :func:`describe` output over
                ``ranking[0]``.
              - ``recall_by_gt_position``: list of floats length K, where
                index i holds recall@1 conditional on GT at slot i+1.
              - ``n_strict_valid``: int
              - ``n_with_gt``: int (intersection with ``gt_map``).
    """
    records = load_jsonl(teacher_path)
    top1_slots: list[int] = []
    hits_by_gt: Counter[int] = Counter()
    totals_by_gt: Counter[int] = Counter()

    n_strict = 0
    for rec in records:
        if rec.get("error") is not None:
            continue
        out = rec.get("teacher_output")
        if not out:
            continue
        ranking = out.get("ranking")
        if not (isinstance(ranking, list) and len(ranking) == K):
            continue
        if not all(isinstance(r, int) and 1 <= r <= K for r in ranking):
            continue
        if len(set(ranking)) != K:
            continue  # duplicate index → not strict-valid
        n_strict += 1
        top1 = ranking[0]
        top1_slots.append(top1)
        gt = gt_map.get(rec["sample_id"])
        if gt is not None:
            totals_by_gt[gt] += 1
            if top1 == gt:
                hits_by_gt[gt] += 1

    if not top1_slots:
        return None

    top1_summary = describe(f"Teacher top-1 slot @ {teacher_path.name}",
                            counts_from_iter(top1_slots))

    recall = [
        round(hits_by_gt[g] / totals_by_gt[g], 4) if totals_by_gt[g] else None
        for g in range(1, K + 1)
    ]

    return {
        "path": str(teacher_path),
        "top1_distribution": top1_summary,
        "recall_by_gt_position": recall,
        "recall_support": [totals_by_gt[g] for g in range(1, K + 1)],
        "n_strict_valid": n_strict,
        "n_with_gt": sum(totals_by_gt.values()),
    }


def render_recall_table(teacher_name: str, recall: list[float | None],
                        support: list[int]) -> str:
    """Render per-GT-slot teacher recall@1 as an ASCII table.

    Args:
        teacher_name (str): label of the teacher file (for the header).
        recall (list[float | None]): length-K recall values from
            :func:`analyze_teacher`; ``None`` where support is zero.
        support (list[int]): length-K support counts.

    Returns:
        str: multi-line table.
    """
    lines = [f"\n== Teacher recall@1 by GT slot — {teacher_name} =="]
    lines.append(f"{'gt_slot':>7}  {'support':>7}  {'recall@1':>9}  bar")
    valid = [r for r in recall if r is not None]
    max_r = max(valid) if valid else 1.0
    for g in range(1, K + 1):
        r = recall[g - 1]
        s = support[g - 1]
        if r is None:
            lines.append(f"{g:>7d}  {s:>7d}       n/a")
        else:
            bar_len = int(round(30 * (r / max_r))) if max_r else 0
            lines.append(f"{g:>7d}  {s:>7d}    {r:>6.4f}  {'#' * bar_len}")
    if valid:
        lines.append(
            f"range: min={min(valid):.4f}, max={max(valid):.4f}, "
            f"spread={max(valid) - min(valid):.4f}"
        )
    return "\n".join(lines)


def main() -> None:
    """Entry point: parse args, load inputs, run analyses, persist JSON."""
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--samples", type=Path, default=DEFAULT_SAMPLES,
                        help="processed samples jsonl with candidates + positive_business_id")
    parser.add_argument("--teachers", type=Path, nargs="*", default=DEFAULT_TEACHERS,
                        help="one or more teacher jsonl paths to analyze")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT,
                        help="where to write the aggregated json summary")
    args = parser.parse_args()

    samples = load_jsonl(args.samples)
    gt_slots: list[int] = []
    gt_map: dict[str, int] = {}
    missing = 0
    for s in samples:
        pos = gt_position(s)
        if pos is None:
            missing += 1
            continue
        gt_slots.append(pos)
        gt_map[s["sample_id"]] = pos

    print(f"Loaded {len(samples)} samples from {args.samples}")
    if missing:
        print(f"  WARNING: {missing} samples had positive_business_id not in candidates")

    gt_summary = describe("Dataset GT slot of positive_business_id",
                          counts_from_iter(gt_slots))
    print(render_table(gt_summary))

    teacher_results = []
    for tpath in args.teachers:
        if not tpath.exists():
            print(f"\n[skip] teacher file not found: {tpath}")
            continue
        tres = analyze_teacher(tpath, gt_map)
        if tres is None:
            print(f"\n[skip] no strict-valid records in {tpath}")
            continue
        teacher_results.append(tres)
        print(render_table(tres["top1_distribution"]))
        print(render_recall_table(tpath.name,
                                  tres["recall_by_gt_position"],
                                  tres["recall_support"]))

    payload = {
        "dataset_gt_position": gt_summary,
        "teachers": teacher_results,
        "samples_path": str(args.samples),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w") as fh:
        json.dump(payload, fh, indent=2)
    print(f"\nSaved summary to {args.output}")


if __name__ == "__main__":
    main()

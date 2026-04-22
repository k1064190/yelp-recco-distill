#!/usr/bin/env python
# ABOUTME: Full rank×slot position-bias analysis with heatmap visualization.
# ABOUTME: Extends the top-1-only audit to all 10 rank positions across every teacher file.
"""Full rank×slot position-bias heatmap analysis.

For each teacher dataset, builds a 10×10 matrix:

    M[rank_pos, slot] = fraction of samples where the teacher placed
                        candidate at slot ``slot`` at rank position ``rank_pos``

Under no position bias, every cell equals 1/K = 10 %.  Primacy bias
shows as slot-1 elevated in the top ranks and depressed in the bottom
ranks; recency bias as slot-10 elevated at top.

Outputs
-------
- A multi-panel heatmap PNG (one panel per dataset).
- A per-dataset χ² table (one test per rank position row, 9 df each).
- Machine-readable JSON with the full matrices and test results.

Usage
-----
    python scripts/teacher/visualize_position_bias.py

All teacher files are auto-discovered from ``data/teacher/``.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from scipy import stats

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))
from configs.teacher_prompt import N_CANDIDATES  # noqa: E402

K = N_CANDIDATES


# --------------- data loading ---------------

def load_strict_valid_rankings(path: Path) -> list[list[int]]:
    """Load strict-valid rankings from a teacher JSONL file.

    Args:
        path (Path): teacher JSONL file.

    Returns:
        list[list[int]]: each inner list is a length-K ranking (1-based,
            best-to-worst). Only records with ``error is None`` and a
            permutation-valid ranking are included.
    """
    rankings: list[list[int]] = []
    with path.open() as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue
            if rec.get("error") is not None:
                continue
            out = rec.get("teacher_output")
            if not isinstance(out, dict):
                continue
            r = out.get("ranking")
            if not (isinstance(r, list) and len(r) == K):
                continue
            if not all(isinstance(x, int) and 1 <= x <= K for x in r):
                continue
            if len(set(r)) != K:
                continue
            rankings.append(r)
    return rankings


# --------------- analysis ---------------

def build_rank_slot_matrix(rankings: list[list[int]]) -> np.ndarray:
    """Build a 10×10 count matrix: rows = rank position, cols = slot.

    Args:
        rankings (list[list[int]]): each ranking is a 1-based best-to-worst
            list of length K.

    Returns:
        np.ndarray: shape ``[K, K]``, dtype float64. ``M[r, s]`` is the
            count of samples where rank position ``r`` (0-indexed, 0=best)
            was assigned to slot ``s+1``.
    """
    M = np.zeros((K, K), dtype=np.float64)
    for ranking in rankings:
        for rank_pos, slot in enumerate(ranking):
            M[rank_pos, slot - 1] += 1
    return M


def matrix_to_percent(M: np.ndarray) -> np.ndarray:
    """Normalize each row of a count matrix to percentages.

    Args:
        M (np.ndarray): shape ``[K, K]`` count matrix from
            :func:`build_rank_slot_matrix`.

    Returns:
        np.ndarray: same shape, each row sums to 100.0.
    """
    row_sums = M.sum(axis=1, keepdims=True)
    row_sums = np.where(row_sums == 0, 1.0, row_sums)
    return M / row_sums * 100.0


def per_row_chi2(M: np.ndarray) -> list[dict[str, float]]:
    """Run χ² goodness-of-fit against uniform for each rank position row.

    Args:
        M (np.ndarray): shape ``[K, K]`` count matrix.

    Returns:
        list[dict]: one dict per row with keys ``rank_pos`` (1-based),
            ``chi2``, ``p_value``, ``max_slot`` (1-based), ``max_pct``.
    """
    results = []
    pct = matrix_to_percent(M)
    for r in range(K):
        row = M[r]
        total = row.sum()
        expected = np.full(K, total / K)
        chi2_stat, p_val = stats.chisquare(f_obs=row, f_exp=expected)
        max_col = int(np.argmax(row))
        results.append({
            "rank_pos": r + 1,
            "chi2": round(float(chi2_stat), 3),
            "p_value": float(f"{p_val:.4g}"),
            "df": K - 1,
            "max_slot": max_col + 1,
            "max_pct": round(float(pct[r, max_col]), 2),
            "verdict": "UNIFORM" if p_val >= 0.05 else "NON-UNIFORM",
        })
    return results


def overall_chi2(M: np.ndarray) -> dict[str, float]:
    """Run a single χ² test across the entire K×K matrix vs uniform.

    Args:
        M (np.ndarray): shape ``[K, K]`` count matrix.

    Returns:
        dict: ``chi2``, ``p_value``, ``df``.
    """
    total = M.sum()
    expected = np.full_like(M, total / (K * K))
    chi2_stat, p_val = stats.chisquare(f_obs=M.ravel(), f_exp=expected.ravel())
    return {
        "chi2": round(float(chi2_stat), 3),
        "p_value": float(f"{p_val:.4g}"),
        "df": K * K - 1,
    }


# --------------- visualization ---------------

LABEL_MAP = {
    "philly_teacher_qwen35.jsonl": "Offline Pass1\n(original)",
    "philly_teacher_qwen35_http_identity.jsonl": "HTTP Pass1\n(identity)",
    "philly_teacher_qwen35_perm_reverse.jsonl": "HTTP Pass2\n(reverse, prompt-space)",
    "philly_teacher_qwen35_borda.jsonl": "Mixed Borda\n(offline+HTTP)",
    "philly_teacher_qwen35_borda_http.jsonl": "HTTP-only Borda\n(production)",
    "philly_teacher.jsonl": "Gemini\n(Flash Preview)",
}


def plot_heatmaps(
    datasets: list[dict[str, Any]],
    output_path: Path,
) -> None:
    """Render a multi-panel heatmap figure.

    Args:
        datasets (list[dict]): each dict has keys ``name`` (str),
            ``pct_matrix`` (np.ndarray [K, K]), ``n`` (int),
            ``per_row`` (list[dict] from :func:`per_row_chi2`).
        output_path (Path): where to save the PNG.
    """
    n_ds = len(datasets)
    cols = min(3, n_ds)
    rows = (n_ds + cols - 1) // cols
    fig, axes = plt.subplots(
        rows, cols,
        figsize=(6.5 * cols, 5.5 * rows),
        squeeze=False,
    )

    slot_labels = [str(i) for i in range(1, K + 1)]
    rank_labels = [f"Rank {i}" for i in range(1, K + 1)]

    for idx, ds in enumerate(datasets):
        ax = axes[idx // cols][idx % cols]
        pct = ds["pct_matrix"]
        deviation = pct - 10.0

        sns.heatmap(
            deviation,
            ax=ax,
            annot=pct,
            fmt=".1f",
            cmap="RdBu_r",
            center=0,
            vmin=-12,
            vmax=12,
            xticklabels=slot_labels,
            yticklabels=rank_labels,
            cbar_kws={"label": "deviation from 10% (pp)", "shrink": 0.8},
            annot_kws={"size": 8},
            linewidths=0.3,
            linecolor="white",
        )
        n_non_uniform = sum(
            1 for r in ds["per_row"] if r["verdict"] == "NON-UNIFORM"
        )
        label = LABEL_MAP.get(ds["name"], ds["name"])
        ax.set_title(
            f"{label}\nN={ds['n']:,}  |  {n_non_uniform}/{K} rows NON-UNIFORM",
            fontsize=11,
            fontweight="bold",
        )
        ax.set_xlabel("Candidate Slot", fontsize=10)
        ax.set_ylabel("Rank Position", fontsize=10)

    for idx in range(n_ds, rows * cols):
        axes[idx // cols][idx % cols].set_visible(False)

    fig.suptitle(
        "Position Bias: Rank × Slot Distribution (cell = %, color = deviation from uniform 10%)",
        fontsize=14,
        fontweight="bold",
        y=1.01,
    )
    fig.tight_layout()
    fig.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def plot_slot1_profile(
    datasets: list[dict[str, Any]],
    output_path: Path,
) -> None:
    """Line plot: slot-1 percentage across all 10 rank positions per dataset.

    Args:
        datasets (list[dict]): same format as :func:`plot_heatmaps`.
        output_path (Path): where to save the PNG.
    """
    fig, ax = plt.subplots(figsize=(10, 6))
    rank_pos = np.arange(1, K + 1)
    colors = plt.cm.tab10(np.linspace(0, 1, len(datasets)))

    for idx, ds in enumerate(datasets):
        slot1_pcts = ds["pct_matrix"][:, 0]
        label = LABEL_MAP.get(ds["name"], ds["name"]).replace("\n", " ")
        ax.plot(rank_pos, slot1_pcts, "o-", color=colors[idx], label=label,
                markersize=6, linewidth=2)

    ax.axhline(10.0, color="gray", linestyle="--", linewidth=1, label="Uniform (10%)")
    ax.set_xlabel("Rank Position (1 = best)", fontsize=12)
    ax.set_ylabel("Slot 1 Assigned (%)", fontsize=12)
    ax.set_title("Slot 1 Representation Across All Rank Positions", fontsize=13,
                 fontweight="bold")
    ax.set_xticks(rank_pos)
    ax.set_xticklabels([f"Rank {i}" for i in rank_pos])
    ax.legend(fontsize=9, loc="upper right")
    ax.grid(True, alpha=0.3)
    ax.set_ylim(4, 24)
    fig.tight_layout()
    fig.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def plot_diagonal_profile(
    datasets: list[dict[str, Any]],
    output_path: Path,
) -> None:
    """Line plot: diagonal element pct[r, r] — does slot r get rank r?

    Under uniform this is flat at 10%. A model with NO position bias but
    strong content understanding would also be flat at 10% (content
    determines rank, not slot). If there IS position bias, the diagonal
    is elevated because the model conflates "slot i" with "rank i".

    Args:
        datasets (list[dict]): same format as above.
        output_path (Path): PNG output path.
    """
    fig, ax = plt.subplots(figsize=(10, 6))
    positions = np.arange(1, K + 1)
    colors = plt.cm.tab10(np.linspace(0, 1, len(datasets)))

    for idx, ds in enumerate(datasets):
        diag = np.diag(ds["pct_matrix"])
        label = LABEL_MAP.get(ds["name"], ds["name"]).replace("\n", " ")
        ax.plot(positions, diag, "s-", color=colors[idx], label=label,
                markersize=6, linewidth=2)

    ax.axhline(10.0, color="gray", linestyle="--", linewidth=1, label="Uniform (10%)")
    ax.set_xlabel("Position i (slot i at rank i)", fontsize=12)
    ax.set_ylabel("M[rank_i, slot_i] (%)", fontsize=12)
    ax.set_title("Diagonal Profile: Does Slot i Get Rank i? (position-rank conflation)",
                 fontsize=13, fontweight="bold")
    ax.set_xticks(positions)
    ax.legend(fontsize=9, loc="upper right")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


# --------------- main ---------------

def parse_args() -> argparse.Namespace:
    """Parse CLI args.

    Returns:
        argparse.Namespace: parsed arguments.
    """
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument(
        "--teachers", type=Path, nargs="*",
        default=[
            PROJECT_ROOT / "data/teacher/philly_teacher_qwen35.jsonl",
            PROJECT_ROOT / "data/teacher/philly_teacher_qwen35_http_identity.jsonl",
            PROJECT_ROOT / "data/teacher/philly_teacher_qwen35_perm_reverse.jsonl",
            PROJECT_ROOT / "data/teacher/philly_teacher_qwen35_borda.jsonl",
            PROJECT_ROOT / "data/teacher/philly_teacher_qwen35_borda_http.jsonl",
            PROJECT_ROOT / "data/teacher/philly_teacher.jsonl",
        ],
    )
    p.add_argument("--output-dir", type=Path,
                   default=PROJECT_ROOT / "data/results")
    return p.parse_args()


def main() -> None:
    """Entry point: load, analyze, visualize, persist."""
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    datasets: list[dict[str, Any]] = []

    for tpath in args.teachers:
        if not tpath.exists():
            print(f"[skip] {tpath}")
            continue
        rankings = load_strict_valid_rankings(tpath)
        if not rankings:
            print(f"[skip] no strict-valid rankings in {tpath}")
            continue

        M = build_rank_slot_matrix(rankings)
        pct = matrix_to_percent(M)
        per_row = per_row_chi2(M)
        overall = overall_chi2(M)

        ds = {
            "name": tpath.name,
            "path": str(tpath),
            "n": len(rankings),
            "count_matrix": M,
            "pct_matrix": pct,
            "per_row": per_row,
            "overall": overall,
        }
        datasets.append(ds)

        print(f"\n{'='*70}")
        print(f"  {tpath.name}  (N={len(rankings)})")
        print(f"  Overall χ²={overall['chi2']}, df={overall['df']}, "
              f"p={overall['p_value']}")
        print(f"{'='*70}")
        print(f"{'rank':>6}  {'max_slot':>8}  {'max_pct':>8}  "
              f"{'chi2':>8}  {'p':>10}  verdict")
        for r in per_row:
            print(f"  {r['rank_pos']:>4}  {r['max_slot']:>8}  "
                  f"{r['max_pct']:>7.2f}%  {r['chi2']:>8.2f}  "
                  f"{r['p_value']:>10.4g}  {r['verdict']}")

    if not datasets:
        print("No datasets to visualize.")
        return

    heatmap_path = args.output_dir / "position_bias_heatmaps.png"
    plot_heatmaps(datasets, heatmap_path)
    print(f"\nHeatmaps saved to {heatmap_path}")

    slot1_path = args.output_dir / "position_bias_slot1_profile.png"
    plot_slot1_profile(datasets, slot1_path)
    print(f"Slot-1 profile saved to {slot1_path}")

    diag_path = args.output_dir / "position_bias_diagonal.png"
    plot_diagonal_profile(datasets, diag_path)
    print(f"Diagonal profile saved to {diag_path}")

    json_payload = []
    for ds in datasets:
        json_payload.append({
            "name": ds["name"],
            "path": ds["path"],
            "n": ds["n"],
            "pct_matrix": ds["pct_matrix"].tolist(),
            "per_row_chi2": ds["per_row"],
            "overall_chi2": ds["overall"],
        })
    json_path = args.output_dir / "position_bias_full.json"
    with json_path.open("w") as fh:
        json.dump(json_payload, fh, indent=2)
    print(f"Full results JSON saved to {json_path}")


if __name__ == "__main__":
    main()

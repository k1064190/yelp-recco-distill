#!/usr/bin/env python
# ABOUTME: Single-panel 10x10 rank-by-slot heatmap for the v4 teacher dataset.
# ABOUTME: Focused figure for portfolio §2.1.3 (varB thinkOff guided).

"""
Render ``docs/assets/v4_rank_slot_heatmap.png`` — one large 10×10
heatmap of the v4 teacher dataset (`varB · thinkOff · guided JSON`).

Rows are rank positions (1 = best, 10 = worst). Columns are candidate
slots (position in the input prompt). Cell values are percentages per
row; under no positional bias every cell is 10 %. Primacy bias shows
as slot-1 elevated in the top rows; recency bias as slot-10 elevated
in the bottom rows. The cells are coloured by their deviation from
the 10 % uniform baseline so the eye immediately spots structure.

Reuses the matrix-building helpers from `visualize_position_bias.py`.

Usage::

    python scripts/teacher/plot_v4_rank_slot_heatmap.py \
        --jsonl data/teacher/philly_teacher_qwen35_B_http_identity.jsonl \
        --out docs/assets/v4_rank_slot_heatmap.png
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from scripts.teacher.visualize_position_bias import (  # noqa: E402
    K,
    build_rank_slot_matrix,
    load_strict_valid_rankings,
    matrix_to_percent,
)


def plot_single_heatmap(pct: np.ndarray, n: int, out_path: Path, title: str) -> None:
    """Render a single heatmap with percentage annotations and uniform reference.

    Args:
        pct: shape [K, K], each row sums to 100 (percent).
        n: number of samples used (for subtitle).
        out_path: destination PNG path.
        title: main title line.
    """
    # Deviation from 10 % uniform, for colour scaling.
    dev = pct - 100.0 / K
    vmax = float(np.max(np.abs(dev)))

    fig, ax = plt.subplots(figsize=(8.2, 6.6))
    sns.heatmap(
        dev,
        annot=pct,
        fmt=".1f",
        cmap="RdBu_r",
        center=0.0,
        vmin=-vmax,
        vmax=vmax,
        cbar_kws={"label": "deviation from uniform 10 %  (pp)"},
        linewidths=0.4,
        linecolor="white",
        annot_kws={"fontsize": 9},
        xticklabels=[str(s) for s in range(1, K + 1)],
        yticklabels=[str(r) for r in range(1, K + 1)],
        ax=ax,
    )
    ax.set_xlabel("candidate slot (position in prompt, 1–10)")
    ax.set_ylabel("rank position assigned (1 = best, 10 = worst)")
    ax.set_title(
        f"{title}\n"
        f"varB prompt · enable_thinking=False · guided JSON  (N = {n} valid)\n"
        f"cell = row-normalised %  (uniform baseline = 10.0 %)",
        fontsize=11,
    )

    # Annotate edge-of-matrix observations as text overlays.
    top_left = pct[0, 0]
    ax.text(
        0.02, 0.98,
        f"slot-1 at rank-1 = {top_left:.1f} % (expect 10 %)",
        transform=ax.transAxes,
        ha="left", va="top",
        fontsize=9,
        bbox=dict(boxstyle="round,pad=0.3", facecolor="white", edgecolor="#888", linewidth=0.6),
    )

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(out_path, dpi=160, bbox_inches="tight")
    print(f"wrote {out_path}  (n={n}, deviation range ±{vmax:.2f} pp)")


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument(
        "--jsonl", type=Path,
        default=PROJECT_ROOT / "data" / "teacher" / "philly_teacher_qwen35_B_http_identity.jsonl",
        help="teacher JSONL to visualise (default = v4)",
    )
    ap.add_argument(
        "--out", type=Path,
        default=PROJECT_ROOT / "docs" / "assets" / "v4_rank_slot_heatmap.png",
    )
    ap.add_argument(
        "--title", type=str,
        default="Qwen3.5-35B-A3B teacher position bias — v4 dataset",
    )
    return ap.parse_args()


def main() -> int:
    args = parse_args()
    if not args.jsonl.exists():
        print(f"error: teacher JSONL not found — {args.jsonl}", file=sys.stderr)
        return 1
    rankings = load_strict_valid_rankings(args.jsonl)
    if not rankings:
        print(f"error: no strict-valid rankings in {args.jsonl}", file=sys.stderr)
        return 2
    M = build_rank_slot_matrix(rankings)
    pct = matrix_to_percent(M)
    plot_single_heatmap(pct, len(rankings), args.out, args.title)
    return 0


if __name__ == "__main__":
    sys.exit(main())

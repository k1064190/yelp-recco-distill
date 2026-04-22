#!/usr/bin/env python
# ABOUTME: Render §2.1.3 slot-1 position-bias bar chart across prompt variant x thinking.
# ABOUTME: Loads slot-1 from eval_qwen35-teacher-*.json where available, hard-codes canon v3.

"""
Produce ``docs/assets/slot1_matrix.png`` — a grouped bar chart showing
slot-1 top-1 % for Qwen3.5-35B-A3B teacher across prompt variants
(A/B/C) × enable_thinking (ON/OFF), plus the uniform-10 % baseline.

Narrative in portfolio §2.1.3:
    - Variant A: thinkOff (21.95 %) vs thinkOn (13.46 %) — the single
      chat-template knob moves slot-1 by 8+ pp.
    - Variant B: thinkOff 13.94 % (uniform-ish) + teacher R@1 = 0.282.
      This is the v4 configuration chosen for all downstream SFT/GKD.
    - Variant C: thinkOff 25.8 % — most biased; unguided parse also 0 %.

Data sources:
    guided·thinkOff cells: prefer `eval_qwen35-teacher-offline-var{A,B}-thinkOff.json`
                            (A offline replay 2026-04-21; B offline 2026-04-21).
                            For C we use `eval_qwen35-teacher-guided-varC-original-v2.json`
                            because no thinkOff-offline variant was run for C.
                            (HTTP vs offline residual ≤ 5 pp per the HTTP vs offline inference-path audit.)
    guided·thinkOn cells:  A canonical v3 = 13.46 % (hard-coded; historical value
                            used for v3 training data). B / C = load from the
                            post-bias-fix re-run `eval_qwen35-teacher-offline-var{B,C}-thinkOn.json`;
                            if missing, cell is rendered as a hashed placeholder.

Usage::

    python scripts/teacher/plot_slot1_matrix.py \\
        --out docs/assets/slot1_matrix.png

Add ``--no-uniform-line`` to drop the baseline, or ``--canonical-v3-ona <float>`` to
override the hard-coded A-thinkOn value.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[2]
RESULTS = PROJECT_ROOT / "data" / "results"


def load_slot1(filename: str) -> float | None:
    """Return slot1_top1_rate from an eval JSON, or None if file missing."""
    p = RESULTS / filename
    if not p.exists():
        return None
    try:
        doc = json.loads(p.read_text())
    except json.JSONDecodeError:
        return None
    rate = doc.get("position_bias", {}).get("slot1_top1_rate")
    return float(rate) if rate is not None else None


def collect_cells(canonical_v3_a_on: float, v4_b_off: float | None = 0.1394) -> dict[tuple[str, str], float | None]:
    """Build { (variant, thinking) → slot1_pct_or_None } for the six-cell grid.

    Keys:
        (variant, thinking) where variant ∈ {A, B, C} and thinking ∈ {off, on}.

    Args:
        canonical_v3_a_on: hard-coded A·thinkOn slot-1 from the v3 training
            data (canonical, 0.1346).
        v4_b_off: override for B·thinkOff — if not None, uses this instead of
            the 287-sample eval value. Default 0.1394 which is the slot-1 of
            the v4 teacher *dataset* (full 3000-sample generation). Pass None
            to fall back to `eval_qwen35-teacher-offline-varB-thinkOff.json`
            (287-eval, slot-1 ≈ 0.109).
    """
    # guided · thinkOff row: A offline replay, B v4-dataset (or eval), C HTTP.
    off_a = load_slot1("eval_qwen35-teacher-offline-varA-thinkOff.json")
    off_b = v4_b_off if v4_b_off is not None else load_slot1("eval_qwen35-teacher-offline-varB-thinkOff.json")
    off_c = load_slot1("eval_qwen35-teacher-guided-varC-original-v2.json")

    # guided · thinkOn row: A = historical canonical v3 (hard-coded), B / C = new.
    on_a = canonical_v3_a_on
    on_b = load_slot1("eval_qwen35-teacher-offline-varB-thinkOn.json")
    on_c = load_slot1("eval_qwen35-teacher-offline-varC-thinkOn.json")

    return {
        ("A", "off"): off_a, ("A", "on"): on_a,
        ("B", "off"): off_b, ("B", "on"): on_b,
        ("C", "off"): off_c, ("C", "on"): on_c,
    }


def plot(cells: dict[tuple[str, str], float | None],
         out_path: Path,
         show_uniform: bool = True,
         title: str = "Teacher (Qwen3.5-35B-A3B) slot-1 top-1 % by prompt variant × thinking") -> None:
    variants = ["A", "B", "C"]
    labels = ["A\n(text description)", "B\n(json example, v4)", "C\n(no instruction)"]
    off_vals = [cells[(v, "off")] for v in variants]
    on_vals  = [cells[(v, "on")]  for v in variants]

    # Convert to percentages for display; None stays None.
    def to_pct(x): return 100.0 * x if x is not None else None
    off_pct = [to_pct(v) for v in off_vals]
    on_pct  = [to_pct(v) for v in on_vals]

    x = np.arange(len(variants))
    width = 0.38
    fig, ax = plt.subplots(figsize=(7.5, 4.4))

    C_OFF = "#1f5a8a"   # darker — thinkOff
    C_ON  = "#8ab8d8"   # lighter — thinkOn
    C_MISS = "#dddddd"  # placeholder for missing measurements

    off_bars = ax.bar(
        x - width/2,
        [v if v is not None else 0 for v in off_pct],
        width, label="thinkOff", color=C_OFF,
        edgecolor="black", linewidth=0.4,
    )
    on_bars = ax.bar(
        x + width/2,
        [v if v is not None else 0 for v in on_pct],
        width, label="thinkOn", color=C_ON,
        edgecolor="black", linewidth=0.4,
    )
    # Hatch + gray-out cells that are None (pending measurement).
    for bar, val in zip(off_bars, off_pct):
        if val is None:
            bar.set_color(C_MISS); bar.set_hatch("///")
    for bar, val in zip(on_bars, on_pct):
        if val is None:
            bar.set_color(C_MISS); bar.set_hatch("///")

    # Value labels above each bar.
    for bar, val in zip(list(off_bars) + list(on_bars), off_pct + on_pct):
        if val is None:
            txt, y = "n/a", 1.5
        else:
            txt, y = f"{val:.1f}", val + 0.5
        ax.text(bar.get_x() + bar.get_width() / 2, y,
                txt, ha="center", va="bottom", fontsize=9)

    if show_uniform:
        ax.axhline(10.0, color="gray", linestyle="--", linewidth=1,
                   label="uniform baseline (10 %)")

    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylabel("slot-1 top-1 %  (candidate position = 1 → rank 1)")
    ax.set_title(title, fontsize=11)
    ax.set_ylim(0, max(
        [v for v in off_pct + on_pct if v is not None] + [30]
    ) * 1.15)
    ax.grid(axis="y", linestyle=":", alpha=0.5)
    ax.legend(loc="upper left", frameon=False)

    # v4 choice marker — small arrow on B·thinkOff bar.
    b_off = off_pct[1]
    if b_off is not None:
        ax.annotate(
            "v4 teacher\n(chosen config)",
            xy=(x[1] - width/2, b_off),
            xytext=(x[1] - width/2 + 0.15, b_off + 6),
            fontsize=8.5, ha="left",
            arrowprops=dict(arrowstyle="->", color="#333", lw=0.8),
        )

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"wrote {out_path}")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", type=Path,
                    default=PROJECT_ROOT / "docs" / "assets" / "slot1_matrix.png")
    ap.add_argument("--canonical-v3-ona", type=float, default=0.1346,
                    help="historical A·thinkOn slot-1 rate (default 0.1346 from the canonical HTTP audit)")
    ap.add_argument("--v4-b-off", type=float, default=0.1394,
                    help="v4 teacher dataset B·thinkOff slot-1 (default 0.1394); pass -1 to fall back to 287-eval")
    ap.add_argument("--no-uniform-line", action="store_false", dest="uniform")
    args = ap.parse_args()

    v4_b = None if args.v4_b_off < 0 else args.v4_b_off
    cells = collect_cells(args.canonical_v3_ona, v4_b_off=v4_b)
    missing = [k for k, v in cells.items() if v is None]
    if missing:
        print(f"note: {len(missing)} cell(s) missing — rendered as hashed placeholders: {missing}")
    plot(cells, args.out, show_uniform=args.uniform)
    return 0


if __name__ == "__main__":
    sys.exit(main())

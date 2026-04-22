#!/usr/bin/env python
# ABOUTME: Read all data/results/eval_*.json + latency_summary*.json and emit a
# ABOUTME: unified comparison: COMPARISON.md, comparison.csv, comparison.html (plotly).

"""Build the unified evaluation dashboard for the post-deadline portfolio.

Reads every `data/results/eval_<tag>.json` (and optional latency summary), emits:

  - ``data/results/COMPARISON.md`` — a single readable markdown table with
    one row per (variant, quantization) and ΔR@1 attribution per stage.
  - ``data/results/comparison.csv`` — same table for spreadsheet ingestion.
  - ``data/results/comparison.html`` — plotly Pareto scatter (size × R@1)
    coloured by recipe group, hoverable per checkpoint. Suitable for
    embedding in Notion via raw HTML.

Variant taxonomy (post-deadline plan ``buzzing-conjuring-hare``):

  - ``teacher-qwen35``       — Qwen3.5-35B-A3B teacher reference (upper bound)
  - ``base-q35-0.8b``        — Qwen3.5-0.8B with no training (lower bound)
  - ``v1-merged``            — Qwen3-4B SFT (legacy reference, different model family)
  - ``v2-sft``               — Qwen3.5-0.8B full-FT SFT baseline
  - ``v2-sft-{w4a16,nf4,gguf-q4km}``         — PTQ on v2-sft
  - ``v2-gkd-warm``          — v2-sft warm-start → on-policy GKD
  - ``v2-gkd-warm-{w4a16,nf4,gguf-q4km}``    — PTQ on v2-gkd-warm
  - ``v2-gkd-cold``          — Qwen3.5-0.8B base → on-policy GKD (no SFT warmup)
  - ``v2-gkd-cold-{w4a16,nf4,gguf-q4km}``    — PTQ on v2-gkd-cold

Attribution table reports ΔR@1 between consecutive stages: SFT-vs-base,
GKD-warm-vs-SFT, GKD-cold-vs-base, and per PTQ-vs-FP16.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable

PROJECT_ROOT = Path(__file__).resolve().parents[2]
RESULTS_DIR = PROJECT_ROOT / "data" / "results"


# ---------- Data model ------------------------------------------------------


@dataclass
class EvalRow:
    """One row of the comparison table.

    Args:
        tag (str): unique identifier for the (variant, quantization) pair.
        recipe (str): training recipe group ("teacher" | "base" | "v1" | "v2-sft" |
            "v2-gkd-warm" | "v2-gkd-cold").
        quant (str): quantization method ("fp16" | "w4a16" | "nf4" | "gguf-q4km").
        size_gb (float | None): on-disk size in GB (None when unknown).
        r1 (float | None): Recall@1 on the deterministic eval split.
        r5 (float | None): Recall@5.
        r10 (float | None): Recall@10.
        mrr10 (float | None): MRR@10.
        ndcg5 (float | None): NDCG@5 (single-label: 1/log2(rank+1) if positive
            within top-5, else 0). Populated post-2026-04-21; earlier eval
            JSONs can be back-filled with ``scripts/eval/backfill_ndcg.py``.
        ndcg10 (float | None): NDCG@10 (same definition, top-10 cutoff).
        slot1_top1_rate (float | None): fraction of predictions whose
            top-ranked candidate is the candidate shown in slot 1 of the
            prompt — primary position-bias indicator (uniform ≈ 0.10).
            Read from ``position_bias.slot1_top1_rate`` which
            ``eval_metrics_vllm.py`` et al. emit.
        kendall_tau (float | None): rank correlation with the teacher.
        top1_agreement (float | None): top-1 student-vs-teacher agreement.
        parse_rate (float | None): fraction of student outputs that parsed as
            the expected JSON schema.
        ms_per_token (float | None): p50 ms/output-token from latency_summary.
        n_eval (int | None): number of eval samples used.
        source_path (Path | None): origin JSON file (for traceability).
    """

    tag: str
    recipe: str
    quant: str
    size_gb: float | None = None
    r1: float | None = None
    r5: float | None = None
    r10: float | None = None
    mrr10: float | None = None
    ndcg5: float | None = None
    ndcg10: float | None = None
    slot1_top1_rate: float | None = None
    kendall_tau: float | None = None
    top1_agreement: float | None = None
    parse_rate: float | None = None
    ms_per_token: float | None = None
    n_eval: int | None = None
    source_path: Path | None = None


# ---------- Parsing ---------------------------------------------------------


_QUANT_SUFFIX = re.compile(r"-(w4a16|nf4|gguf-q4km)$")


def classify_tag(tag: str) -> tuple[str, str]:
    """Map an eval tag to (recipe, quant).

    Examples:
        "v2-sft"            -> ("v2-sft", "fp16")
        "v2-sft-w4a16"      -> ("v2-sft", "w4a16")
        "v2-gkd-warm-nf4"   -> ("v2-gkd-warm", "nf4")
        "teacher-qwen35"    -> ("teacher", "n/a")
        "v1-merged"         -> ("v1", "fp16")

    Args:
        tag (str): the ``--tag`` value the eval script wrote into the JSON.

    Returns:
        tuple[str, str]: (recipe, quant).
    """
    if tag.startswith("teacher"):
        return ("teacher", "n/a")
    if tag.startswith("base"):
        return ("base", "fp16")
    if tag.startswith("v1"):
        m = _QUANT_SUFFIX.search(tag)
        if m:
            return ("v1", m.group(1))
        return ("v1", "fp16")

    # Legacy v0 tags (no v0 prefix — original Gemini-trained baseline) — name
    # appears as bare "merged", "w4a16", "nf4", "gguf-q4km" in old result files.
    if tag in {"merged", "w4a16", "nf4", "gguf-q4km"} or tag.endswith("-smoke"):
        if tag == "merged" or tag.endswith("merged"):
            return ("v0-legacy", "fp16")
        return ("v0-legacy", tag.replace("-smoke", ""))

    m = _QUANT_SUFFIX.search(tag)
    if m:
        recipe = tag[: m.start()]
        quant = m.group(1)
    else:
        recipe = tag
        quant = "fp16"
    return (recipe, quant)


def load_eval_jsons(results_dir: Path) -> list[EvalRow]:
    """Scan ``results_dir`` for ``eval_*.json`` files and parse each.

    The eval scripts (``eval_metrics.py``, ``eval_gguf.py``) write summary
    JSONs with the structure::

        {
          "tag": ...,
          "split": "eval",
          "n_samples": int,
          "model": str,
          "positive_metrics": {
            "student": {"recall@1": ..., "recall@5": ..., "mrr@10": ..., "n_evaluated": ...},
            "teacher": {"recall@1": ..., ...}
          },
          "parsing": {"total": ..., "valid": ..., "valid_rate": ...},
          "teacher_agreement": {"top1_agreement": ..., "kendall_tau_mean": ...}
        }

    Teacher-only runs lack the ``student`` block. This loader picks
    ``student`` when present, falling back to ``teacher`` (the upper-bound
    reference rows).

    Args:
        results_dir (Path): directory to scan (typically ``data/results/``).

    Returns:
        list[EvalRow]: one row per eval JSON file, in load order.
    """
    rows: list[EvalRow] = []
    for path in sorted(results_dir.glob("eval_*.json")):
        try:
            payload = json.loads(path.read_text())
        except json.JSONDecodeError:
            print(f"[warn] could not parse {path}; skipping", file=sys.stderr)
            continue

        tag = payload.get("tag") or path.stem.removeprefix("eval_")
        recipe, quant = classify_tag(tag)

        positive = payload.get("positive_metrics") or {}
        # Prefer student metrics; fall back to teacher (e.g. teacher-only eval).
        metrics = positive.get("student") or positive.get("teacher") or {}
        agreement = payload.get("teacher_agreement") or {}
        parsing = payload.get("parsing") or {}
        position_bias = payload.get("position_bias") or {}

        rows.append(
            EvalRow(
                tag=tag,
                recipe=recipe,
                quant=quant,
                r1=metrics.get("recall@1"),
                r5=metrics.get("recall@5"),
                r10=metrics.get("recall@10"),
                mrr10=metrics.get("mrr@10"),
                ndcg5=metrics.get("ndcg@5"),
                ndcg10=metrics.get("ndcg@10"),
                slot1_top1_rate=position_bias.get("slot1_top1_rate"),
                kendall_tau=agreement.get("kendall_tau_mean"),
                top1_agreement=agreement.get("top1_agreement"),
                parse_rate=parsing.get("valid_rate"),
                n_eval=metrics.get("n_evaluated") or payload.get("n_samples"),
                source_path=path,
            )
        )
    return rows


def attach_disk_sizes(rows: list[EvalRow]) -> None:
    """Fill in ``size_gb`` by stat-ing the corresponding ckpt directory.

    Maps each row's (recipe, quant) to ``ckpt/student-<recipe>-<quant>/`` (or
    ``ckpt/student-<recipe>-merged/`` for fp16). Sums file sizes recursively.
    Leaves ``size_gb`` as None if the directory does not exist (e.g. legacy
    v1 ckpt removed).

    Args:
        rows (list[EvalRow]): rows to mutate in place.
    """
    for row in rows:
        ckpt = _guess_ckpt_dir(row.recipe, row.quant)
        if ckpt is None or not ckpt.exists():
            continue
        total = sum(p.stat().st_size for p in ckpt.rglob("*") if p.is_file())
        row.size_gb = round(total / (1024**3), 3)


def _guess_ckpt_dir(recipe: str, quant: str) -> Path | None:
    """Return the conventional ckpt directory for a (recipe, quant) pair."""
    if recipe in {"teacher", "base"}:
        return None
    suffix = "merged" if quant in {"fp16", "n/a"} else quant
    return PROJECT_ROOT / "ckpt" / f"student-{recipe}-{suffix}"


def attach_latency(rows: list[EvalRow], latency_path: Path) -> None:
    """Read latency_summary.json (if present) and attach ms/tok to matching rows.

    Args:
        rows (list[EvalRow]): rows to mutate.
        latency_path (Path): typically ``data/results/latency_summary.json``.
    """
    if not latency_path.exists():
        return
    try:
        payload = json.loads(latency_path.read_text())
    except json.JSONDecodeError:
        return
    # latency_summary.json records per-backend p50 ms/tok under varying keys
    # depending on bench_latency.py version. Try a couple of layouts.
    by_tag: dict[str, float] = {}
    if isinstance(payload, dict):
        backends = payload.get("backends") or payload
        if isinstance(backends, dict):
            for tag, stats in backends.items():
                if isinstance(stats, dict):
                    val = stats.get("p50_ms_per_tok") or stats.get("p50_ms_per_token")
                    if val is not None:
                        by_tag[str(tag)] = float(val)
    for row in rows:
        if row.tag in by_tag:
            row.ms_per_token = by_tag[row.tag]


# ---------- Output renderers ------------------------------------------------


_COLUMNS = [
    ("tag", "Tag"),
    ("recipe", "Recipe"),
    ("quant", "Quant"),
    ("size_gb", "Size (GB)"),
    ("r1", "R@1"),
    ("r5", "R@5"),
    ("r10", "R@10"),
    ("mrr10", "MRR@10"),
    ("ndcg5", "NDCG@5"),
    ("ndcg10", "NDCG@10"),
    ("slot1_top1_rate", "Slot-1 top-1 %"),
    ("kendall_tau", "Kendall τ"),
    ("top1_agreement", "Top-1 vs Teacher"),
    ("parse_rate", "Parse rate"),
    ("ms_per_token", "ms/tok (p50)"),
    ("n_eval", "N eval"),
]


def render_markdown(rows: list[EvalRow], out_path: Path) -> None:
    """Write a Markdown table sorted by recipe then quant to ``out_path``."""
    ordered = sorted(rows, key=lambda r: (_recipe_order(r.recipe), _quant_order(r.quant), r.tag))
    header = "| " + " | ".join(label for _, label in _COLUMNS) + " |"
    sep = "|" + "|".join(["---"] * len(_COLUMNS)) + "|"
    body_lines = []
    for row in ordered:
        cells = []
        for attr, _label in _COLUMNS:
            cells.append(_fmt_cell(getattr(row, attr)))
        body_lines.append("| " + " | ".join(cells) + " |")

    attribution = _render_attribution(ordered)

    out_path.write_text(
        "# Comparison — `buzzing-conjuring-hare` portfolio dashboard\n\n"
        "Auto-generated by `scripts/eval/compare_results.py` from the JSON files in\n"
        "`data/results/`. Re-run after each new eval to refresh.\n\n"
        "## Per-checkpoint metrics\n\n"
        + header + "\n" + sep + "\n" + "\n".join(body_lines) + "\n\n"
        "## Attribution (ΔR@1 between stages)\n\n"
        + attribution + "\n",
        encoding="utf-8",
    )


def render_csv(rows: list[EvalRow], out_path: Path) -> None:
    """Write a CSV file (one row per checkpoint, all columns) to ``out_path``."""
    import csv

    ordered = sorted(rows, key=lambda r: (_recipe_order(r.recipe), _quant_order(r.quant), r.tag))
    with out_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow([label for _, label in _COLUMNS])
        for row in ordered:
            writer.writerow([_fmt_cell(getattr(row, attr), for_csv=True) for attr, _ in _COLUMNS])


def render_plotly(rows: list[EvalRow], out_path: Path) -> bool:
    """Write an interactive Plotly Pareto scatter (size × R@1) to ``out_path``.

    Returns True if plotly is available and the plot was written; False
    otherwise (plotly is an optional dependency — caller may skip).
    """
    try:
        import plotly.express as px
    except ImportError:
        print("[warn] plotly not installed; skipping HTML output", file=sys.stderr)
        return False

    plot_rows = [r for r in rows if r.r1 is not None and r.size_gb is not None]
    if not plot_rows:
        print("[warn] no rows with both size_gb and r@1; skipping HTML", file=sys.stderr)
        return False

    data = {
        "tag": [r.tag for r in plot_rows],
        "recipe": [r.recipe for r in plot_rows],
        "quant": [r.quant for r in plot_rows],
        "size_gb": [r.size_gb for r in plot_rows],
        "r1": [r.r1 for r in plot_rows],
        "mrr10": [r.mrr10 for r in plot_rows],
        "tau": [r.kendall_tau for r in plot_rows],
    }
    fig = px.scatter(
        data,
        x="size_gb",
        y="r1",
        color="recipe",
        symbol="quant",
        hover_data=["tag", "mrr10", "tau"],
        title="Pareto: model size (GB) vs Recall@1",
        labels={"size_gb": "On-disk size (GB)", "r1": "Recall@1"},
    )
    fig.write_html(str(out_path), include_plotlyjs="cdn")
    return True


# ---------- Helpers ---------------------------------------------------------


_RECIPE_RANK = {
    "teacher": 0, "base": 1, "v0-legacy": 2, "v1": 3, "v2-sft": 4,
    "v2-gkd-warm": 5, "v2-gkd-cold": 6,
}
_QUANT_RANK = {"n/a": 0, "fp16": 1, "w4a16": 2, "nf4": 3, "gguf-q4km": 4}


def _recipe_order(recipe: str) -> int:
    return _RECIPE_RANK.get(recipe, 99)


def _quant_order(quant: str) -> int:
    return _QUANT_RANK.get(quant, 99)


def _fmt_cell(value: Any, for_csv: bool = False) -> str:
    """Render a cell value: round floats, blank for None, str passthrough."""
    if value is None:
        return "" if for_csv else "—"
    if isinstance(value, float):
        return f"{value:.3f}" if abs(value) < 1000 else f"{value:.1f}"
    return str(value)


def _render_attribution(rows: list[EvalRow]) -> str:
    """Compute simple ΔR@1 attribution between selected stages."""
    by_tag = {r.tag: r for r in rows}
    pairs: list[tuple[str, str, str]] = [
        ("Off-policy SFT vs base (Qwen3.5-0.8B)", "v2-sft", "base-q35-0.8b"),
        ("On-policy GKD-warm vs SFT", "v2-gkd-warm", "v2-sft"),
        ("On-policy GKD-cold vs base (no warmup)", "v2-gkd-cold", "base-q35-0.8b"),
        ("SFT-warmup effect: GKD-warm vs GKD-cold", "v2-gkd-warm", "v2-gkd-cold"),
        ("Teacher headroom: teacher vs v2-gkd-warm", "teacher-qwen35", "v2-gkd-warm"),
        ("PTQ cost (W4A16): v2-sft-w4a16 vs v2-sft", "v2-sft-w4a16", "v2-sft"),
        ("PTQ cost (NF4): v2-sft-nf4 vs v2-sft", "v2-sft-nf4", "v2-sft"),
        ("PTQ cost (GGUF): v2-sft-gguf-q4km vs v2-sft", "v2-sft-gguf-q4km", "v2-sft"),
    ]
    lines = ["| Comparison | A | B | ΔR@1 (A − B) |", "|---|---|---|---|"]
    for label, a_tag, b_tag in pairs:
        a, b = by_tag.get(a_tag), by_tag.get(b_tag)
        delta_str = "—"
        if a and b and a.r1 is not None and b.r1 is not None:
            delta = a.r1 - b.r1
            delta_str = f"{delta:+.3f}"
        lines.append(f"| {label} | {a_tag} | {b_tag} | {delta_str} |")
    return "\n".join(lines)


# ---------- CLI -------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments.

    Returns:
        argparse.Namespace: parsed arguments.
    """
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--results-dir", type=Path, default=RESULTS_DIR,
        help="directory containing eval_*.json files (default: data/results)",
    )
    p.add_argument(
        "--out-md", type=Path, default=RESULTS_DIR / "COMPARISON.md",
        help="output Markdown path",
    )
    p.add_argument(
        "--out-csv", type=Path, default=RESULTS_DIR / "comparison.csv",
        help="output CSV path",
    )
    p.add_argument(
        "--out-html", type=Path, default=RESULTS_DIR / "comparison.html",
        help="output Plotly HTML path (skipped if plotly missing)",
    )
    p.add_argument(
        "--latency-json", type=Path, default=RESULTS_DIR / "latency_summary.json",
        help="latency summary JSON path (optional)",
    )
    p.add_argument(
        "--no-disk-sizes", action="store_true",
        help="skip ckpt directory disk-size lookup (faster)",
    )
    return p.parse_args()


def main() -> int:
    """Build the dashboard. Returns 0 on success, 2 on no rows."""
    args = parse_args()
    rows = load_eval_jsons(args.results_dir)
    if not rows:
        print(f"no eval_*.json files found in {args.results_dir}", file=sys.stderr)
        return 2
    if not args.no_disk_sizes:
        attach_disk_sizes(rows)
    attach_latency(rows, args.latency_json)

    args.out_md.parent.mkdir(parents=True, exist_ok=True)
    render_markdown(rows, args.out_md)
    render_csv(rows, args.out_csv)
    rendered_html = render_plotly(rows, args.out_html)

    print(f"wrote {args.out_md}")
    print(f"wrote {args.out_csv}")
    if rendered_html:
        print(f"wrote {args.out_html}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

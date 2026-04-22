#!/usr/bin/env python
# ABOUTME: Measure the true SFT sequence-length distribution (chat template +
# ABOUTME: JSON indent=2 completion) to pick max_length without silent truncation.

"""
Profile the tokenized length of every (prompt, completion) pair as it will be
seen by SFTTrainer, and report the distribution + drop-rate at candidate
max_length values.

Rationale:
    train_student.py drops (never truncates) examples that exceed max_length,
    to prevent silent truncation of the assistant JSON target. To pick the
    smallest max_length that still keeps ~100 % of training data, we need the
    tokenized length of the *rendered* chat template (system + user + assistant)
    with the same ``json.dumps(..., indent=2, ensure_ascii=False)`` formatting
    that ``teacher_output_to_assistant_text`` produces. This script reuses
    train_student.py's pure functions so the measurement matches training
    bit-for-bit.

Usage:
    python scripts/data/profile_token_distribution.py \\
        --tokenizer Qwen/Qwen3.5-0.8B \\
        --samples data/processed/philly_samples.jsonl \\
        --teacher data/teacher/philly_teacher_qwen35.jsonl

Notes:
    - Qwen3.5 family shares a single tokenizer (vocab 248320). Using the 0.8B
      repo as tokenizer source produces the same token ids as Qwen3.5-9B.
    - Output: text table + JSON file under data/results/ for later reference.
"""

from __future__ import annotations

import argparse
import bisect
import json
import logging
import sys
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.train.train_student import (  # noqa: E402
    build_training_example,
    load_and_filter,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("profile_tokens")


CANDIDATE_MAX_LENGTHS = [2048, 3072, 4096, 4608, 5120, 6144, 7168, 8192, 10240, 12288]


def percentile(sorted_values: list[int], p: float) -> int:
    """Return the p-th percentile from a pre-sorted list of ints (nearest-rank).

    Args:
        sorted_values (list[int]): ascending-sorted sequence of non-negative ints.
        p (float): percentile in [0, 100].

    Returns:
        int: the value at the nearest rank. For empty input, returns 0.
    """
    if not sorted_values:
        return 0
    n = len(sorted_values)
    # Nearest-rank method: rank = ceil(p/100 * n), 1-indexed.
    rank = max(1, int((p / 100.0) * n + 0.999999))
    rank = min(rank, n)
    return sorted_values[rank - 1]


def drop_rate_at(sorted_values: list[int], cap: int) -> tuple[int, float]:
    """Count values strictly greater than ``cap``.

    Args:
        sorted_values (list[int]): ascending-sorted token lengths.
        cap (int): candidate max_length.

    Returns:
        tuple: (count_over_cap, drop_fraction_in_[0,1]).
    """
    if not sorted_values:
        return 0, 0.0
    first_over = bisect.bisect_right(sorted_values, cap)
    over = len(sorted_values) - first_over
    return over, over / len(sorted_values)


def measure(
    examples: list[dict[str, Any]],
    tokenizer: Any,
) -> list[int]:
    """Tokenize the full rendered chat template for each example.

    The rendering mirrors train_student.py's SFTTrainer input path exactly:
    ``tokenizer.apply_chat_template(prompt + completion, tokenize=False,
    add_generation_prompt=False)`` followed by a plain ``tokenizer()`` encode.

    Args:
        examples (list[dict]): prompt-completion dicts built by
            ``build_training_example``.
        tokenizer: HuggingFace fast tokenizer.

    Returns:
        list[int]: token count per example, in the same order as input.
    """
    lengths: list[int] = []
    for i, ex in enumerate(examples):
        full_text = tokenizer.apply_chat_template(
            ex["prompt"] + ex["completion"],
            tokenize=False,
            add_generation_prompt=False,
        )
        ids = tokenizer(full_text, add_special_tokens=False)["input_ids"]
        lengths.append(len(ids))
        if (i + 1) % 500 == 0:
            log.info("  tokenized %d / %d", i + 1, len(examples))
    return lengths


def prompt_completion_split(
    examples: list[dict[str, Any]],
    tokenizer: Any,
) -> tuple[list[int], list[int]]:
    """Separate prompt-token and completion-token counts.

    Useful for understanding where headroom goes. Prompt is tokenized with
    ``add_generation_prompt=True`` (so the assistant-start marker is counted
    in the prompt side, matching completion_only_loss masking), completion is
    the remainder of the full rendering.

    Args:
        examples (list[dict]): prompt-completion dicts.
        tokenizer: HuggingFace fast tokenizer.

    Returns:
        tuple: (prompt_lengths, completion_lengths), same length as input.
    """
    p_lens: list[int] = []
    c_lens: list[int] = []
    for ex in examples:
        prompt_text = tokenizer.apply_chat_template(
            ex["prompt"],
            tokenize=False,
            add_generation_prompt=True,
        )
        full_text = tokenizer.apply_chat_template(
            ex["prompt"] + ex["completion"],
            tokenize=False,
            add_generation_prompt=False,
        )
        p_ids = tokenizer(prompt_text, add_special_tokens=False)["input_ids"]
        full_ids = tokenizer(full_text, add_special_tokens=False)["input_ids"]
        p_lens.append(len(p_ids))
        c_lens.append(max(0, len(full_ids) - len(p_ids)))
    return p_lens, c_lens


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--tokenizer",
        type=str,
        default="Qwen/Qwen3.5-0.8B",
        help=(
            "HF tokenizer id. Qwen3.5 family shares a single tokenizer "
            "(vocab 248320) so Qwen3.5-0.8B is sufficient to profile for "
            "Qwen3.5-9B and Qwen3.5-35B-A3B."
        ),
    )
    p.add_argument(
        "--samples",
        type=Path,
        default=PROJECT_ROOT / "data/processed/philly_samples.jsonl",
    )
    p.add_argument(
        "--teacher",
        type=Path,
        default=PROJECT_ROOT / "data/teacher/philly_teacher_qwen35.jsonl",
    )
    p.add_argument(
        "--output",
        type=Path,
        default=PROJECT_ROOT / "data/results/token_distribution.json",
        help="JSON dump of per-example lengths + summary stats.",
    )
    p.add_argument(
        "--no-split",
        action="store_true",
        help=(
            "skip the prompt/completion per-side breakdown (faster, single "
            "tokenization pass). The full-length distribution is always "
            "computed."
        ),
    )
    return p.parse_args()


def main() -> int:
    args = parse_args()

    from transformers import AutoTokenizer

    log.info("loading tokenizer %s", args.tokenizer)
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer, use_fast=True)

    log.info("join + validate teacher records ...")
    joined, stats = load_and_filter(args.samples, args.teacher)
    if not joined:
        log.error("no valid records after join+filter; aborting")
        return 2
    log.info("join+filter stats: %s", stats)

    log.info("building %d prompt-completion examples ...", len(joined))
    examples = [build_training_example(j["sample"], j["teacher"]) for j in joined]

    log.info("tokenizing full (prompt + completion) ...")
    full_lens = measure(examples, tokenizer)

    if not args.no_split:
        log.info("tokenizing prompt-only and computing completion deltas ...")
        p_lens, c_lens = prompt_completion_split(examples, tokenizer)
    else:
        p_lens, c_lens = [], []

    sorted_full = sorted(full_lens)
    sorted_prompt = sorted(p_lens)
    sorted_completion = sorted(c_lens)

    def summarize(sorted_vals: list[int]) -> dict[str, int | float]:
        """Compute p50/p90/p95/p99/p99.5/p99.9/max and mean.

        Args:
            sorted_vals (list[int]): ascending-sorted token lengths.

        Returns:
            dict: summary stats.
        """
        if not sorted_vals:
            return {}
        return {
            "n": len(sorted_vals),
            "min": sorted_vals[0],
            "p50": percentile(sorted_vals, 50),
            "p90": percentile(sorted_vals, 90),
            "p95": percentile(sorted_vals, 95),
            "p99": percentile(sorted_vals, 99),
            "p99.5": percentile(sorted_vals, 99.5),
            "p99.9": percentile(sorted_vals, 99.9),
            "max": sorted_vals[-1],
            "mean": sum(sorted_vals) / len(sorted_vals),
        }

    summary_full = summarize(sorted_full)
    summary_prompt = summarize(sorted_prompt) if sorted_prompt else {}
    summary_completion = summarize(sorted_completion) if sorted_completion else {}

    drop_table: list[dict[str, Any]] = []
    for cap in CANDIDATE_MAX_LENGTHS:
        over, frac = drop_rate_at(sorted_full, cap)
        drop_table.append({
            "max_length": cap,
            "dropped": over,
            "drop_fraction": frac,
            "kept": len(sorted_full) - over,
        })

    # Recommendation heuristic — smallest cap with drop_fraction == 0, else
    # smallest with drop_fraction ≤ 0.003 (0.3 %).
    zero_drop = [d["max_length"] for d in drop_table if d["dropped"] == 0]
    tight_drop = [d["max_length"] for d in drop_table if d["drop_fraction"] <= 0.003]
    recommended = zero_drop[0] if zero_drop else (tight_drop[0] if tight_drop else None)

    # ---- Text report ----
    print()
    print("=" * 74)
    print(f"Token length distribution   tokenizer={args.tokenizer}   n={len(full_lens)}")
    print("=" * 74)

    def fmt_row(label: str, s: dict[str, int | float]) -> str:
        if not s:
            return f"  {label:<16} (skipped)"
        return (
            f"  {label:<16} "
            f"p50={s['p50']:>5}  p95={s['p95']:>5}  p99={s['p99']:>5}  "
            f"p99.9={s['p99.9']:>5}  max={s['max']:>5}  mean={s['mean']:.0f}"
        )

    print(fmt_row("full seq", summary_full))
    print(fmt_row("prompt", summary_prompt))
    print(fmt_row("completion", summary_completion))
    print()

    print("drop rate at candidate max_length:")
    print(f"  {'max_length':>10}  {'dropped':>8}  {'drop %':>8}  {'kept':>6}")
    for d in drop_table:
        print(
            f"  {d['max_length']:>10}  {d['dropped']:>8}  "
            f"{d['drop_fraction'] * 100:>7.3f}%  {d['kept']:>6}"
        )
    print()
    print(f"recommended max_length: {recommended}")
    print()

    # ---- JSON dump ----
    args.output.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "tokenizer": args.tokenizer,
        "samples_path": str(args.samples),
        "teacher_path": str(args.teacher),
        "n": len(full_lens),
        "summary_full": summary_full,
        "summary_prompt": summary_prompt,
        "summary_completion": summary_completion,
        "drop_table": drop_table,
        "recommended_max_length": recommended,
        "lengths_full": full_lens,
        "lengths_prompt": p_lens,
        "lengths_completion": c_lens,
    }
    args.output.write_text(json.dumps(payload, ensure_ascii=False, indent=2))
    log.info("wrote %s", args.output)
    return 0


if __name__ == "__main__":
    sys.exit(main())

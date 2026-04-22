#!/usr/bin/env python
# ABOUTME: Convert eval_metrics_vllm raw.jsonl outputs into the inference-cache JSON
# ABOUTME: shape consumed by scripts/judge/judge_listwise.py (no re-inference needed).

"""Adapter: ``data/results/eval_*_raw.jsonl`` -> inference cache JSON.

``eval_metrics_vllm.py`` writes one JSON per completion into a streaming
``_raw.jsonl`` during evaluation. The record shape is::

    {
      "sample_id": "...",
      "user_id": "...",
      "positive_business_id": "...",
      "model": "<checkpoint path / url>",
      "eval_tag": "<backend tag>",
      "guided_decoding": bool,
      "raw_response": "<the model's JSON output string>",
      "teacher_output": {...},
      "finish_reason": "stop" | ...,
      "output_tokens": int,
      "error": null
    }

The listwise judge (``scripts/judge/judge_listwise.py``) expects the
inference-cache JSON shape produced by
``scripts/eval/generate_inference_samples.py``::

    {
      "backend": "<tag>",
      "model_path_or_url": "<model>",
      "dtype": "bf16",
      "generated_at": "<iso8601>",
      "samples": [
        {
          "sample_id": "...",
          "positive_business_id": "...",
          "output_text": "...",            # raw_response renamed
          "parsed_ranking": [int, ...] | null,
          "recovered_business_ids": [str, ...] | null,
          "json_parse_ok": bool,
          "output_tokens": int,
          "latency_sec": 0.0,
          "ms_per_output_token": 0.0
        },
        ...
      ]
    }

This script does the rename + derivation (``parsed_ranking``,
``recovered_business_ids``, ``json_parse_ok`` are computed from
``raw_response`` via the same ``summarize_output`` helper that the
canonical feeder uses, so the judge sees numerically identical fields).
It also filters out any sample_ids that the teacher-side
``load_and_filter`` drops (pre-existing errors, invalid schema, missing
business_id), to match the judge's eval-split definition.

Example::

    $ python scripts/judge/adapt_raw_to_cache.py \
        --raw data/results/eval_v4-gkd-guided-B-vllm-guided_raw.jsonl \
        --backend v4-gkd-guided-B-vllm-guided \
        --out data/inference_samples/v4-gkd-guided-B-vllm-guided.json
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.eval.generate_inference_samples import summarize_output  # noqa: E402
from scripts.train.train_student import load_and_filter  # noqa: E402

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("adapt_raw_to_cache")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--raw",
        type=Path,
        required=True,
        help="Input eval_<tag>_raw.jsonl (from eval_metrics_vllm.py).",
    )
    p.add_argument(
        "--backend",
        type=str,
        required=True,
        help=(
            "Backend tag to stamp on the output cache. This is the "
            "``by_backend`` key that judge_listwise.py --models will "
            "reference."
        ),
    )
    p.add_argument(
        "--out",
        type=Path,
        required=True,
        help="Output inference-cache JSON path.",
    )
    p.add_argument(
        "--samples",
        type=Path,
        default=PROJECT_ROOT / "data/processed/philly_samples.jsonl",
        help="Preprocessed samples JSONL (for candidate lookups).",
    )
    p.add_argument(
        "--teacher",
        type=Path,
        default=PROJECT_ROOT / "data/teacher/philly_teacher_qwen35.jsonl",
        help="Teacher JSONL used by load_and_filter to define the eval split.",
    )
    return p.parse_args()


def _iter_raw_records(path: Path):
    """Yield decoded JSON records from a JSONL file, skipping blanks.

    Args:
        path (Path): JSONL file path.

    Yields:
        dict: one decoded record per non-empty line.
    """
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            yield json.loads(line)


def main() -> int:
    args = parse_args()

    # Build sample_id -> preprocessed sample map so we can recover
    # candidates (needed by summarize_output to map ranking indices to
    # business_ids).
    examples, stats = load_and_filter(args.samples, args.teacher)
    log.info("join+filter stats: %s", stats)
    by_id: dict[str, dict[str, Any]] = {ex["sample_id"]: ex for ex in examples}

    # Stream raw.jsonl, derive cache records.
    out_samples: list[dict[str, Any]] = []
    n_total = 0
    n_dropped_filter = 0
    n_dropped_error = 0
    model_path: str | None = None

    for rec in _iter_raw_records(args.raw):
        n_total += 1
        if model_path is None:
            model_path = rec.get("model")

        if rec.get("error"):
            n_dropped_error += 1
            continue

        sid = rec["sample_id"]
        if sid not in by_id:
            n_dropped_filter += 1
            continue

        candidates = by_id[sid]["sample"]["candidates"]
        output_text = rec.get("raw_response") or ""
        summary = summarize_output(output_text, candidates)

        out_samples.append(
            {
                "sample_id": sid,
                "positive_business_id": rec.get("positive_business_id"),
                "output_text": output_text,
                "output_tokens": rec.get("output_tokens", 0),
                "latency_sec": 0.0,
                "ms_per_output_token": 0.0,
                **summary,
            }
        )

    log.info(
        "adapted %d/%d records (dropped: %d filtered, %d raw errors)",
        len(out_samples), n_total, n_dropped_filter, n_dropped_error,
    )

    cache = {
        "backend": args.backend,
        "model_path_or_url": model_path or "unknown",
        "dtype": "bf16",
        "generated_at": datetime.now(tz=timezone.utc).isoformat(),
        "source_raw_jsonl": str(args.raw.relative_to(PROJECT_ROOT)) if args.raw.is_relative_to(PROJECT_ROOT) else str(args.raw),
        "samples": out_samples,
    }

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(cache, indent=2, ensure_ascii=False))
    log.info("wrote %s", args.out)
    return 0


if __name__ == "__main__":
    sys.exit(main())

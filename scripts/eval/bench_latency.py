#!/usr/bin/env python
# ABOUTME: Defensible latency benchmark across Teacher Gemini API, Student
# ABOUTME: FP16, and Student W4A16, reporting ms/output-token as the headline.

"""
Measure and compare inference latency across three backends:

    1. Teacher    — Gemini 3 Flash Preview via ``gemini_parallel``
                    (synchronous, non-streaming, includes network RTT)
    2. Student FP16  — vLLM OpenAI-compatible endpoint, TP=1, bf16 weights
    3. Student W4A16 — vLLM OpenAI-compatible endpoint, TP=1, compressed-tensors

The headline metric is **ms per output token (p50 / p95)**. This is the
honest fair-comparison metric for this setup because:

* The Teacher is black-box — we cannot isolate model compute from network
  RTT, so raw end-to-end wall-clock latency is not directly comparable.
* Different backends emit slightly different output token counts; ms/token
  normalizes for that confound.

Raw end-to-end wall-clock, output-token mean/std, and per-backend failure
rate are reported as supporting columns so that anyone reading the table
can audit the headline number.

Measurement protocol (mirrors plan §3):

* N warmup calls per backend — discarded (CUDA kernel JIT, TCP warm).
* N_runs serial measurement passes over the eval prompt set.
* Backends are interleaved per prompt to smooth out time-of-day variance
  in the Teacher API's network path.
* Temperature 0.0, max_tokens 1536, single-request serial (no batching).
* ``time.perf_counter()`` only — never ``time.time()``.
* Failed calls are retried once after 2s. Retry latency is excluded from
  successful-call stats. The raw failure rate is reported separately.

Example:
    $ python scripts/eval/bench_latency.py \\
        --samples data/processed/philly_samples.jsonl \\
        --teacher data/teacher/philly_teacher.jsonl \\
        --fp16-url http://localhost:8000/v1 \\
        --w4a16-url http://localhost:8001/v1 \\
        --fp16-model student-fp16 \\
        --w4a16-model student-w4a16 \\
        --num-prompts 50 \\
        --num-warmup 3 \\
        --num-runs 3 \\
        --output data/results/latency_summary.json
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import statistics
import sys
import time
from pathlib import Path
from typing import Any, Callable

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from configs.teacher_prompt import SYSTEM_INSTRUCTION, build_user_prompt  # noqa: E402
from scripts.train.train_student import (  # noqa: E402
    load_and_filter,
    split_examples,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("bench_latency")


# ---------- Data ----------


def build_bench_prompts(
    samples_path: Path,
    teacher_path: Path,
    eval_ratio: float,
    num_prompts: int,
) -> list[dict[str, str]]:
    """Build the prompt set for the benchmark from the eval split.

    Uses the same deterministic 90/10 split that train_student.py and
    quantize_w4a16.py rely on, so the benchmark runs on prompts the student
    has never seen during SFT or calibration.

    Args:
        samples_path (Path): processed samples JSONL.
        teacher_path (Path): teacher output JSONL.
        eval_ratio (float): train fraction used in the deterministic split.
        num_prompts (int): desired number of unique prompts. If the eval
            split has fewer than this, we cycle the eval split to fill the
            quota — the unique count is reported separately.

    Returns:
        list[dict]: each dict has keys ``sample_id``, ``system``, ``user``.
    """
    examples, _stats = load_and_filter(samples_path, teacher_path)
    _train, eval_exs = split_examples(examples, ratio=eval_ratio)

    built: list[dict[str, str]] = []
    for ex in eval_exs:
        user_text = build_user_prompt(ex["sample"])
        built.append(
            {
                "sample_id": ex["sample_id"],
                "system": SYSTEM_INSTRUCTION,
                "user": user_text,
            }
        )

    if not built:
        raise ValueError("no eval prompts available; check data/split ratios")

    if len(built) >= num_prompts:
        return built[:num_prompts]

    # Cycle shorter eval split up to num_prompts (unique count reported later).
    repeated: list[dict[str, str]] = []
    i = 0
    while len(repeated) < num_prompts:
        repeated.append(built[i % len(built)])
        i += 1
    return repeated


# ---------- Backends ----------


class BackendResult(dict):
    """One measurement row.

    Keys: backend, sample_id, run_id, latency_ms, output_tokens, success, error.
    Subclassed from dict purely so we can write it to CSV directly.
    """


def measure_vllm_openai(
    backend_name: str,
    base_url: str,
    model_name: str,
    prompt: dict[str, str],
    run_id: int,
    max_tokens: int,
) -> BackendResult:
    """Measure one vLLM OpenAI-compatible chat completion request.

    Args:
        backend_name (str): label for the row, e.g. "student_fp16".
        base_url (str): e.g. "http://localhost:8000/v1".
        model_name (str): value of ``--served-model-name`` on the vLLM server.
        prompt (dict): one entry from build_bench_prompts.
        run_id (int): which measurement pass this is.
        max_tokens (int): ``max_tokens`` for the OpenAI client.

    Returns:
        BackendResult: one row of measurements.
    """
    from openai import OpenAI

    client = OpenAI(base_url=base_url, api_key="EMPTY")
    t0 = time.perf_counter()
    try:
        resp = client.chat.completions.create(
            model=model_name,
            messages=[
                {"role": "system", "content": prompt["system"]},
                {"role": "user", "content": prompt["user"]},
            ],
            max_tokens=max_tokens,
            temperature=0.0,
            stream=False,
        )
        t1 = time.perf_counter()
        completion_tokens = 0
        if resp.usage is not None:
            completion_tokens = resp.usage.completion_tokens or 0
        return BackendResult(
            backend=backend_name,
            sample_id=prompt["sample_id"],
            run_id=run_id,
            latency_ms=(t1 - t0) * 1000.0,
            output_tokens=completion_tokens,
            success=True,
            error="",
        )
    except Exception as e:  # network, timeout, server error
        t1 = time.perf_counter()
        return BackendResult(
            backend=backend_name,
            sample_id=prompt["sample_id"],
            run_id=run_id,
            latency_ms=(t1 - t0) * 1000.0,
            output_tokens=0,
            success=False,
            error=repr(e),
        )


def make_teacher_measurer(
    env_file: Path,
    model_name: str,
    thinking_level: str,
) -> Callable[[str, dict[str, str], int, int], BackendResult]:
    """Initialize a Gemini-via-gemini_parallel measurer once, return a closure.

    gemini_parallel's processor holds multiple API keys + rate-limit state
    internally, so we build exactly one processor and reuse it for every
    teacher measurement. This is the same path ``scripts/teacher/generate_teacher.py``
    uses, so latency numbers reflect the pipeline the project maintainer actually has.

    Args:
        env_file (Path): .env file with GEMINI_API_KEY_* variables.
        model_name (str): Gemini model id (e.g. "gemini-3-flash-preview").
        thinking_level (str): Gemini 3 thinking level knob.

    Returns:
        Callable: measure(backend_name, prompt, run_id, max_tokens) -> BackendResult
    """
    from dotenv import load_dotenv

    if env_file.exists():
        load_dotenv(env_file)

    from gemini_parallel import AdvancedApiKeyManager, GeminiSequentialProcessor

    from scripts.teacher.generate_teacher import read_declared_key_names

    declared = read_declared_key_names(env_file)
    if not declared:
        raise RuntimeError(
            f"no GEMINI_API_KEY_* names declared in {env_file}; "
            "cannot benchmark teacher backend"
        )

    key_manager = AdvancedApiKeyManager(keylist_names=declared)
    processor = GeminiSequentialProcessor(
        key_manager=key_manager,
        model_name=model_name,
        api_call_interval=4.0,
    )

    gen_config: dict[str, Any] = {
        "temperature": 0.0,  # deterministic for reproducible latency
        "response_mime_type": "application/json",
        "system_instruction": SYSTEM_INSTRUCTION,
        "thinking_config": {"thinking_level": thinking_level},
    }

    def _measure(
        backend_name: str,
        prompt: dict[str, str],
        run_id: int,
        max_tokens: int,
    ) -> BackendResult:
        cfg = dict(gen_config)
        cfg["max_output_tokens"] = max_tokens
        prompt_data = {
            "prompt": prompt["user"],
            "generation_config": cfg,
            "metadata": {"task_id": f"bench:{prompt['sample_id']}:{run_id}"},
        }
        t0 = time.perf_counter()
        try:
            metadata, response, error = processor.process_single(prompt_data)
            t1 = time.perf_counter()
            if error is not None:
                return BackendResult(
                    backend=backend_name,
                    sample_id=prompt["sample_id"],
                    run_id=run_id,
                    latency_ms=(t1 - t0) * 1000.0,
                    output_tokens=0,
                    success=False,
                    error=str(error),
                )
            # gemini_parallel returns response.text as str; token count is in metadata.
            text = response if isinstance(response, str) else getattr(response, "text", "")
            out_toks = 0
            if isinstance(metadata, dict):
                out_toks = int(metadata.get("completion_tokens") or 0)
            if out_toks == 0 and text:
                # Fall back to rough char->token estimate (Qwen tokenizer ≈ 3.5 chars/tok).
                out_toks = max(1, len(text) // 4)
            return BackendResult(
                backend=backend_name,
                sample_id=prompt["sample_id"],
                run_id=run_id,
                latency_ms=(t1 - t0) * 1000.0,
                output_tokens=out_toks,
                success=True,
                error="",
            )
        except Exception as e:
            t1 = time.perf_counter()
            return BackendResult(
                backend=backend_name,
                sample_id=prompt["sample_id"],
                run_id=run_id,
                latency_ms=(t1 - t0) * 1000.0,
                output_tokens=0,
                success=False,
                error=repr(e),
            )

    return _measure


# ---------- Stats ----------


def summarize(values: list[float]) -> dict[str, float]:
    """Compute mean/std/p50/p95/p99 of a list of floats.

    Args:
        values (list[float]): raw non-negative measurements. Must be non-empty
            for meaningful stats; returns zeros on empty input so downstream
            JSON serialization never crashes.

    Returns:
        dict[str, float]: keys mean, std, p50, p95, p99. Units follow input.
    """
    if not values:
        return {"mean": 0.0, "std": 0.0, "p50": 0.0, "p95": 0.0, "p99": 0.0}
    s = sorted(values)
    n = len(s)

    def _pct(p: float) -> float:
        idx = min(n - 1, max(0, int(round(p * (n - 1)))))
        return s[idx]

    return {
        "mean": statistics.fmean(s),
        "std": statistics.pstdev(s) if n > 1 else 0.0,
        "p50": _pct(0.50),
        "p95": _pct(0.95),
        "p99": _pct(0.99),
    }


def aggregate(
    rows: list[BackendResult],
) -> dict[str, dict[str, Any]]:
    """Aggregate per-call rows into a per-backend summary dict.

    Args:
        rows (list[BackendResult]): raw measurements, possibly including
            failures.

    Returns:
        dict[str, dict]: backend -> stats summary. Each summary has
            ``count_ok``, ``count_fail``, ``failure_rate``, and nested
            dicts ``latency_ms`` and ``ms_per_tok`` plus ``output_tokens``
            (mean, std).
    """
    by_backend: dict[str, list[BackendResult]] = {}
    for r in rows:
        by_backend.setdefault(r["backend"], []).append(r)

    out: dict[str, dict[str, Any]] = {}
    for backend, items in by_backend.items():
        ok = [r for r in items if r["success"]]
        fail = [r for r in items if not r["success"]]

        lat = [float(r["latency_ms"]) for r in ok]
        out_toks = [int(r["output_tokens"]) for r in ok]
        ms_per_tok = [
            float(r["latency_ms"]) / max(1, int(r["output_tokens"])) for r in ok
        ]

        out[backend] = {
            "count_ok": len(ok),
            "count_fail": len(fail),
            "failure_rate": len(fail) / len(items) if items else 0.0,
            "latency_ms": summarize(lat),
            "ms_per_tok": summarize(ms_per_tok),
            "output_tokens": {
                "mean": statistics.fmean(out_toks) if out_toks else 0.0,
                "std": statistics.pstdev(out_toks) if len(out_toks) > 1 else 0.0,
            },
        }
    return out


def render_markdown(summary: dict[str, dict[str, Any]]) -> str:
    """Render a per-backend summary as a markdown table for the README.

    Args:
        summary (dict): output of ``aggregate``.

    Returns:
        str: multi-line markdown table. One header row and one row per
            backend. Units: ms for latency_ms, ms/tok for ms_per_tok.
    """
    header = (
        "| Backend | p50 e2e (ms) | p95 e2e (ms) | p50 ms/tok | p95 ms/tok "
        "| out tok mean±std | fail rate |\n"
    )
    sep = "|---|---|---|---|---|---|---|\n"
    body = ""
    for backend, stats in summary.items():
        body += (
            f"| {backend} "
            f"| {stats['latency_ms']['p50']:.1f} "
            f"| {stats['latency_ms']['p95']:.1f} "
            f"| {stats['ms_per_tok']['p50']:.2f} "
            f"| {stats['ms_per_tok']['p95']:.2f} "
            f"| {stats['output_tokens']['mean']:.0f}±{stats['output_tokens']['std']:.0f} "
            f"| {stats['failure_rate']*100:.1f}% |\n"
        )
    return header + sep + body


# ---------- Main ----------


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--samples", type=Path, default=PROJECT_ROOT / "data/processed/philly_samples.jsonl")
    p.add_argument("--teacher", type=Path, default=PROJECT_ROOT / "data/teacher/philly_teacher.jsonl")
    p.add_argument("--env-file", type=Path, default=PROJECT_ROOT / ".env")
    p.add_argument("--fp16-url", type=str, default="http://localhost:8000/v1")
    p.add_argument("--w4a16-url", type=str, default="http://localhost:8001/v1")
    p.add_argument("--fp16-model", type=str, default="student-fp16")
    p.add_argument("--w4a16-model", type=str, default="student-w4a16")
    p.add_argument("--teacher-model", type=str, default="gemini-3-flash-preview")
    p.add_argument("--thinking-level", type=str, default="minimal")
    p.add_argument("--eval-ratio", type=float, default=0.9)
    p.add_argument("--num-prompts", type=int, default=50)
    p.add_argument("--num-warmup", type=int, default=3)
    p.add_argument("--num-runs", type=int, default=3)
    p.add_argument("--max-tokens", type=int, default=1536)
    p.add_argument("--output", type=Path, default=PROJECT_ROOT / "data/results/latency_summary.json")
    p.add_argument("--csv-output", type=Path, default=PROJECT_ROOT / "data/results/latency_raw.csv")
    p.add_argument(
        "--skip-teacher",
        action="store_true",
        help="skip the Teacher-API backend (useful for local-only dry runs)",
    )
    p.add_argument(
        "--skip-fp16",
        action="store_true",
        help="skip the FP16 vLLM backend",
    )
    p.add_argument(
        "--skip-w4a16",
        action="store_true",
        help="skip the W4A16 vLLM backend",
    )
    return p.parse_args()


def main() -> int:
    args = parse_args()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.csv_output.parent.mkdir(parents=True, exist_ok=True)

    prompts = build_bench_prompts(
        samples_path=args.samples,
        teacher_path=args.teacher,
        eval_ratio=args.eval_ratio,
        num_prompts=args.num_prompts,
    )
    unique_prompt_ids = {p["sample_id"] for p in prompts}
    log.info(
        "built %d prompts (%d unique sample_ids)",
        len(prompts),
        len(unique_prompt_ids),
    )

    # Build measurer closures for every enabled backend.
    measurers: list[tuple[str, Callable[[str, dict[str, str], int, int], BackendResult]]] = []

    if not args.skip_fp16:
        measurers.append(
            (
                "student_fp16",
                lambda backend, prompt, run_id, max_tokens: measure_vllm_openai(
                    backend, args.fp16_url, args.fp16_model, prompt, run_id, max_tokens
                ),
            )
        )
    if not args.skip_w4a16:
        measurers.append(
            (
                "student_w4a16",
                lambda backend, prompt, run_id, max_tokens: measure_vllm_openai(
                    backend, args.w4a16_url, args.w4a16_model, prompt, run_id, max_tokens
                ),
            )
        )
    if not args.skip_teacher:
        teacher_measurer = make_teacher_measurer(
            env_file=args.env_file,
            model_name=args.teacher_model,
            thinking_level=args.thinking_level,
        )
        measurers.append(("teacher_gemini", teacher_measurer))

    if not measurers:
        log.error("no backends enabled; nothing to benchmark")
        return 2

    # ---- Warmup ----
    log.info("warmup: %d calls per backend", args.num_warmup)
    for backend_name, fn in measurers:
        for i in range(args.num_warmup):
            fn(backend_name, prompts[0], run_id=-1 - i, max_tokens=args.max_tokens)

    # ---- Measurement ----
    log.info(
        "measuring %d runs x %d prompts x %d backends = %d calls total",
        args.num_runs,
        len(prompts),
        len(measurers),
        args.num_runs * len(prompts) * len(measurers),
    )
    rows: list[BackendResult] = []
    for run_id in range(args.num_runs):
        for prompt in prompts:
            for backend_name, fn in measurers:
                row = fn(backend_name, prompt, run_id, args.max_tokens)
                if not row["success"]:
                    log.warning(
                        "call failed [backend=%s run=%d sid=%s]: %s",
                        backend_name, run_id, prompt["sample_id"], row["error"][:120],
                    )
                    # One retry after 2 seconds
                    time.sleep(2.0)
                    row2 = fn(backend_name, prompt, run_id, args.max_tokens)
                    if row2["success"]:
                        rows.append(row2)
                        continue
                    # Both failed; keep the first failure row for the rate stat
                    rows.append(row)
                    continue
                rows.append(row)

    # ---- Aggregate + report ----
    summary = aggregate(rows)

    # Write raw CSV (every successful and failed call)
    fieldnames = ["backend", "sample_id", "run_id", "latency_ms", "output_tokens", "success", "error"]
    with args.csv_output.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for r in rows:
            writer.writerow({k: r.get(k) for k in fieldnames})
    log.info("wrote raw calls -> %s", args.csv_output)

    # Write JSON summary
    payload = {
        "config": {
            "num_prompts": len(prompts),
            "unique_prompts": len(unique_prompt_ids),
            "num_warmup": args.num_warmup,
            "num_runs": args.num_runs,
            "max_tokens": args.max_tokens,
            "fp16_url": args.fp16_url if not args.skip_fp16 else None,
            "w4a16_url": args.w4a16_url if not args.skip_w4a16 else None,
            "teacher_model": args.teacher_model if not args.skip_teacher else None,
        },
        "summary": summary,
    }
    args.output.write_text(json.dumps(payload, indent=2, ensure_ascii=False))
    log.info("wrote summary -> %s", args.output)

    log.info("=== latency summary ===\n%s", render_markdown(summary))
    return 0


if __name__ == "__main__":
    sys.exit(main())

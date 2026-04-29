# ABOUTME: Parallel LLM-as-a-Judge driver using gemini_parallel multi-key rotation
# ABOUTME: + ThreadPool fan-out so 25 free-tier keys run concurrently on the P2 ablation.
"""Parallel-Gemini variant of judge_listwise.py.

This driver is thin: it imports prompt builders, the Pydantic schema, and
the inference-cache loader from ``judge_listwise.py``, then runs the
per-pair loop through a ThreadPoolExecutor where each worker owns its
own ``GeminiSequentialProcessor``. All workers share one
``AdvancedApiKeyManager`` so that key rotation, exhaustion bookkeeping,
and IP-ban detection are coherent across threads.

The .env layout this script targets:

    /workspace/projects/.env              -> 25× GEMINI_API_KEY_*  (free tier)
    /workspace/projects/LLM_distillation/.env -> GOOGLE_API_KEY     (paid)

We load BOTH (parent first, then project, with project values *not*
overriding the parent because the parent is where the rotation keys
live). The driver uses only the GEMINI_API_KEY_* set since
``AdvancedApiKeyManager`` is built around named-key rotation.

Usage::

    python scripts/judge/judge_listwise_parallel.py \\
        --inference-cache data/inference_samples/teacher-p2-rationale-swapped.json \\
        --models teacher-p2-rationale-swapped \\
        --raw data/results/judge_listwise_raw_v3.1_thinkLow_parallel.jsonl \\
        --thinking-level LOW \\
        --max-output-tokens 32768 \\
        --workers 8

The judge prompt and rubric come from
``scripts/judge/judge_listwise.py`` — this file does not duplicate the
template.
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.judge.judge_listwise import (  # noqa: E402
    JUDGE_SYSTEM_LISTWISE,
    ListwiseVerdict,
    RUBRIC_VERSION,
    aggregate_per_model,
    build_judge_prompt_listwise,
    load_done_keys,
    load_inference_cache,
    parse_model_output,
    pick_eval_samples,
)
from scripts.train.train_student import load_and_filter, split_examples  # noqa: E402
from scripts.teacher.generate_teacher import read_declared_key_names  # noqa: E402

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("judge_parallel")

DEFAULT_RAW = PROJECT_ROOT / f"data/results/judge_listwise_raw_{RUBRIC_VERSION}_parallel.jsonl"
DEFAULT_SUMMARY = PROJECT_ROOT / f"data/results/judge_listwise_summary_{RUBRIC_VERSION}_parallel.json"
DEFAULT_PROJECT_ENV = PROJECT_ROOT / ".env"
DEFAULT_PARENT_ENV = PROJECT_ROOT.parent / ".env"


def merge_envs(env_files: list[Path]) -> list[str]:
    """Load multiple .env files and return GEMINI_API_KEY_* names from all of them.

    Args:
        env_files (list[Path]): paths to .env files. Files later in the
            list do NOT override earlier ones (we use ``override=False``)
            so the rotation keys defined in the first file (the parent
            ``.env`` by convention) are stable.

    Returns:
        list[str]: deduplicated list of GEMINI_API_KEY_* variable names
            actually present in os.environ after loading. Order follows
            first declaration across the files.
    """
    from dotenv import load_dotenv

    seen: set[str] = set()
    declared_in_order: list[str] = []
    for path in env_files:
        if not path.exists():
            log.warning("env file missing, skipping: %s", path)
            continue
        load_dotenv(path, override=False)
        for name in read_declared_key_names(path):
            if name not in seen:
                seen.add(name)
                declared_in_order.append(name)
    # Final filter: only keep names that resolved to a value in os.environ.
    resolved = [n for n in declared_in_order if os.environ.get(n)]
    if not resolved:
        raise RuntimeError(
            f"no GEMINI_API_KEY_* values resolved after loading {env_files}; "
            f"check the parent .env path"
        )
    return resolved


def build_prompt_dicts(
    eval_exs: list[dict[str, Any]],
    cache: dict[str, dict[str, Any]],
    requested_models: list[str],
    done: set[tuple[str, str]],
    generation_config: Any,
) -> tuple[list[dict[str, Any]], int, int]:
    """Build the parallel-ready prompt list, skipping resume hits and parse failures.

    Args:
        eval_exs (list[dict]): the picked eval examples (with ``sample_id``
            + ``sample`` payload).
        cache (dict): inference cache indexed by sample_id.
        requested_models (list[str]): model_tag values to score.
        done (set[tuple[str, str]]): already-scored (sample_id, model_tag).
        generation_config: the GenerateContentConfig instance to attach
            to every prompt.

    Returns:
        tuple: (prompt_dicts, n_skipped_resume, n_parse_fail). prompt_dicts
            each have keys ``prompt`` (str), ``metadata`` (dict),
            ``generation_config`` (GenerateContentConfig). Metadata
            carries ``sample_id`` and ``model_tag`` so the worker can
            attribute the result.
    """
    out: list[dict[str, Any]] = []
    n_skipped = 0
    n_parse_fail = 0
    for ex in eval_exs:
        sid = ex["sample_id"]
        sample = ex["sample"]
        entry = cache.get(sid)
        if not entry:
            continue
        for tag in requested_models:
            if (sid, tag) in done:
                n_skipped += 1
                continue
            backend = (entry.get("by_backend") or {}).get(tag)
            if backend is None:
                continue
            parsed = parse_model_output(backend.get("output_text", ""))
            if parsed is None:
                n_parse_fail += 1
                continue
            user_prompt = build_judge_prompt_listwise(sample, parsed)
            out.append({
                "prompt": user_prompt,
                "metadata": {
                    "task_id": f"{sid}::{tag}",
                    "sample_id": sid,
                    "model_tag": tag,
                },
                "generation_config": generation_config,
            })
    return out, n_skipped, n_parse_fail


def make_thinking_config(level: str, genai_types) -> Any:
    """Build a ThinkingConfig for the requested level (case-insensitive name).

    Args:
        level (str): one of OFF / MINIMAL / LOW / MEDIUM / HIGH.
        genai_types: ``google.genai.types`` module.

    Returns:
        google.genai.types.ThinkingConfig instance.
    """
    attr = (level or "MINIMAL").upper()
    if not hasattr(genai_types.ThinkingLevel, attr):
        raise ValueError(
            f"unknown thinking_level={level!r}; valid: OFF / MINIMAL / LOW / MEDIUM / HIGH"
        )
    return genai_types.ThinkingConfig(thinking_level=getattr(genai_types.ThinkingLevel, attr))


def submit_one(
    processor: Any,
    prompt_data: dict[str, Any],
) -> tuple[str, str, dict[str, Any]]:
    """Run one judge call through the given processor and assemble the verdict record.

    Args:
        processor: a thread-local ``GeminiSequentialProcessor``.
        prompt_data (dict): contains the prompt + metadata + generation_config.

    Returns:
        tuple: (sample_id, model_tag, verdict_record). The verdict_record
            mirrors the schema ``judge_listwise.py`` writes: a flat dict
            with ``sample_id``, ``model_tag``, ``rubric_version``, the
            three axes + their evidence strings, and ``error``.
    """
    sid = prompt_data["metadata"]["sample_id"]
    tag = prompt_data["metadata"]["model_tag"]
    metadata, response, error = processor.process_single(prompt_data)
    if error:
        return sid, tag, {
            "sample_id": sid,
            "model_tag": tag,
            "rubric_version": RUBRIC_VERSION,
            "groundedness": None,
            "personalization": None,
            "ranking_coherence": None,
            "groundedness_evidence": "",
            "personalization_evidence": "",
            "ranking_coherence_evidence": "",
            "error": error,
        }
    text = response if isinstance(response, str) else getattr(response, "text", None)
    text = (text or "").strip()
    if text.startswith("```"):
        text = text.strip("`")
        if text.lower().startswith("json"):
            text = text[4:].lstrip()
    try:
        parsed = json.loads(text) if text else {}
    except json.JSONDecodeError as e:
        return sid, tag, {
            "sample_id": sid,
            "model_tag": tag,
            "rubric_version": RUBRIC_VERSION,
            "groundedness": None,
            "personalization": None,
            "ranking_coherence": None,
            "groundedness_evidence": "",
            "personalization_evidence": "",
            "ranking_coherence_evidence": "",
            "error": f"json decode failed: {e}",
        }
    try:
        verdict = ListwiseVerdict.model_validate(parsed)
    except Exception as e:
        return sid, tag, {
            "sample_id": sid,
            "model_tag": tag,
            "rubric_version": RUBRIC_VERSION,
            "groundedness": None,
            "personalization": None,
            "ranking_coherence": None,
            "groundedness_evidence": "",
            "personalization_evidence": "",
            "ranking_coherence_evidence": "",
            "error": f"schema validation failed: {e}",
        }
    record: dict[str, Any] = {
        "sample_id": sid,
        "model_tag": tag,
        "rubric_version": RUBRIC_VERSION,
        "groundedness": verdict.groundedness,
        "personalization": verdict.personalization,
        "ranking_coherence": verdict.ranking_coherence,
        "groundedness_evidence": verdict.groundedness_evidence,
        "personalization_evidence": verdict.personalization_evidence,
        "ranking_coherence_evidence": verdict.ranking_coherence_evidence,
        "error": None,
    }
    return sid, tag, record


def main(argv: list[str] | None = None) -> int:
    """CLI entrypoint. Mirrors judge_listwise.py main() but runs in parallel.

    Args:
        argv (list[str] | None): forwarded to argparse.

    Returns:
        int: exit code.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--samples",
        type=Path,
        default=PROJECT_ROOT / "data/processed/philly_samples.jsonl",
    )
    parser.add_argument(
        "--teacher",
        type=Path,
        default=PROJECT_ROOT / "data/teacher/philly_teacher_qwen35.jsonl",
    )
    parser.add_argument(
        "--inference-cache",
        type=Path,
        action="append",
        default=None,
    )
    parser.add_argument("--models", type=str, default="teacher,v2-sft,v2-sft-w4a16")
    parser.add_argument("--env-file", type=Path, default=DEFAULT_PROJECT_ENV)
    parser.add_argument("--parent-env-file", type=Path, default=DEFAULT_PARENT_ENV)
    parser.add_argument("--judge-model", type=str, default="gemini-3-flash-preview")
    parser.add_argument(
        "--api-call-interval",
        type=float,
        default=0.5,
        help="per-worker minimum interval between calls; with N workers the global rate is N/interval",
    )
    parser.add_argument(
        "--max-output-tokens",
        type=int,
        default=32768,
        help=(
            "Output token cap. Set high (32768) so MEDIUM/HIGH thinking + "
            "the v3.1 scratchpad mandate cannot truncate the JSON verdict."
        ),
    )
    parser.add_argument("--eval-ratio", type=float, default=0.9)
    parser.add_argument("--n", type=int, default=50)
    parser.add_argument("--raw", type=Path, default=DEFAULT_RAW)
    parser.add_argument("--summary", type=Path, default=DEFAULT_SUMMARY)
    parser.add_argument(
        "--thinking-level",
        type=str,
        default="MINIMAL",
        help="ThinkingLevel: OFF / MINIMAL / LOW / MEDIUM / HIGH.",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=8,
        help="ThreadPool size. Capped at the number of declared keys.",
    )
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)

    args.raw.parent.mkdir(parents=True, exist_ok=True)
    args.summary.parent.mkdir(parents=True, exist_ok=True)
    log.info("rubric version: %s  raw: %s", RUBRIC_VERSION, args.raw)

    # Load + merge envs.
    declared = merge_envs([args.parent_env_file, args.env_file])
    log.info("loaded %d rotation keys: %s", len(declared), declared[:6] + (["..."] if len(declared) > 6 else []))

    # Resolve eval split and cache.
    requested_models = [m.strip() for m in args.models.split(",") if m.strip()]
    if not requested_models:
        log.error("no models requested; pass --models")
        return 2

    cache_paths: list[Path] = args.inference_cache or [
        PROJECT_ROOT / "data/inference_samples/all_backends_merged.json"
    ]
    examples, _ = load_and_filter(args.samples, args.teacher)
    _train, eval_exs = split_examples(examples, ratio=args.eval_ratio)
    eval_exs = pick_eval_samples(eval_exs, args.n)
    eval_by_id = {ex["sample_id"]: ex for ex in eval_exs}
    cache = load_inference_cache(cache_paths)
    log.info("eval pool: %d records  cache: %d sample_ids", len(eval_by_id), len(cache))

    missing = [sid for sid in eval_by_id if sid not in cache]
    if missing:
        log.error("%d/%d samples missing from cache (e.g. %s)", len(missing), len(eval_by_id), missing[:3])
        return 3

    available_tags: set[str] = set()
    for entry in cache.values():
        for tag in (entry.get("by_backend") or {}):
            available_tags.add(tag)
    unknown = [m for m in requested_models if m not in available_tags]
    if unknown:
        log.error("requested models %s not present in cache (have: %s)", unknown, sorted(available_tags))
        return 3

    done = load_done_keys(args.raw)
    if done:
        log.info("resume: %d (sample_id, model_tag) already scored", len(done))

    # Build generation config (must be a real GenerateContentConfig instance
    # so the schema + system instruction survive the gemini_parallel
    # snake/camel conversion logic).
    from google.genai import types as genai_types

    gen_config = genai_types.GenerateContentConfig(
        temperature=0.0,
        response_mime_type="application/json",
        response_schema=ListwiseVerdict,
        system_instruction=JUDGE_SYSTEM_LISTWISE,
        thinking_config=make_thinking_config(args.thinking_level, genai_types),
        max_output_tokens=args.max_output_tokens,
    )

    prompt_dicts, n_skip, n_parse_fail = build_prompt_dicts(
        eval_exs, cache, requested_models, done, gen_config
    )
    n_total = len(prompt_dicts)
    log.info(
        "ready to dispatch: %d prompts (resume_skipped=%d, parse_fail=%d)",
        n_total, n_skip, n_parse_fail,
    )

    if args.dry_run:
        log.info("dry-run: not calling Gemini")
        return 0
    if n_total == 0:
        log.info("nothing to do; exiting")
        return 0

    # Build one shared key manager + per-worker processors so each worker
    # can hold its own outbound connection while sharing key state.
    from gemini_parallel import AdvancedApiKeyManager, GeminiSequentialProcessor  # noqa: E402

    key_manager = AdvancedApiKeyManager(keylist_names=declared)
    log.info("key manager: %s", key_manager.get_keys_status_summary())

    n_workers = max(1, min(args.workers, len(declared)))
    log.info("dispatching with %d workers", n_workers)

    write_lock = threading.Lock()
    f_raw = args.raw.open("a", encoding="utf-8")

    def make_processor() -> Any:
        return GeminiSequentialProcessor(
            key_manager=key_manager,
            model_name=args.judge_model,
            api_call_interval=args.api_call_interval,
        )

    # Each thread keeps its own processor in a thread-local container.
    tls = threading.local()

    def worker(prompt_data: dict[str, Any]) -> tuple[str, str, dict[str, Any]]:
        if not hasattr(tls, "processor"):
            tls.processor = make_processor()
        return submit_one(tls.processor, prompt_data)

    n_scored = 0
    n_err = 0
    t_start = time.time()
    with ThreadPoolExecutor(max_workers=n_workers) as pool:
        futures = [pool.submit(worker, pd) for pd in prompt_dicts]
        for fut in as_completed(futures):
            sid, tag, record = fut.result()
            with write_lock:
                f_raw.write(json.dumps(record, ensure_ascii=False) + "\n")
                f_raw.flush()
            if record.get("error"):
                n_err += 1
            else:
                n_scored += 1
            if (n_scored + n_err) % 10 == 0 or (n_scored + n_err) == n_total:
                log.info(
                    "progress: %d/%d  (scored=%d err=%d wall=%ds)",
                    n_scored + n_err, n_total, n_scored, n_err, int(time.time() - t_start),
                )
    f_raw.close()

    # Re-emit summary using the same aggregator as the sequential judge.
    log.info(
        "wrote %d new verdicts -> %s (scored=%d, err=%d)",
        n_total, args.raw, n_scored, n_err,
    )
    # Read all written verdicts and aggregate via the same aggregator the
    # sequential judge uses (so summary JSON shape stays compatible).
    all_verdicts: list[dict[str, Any]] = []
    with args.raw.open("r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                all_verdicts.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    summary = aggregate_per_model(all_verdicts)
    args.summary.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    log.info("wrote summary -> %s", args.summary)
    for tag, s in summary.items():
        g = s.get("groundedness")
        if g and g.get("mean") is not None:
            log.info(
                "%s groundedness=%.2f [%.2f, %.2f]  n_scored=%d/%d",
                tag, g["mean"], g.get("ci_lo", float('nan')), g.get("ci_hi", float('nan')),
                s.get("n_scored", 0), s.get("n_total", 0),
            )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

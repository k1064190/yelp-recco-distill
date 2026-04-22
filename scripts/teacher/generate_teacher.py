#!/usr/bin/env python
# ABOUTME: Generate Teacher outputs (persona + rationales + ranking) from a
# ABOUTME: preprocessed Yelp sample JSONL using Gemini via gemini_parallel.

"""
Call a Teacher LLM (default: gemini-3-flash-preview) on preprocessed Yelp
samples to produce structured persona / rationale / ranking outputs. Each
Teacher output is a JSON object matching the schema in configs.teacher_prompt.

This script is resumable: if the output file already contains lines for a
given sample_id, those samples are skipped and new work is appended.

Example:
    $ python scripts/teacher/generate_teacher.py \\
        --input data/processed/philly_samples.jsonl \\
        --output data/teacher/philly_teacher.jsonl \\
        --num-samples 100 \\
        --model gemini-3-flash-preview \\
        --thinking-level minimal
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import time
from pathlib import Path
from typing import Any, Iterator

# Make configs importable when this script is run from the project root.
PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from dotenv import load_dotenv  # noqa: E402

from configs.teacher_prompt import (  # noqa: E402
    build_generation_config,
    build_user_prompt,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("generate_teacher")


def iter_samples(path: Path) -> Iterator[dict[str, Any]]:
    """Stream preprocessed samples from a JSONL file.

    Args:
        path (Path): samples.jsonl produced by preprocess_yelp.py.

    Yields:
        dict: one sample record.
    """
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            yield json.loads(line)


def read_declared_key_names(env_file: Path) -> list[str]:
    """Return GEMINI_API_KEY_* variable names explicitly declared in a .env file.

    We parse the .env text ourselves (instead of trusting os.environ) because
    the gemini_parallel package calls load_dotenv() at import time, and
    find_dotenv() can pick up an unrelated .env next to the editable install
    location, polluting os.environ with keys we do not intend to use. By
    returning only the names explicitly listed in our project's .env file we
    force AdvancedApiKeyManager(keylist_names=<list>) to ignore stray keys.

    Args:
        env_file (Path): path to our project .env file.

    Returns:
        list[str]: declared variable names in file order, deduplicated.
            Empty list if the file is missing.
    """
    if not env_file.exists():
        return []
    seen: set[str] = set()
    names: list[str] = []
    for raw in env_file.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith("export "):
            line = line[len("export ") :].lstrip()
        key, sep, _ = line.partition("=")
        if not sep:
            continue
        key = key.strip()
        if key.startswith("GEMINI_API_KEY_") and key not in seen:
            seen.add(key)
            names.append(key)
    return names


def load_done_sample_ids(output_path: Path) -> set[str]:
    """Collect sample_ids already present in an existing output file.

    Args:
        output_path (Path): teacher output JSONL (may not yet exist).

    Returns:
        set[str]: set of sample_ids that should be skipped (resume support).
            Only samples whose 'error' field is None count as done.
    """
    done: set[str] = set()
    if not output_path.exists():
        return done
    with output_path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue
            if rec.get("error") is None and rec.get("sample_id"):
                done.add(rec["sample_id"])
    return done


def parse_teacher_response(response_text: str) -> tuple[dict[str, Any] | None, str | None]:
    """Parse a Teacher response string into a JSON dict.

    Args:
        response_text (str): raw text returned by Gemini.

    Returns:
        tuple: (parsed_dict_or_None, error_message_or_None)
    """
    if not response_text:
        return None, "empty response"
    text = response_text.strip()
    # Gemini with responseMimeType=application/json should return pure JSON,
    # but occasionally wraps it in a ```json fence. Strip that defensively.
    if text.startswith("```"):
        text = text.strip("`")
        # Remove optional leading "json"
        if text.lower().startswith("json"):
            text = text[4:].lstrip()
    try:
        return json.loads(text), None
    except json.JSONDecodeError as e:
        return None, f"json decode failed: {e}"


def _coerce_digit_string(value: Any) -> Any:
    """Return int(value) when value is a digit-only string, else return value unchanged."""
    if isinstance(value, str) and value.isdigit():
        try:
            return int(value)
        except ValueError:
            return value
    return value


def coerce_indices_to_int(parsed: dict[str, Any] | None) -> dict[str, Any] | None:
    """Coerce digit-string candidate indices to int in a parsed Teacher payload.

    Gemini's response_schema requires string enums for token-level constraint
    (configs.teacher_prompt.build_gemini_response_schema_dict), so a fresh
    Gemini reply has ``ranking = ["3","1",...]`` and
    ``rationales[i].candidate_index = "3"``. The canonical Pydantic schema
    (TeacherResponse) plus all downstream readers (validate_teacher,
    train_student, eval_metrics) expect ``int``. This helper bridges the two
    on disk so on-disk records stay int regardless of teacher backend.

    Non-digit strings are left untouched (so legacy fixtures that use opaque
    strings like ``"b1"`` are unaffected) and ints are passed through, making
    the helper idempotent.

    Args:
        parsed (dict | None): JSON-decoded teacher payload, possibly None.

    Returns:
        dict | None: same object (mutated) with integer indices, or None
            unchanged if the input was None / not a dict.
    """
    if not isinstance(parsed, dict):
        return parsed
    rationales = parsed.get("rationales")
    if isinstance(rationales, list):
        for r in rationales:
            if isinstance(r, dict) and "candidate_index" in r:
                r["candidate_index"] = _coerce_digit_string(r["candidate_index"])
    ranking = parsed.get("ranking")
    if isinstance(ranking, list):
        parsed["ranking"] = [_coerce_digit_string(v) for v in ranking]
    return parsed


def build_prompts_for_batch(
    samples: list[dict[str, Any]],
    generation_config: dict[str, Any],
) -> list[dict[str, Any]]:
    """Convert samples into gemini_parallel prompt dicts.

    Args:
        samples (list[dict]): preprocessed samples.
        generation_config (dict): Gemini generation config shared across prompts.

    Returns:
        list[dict]: prompt dicts compatible with GeminiSequentialProcessor.
    """
    prompts: list[dict[str, Any]] = []
    for sample in samples:
        prompts.append(
            {
                "prompt": build_user_prompt(sample),
                "generation_config": generation_config,
                "metadata": {
                    "task_id": sample["sample_id"],
                    "user_id": sample["user_id"],
                    "positive_business_id": sample["positive_business_id"],
                },
            }
        )
    return prompts


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--input", type=Path, required=True, help="samples.jsonl")
    p.add_argument("--output", type=Path, required=True, help="teacher output JSONL")
    p.add_argument(
        "--num-samples",
        type=int,
        default=None,
        help="process at most this many NEW samples (after resume skip)",
    )
    p.add_argument(
        "--model",
        type=str,
        default="gemini-3-flash-preview",
        help="Gemini model name",
    )
    p.add_argument(
        "--thinking-level",
        type=str,
        default="minimal",
        choices=["minimal", "low", "medium", "high"],
    )
    p.add_argument(
        "--api-interval",
        type=float,
        default=5.0,
        help=(
            "minimum seconds between API calls (IP-ban safety). Raise if the "
            "model returns 503 UNAVAILABLE; lower carefully if you own many paid keys."
        ),
    )
    p.add_argument(
        "--env-file",
        type=Path,
        default=PROJECT_ROOT / ".env",
        help="path to .env file containing GEMINI_API_KEY_* variables",
    )
    p.add_argument(
        "--keylist",
        type=str,
        default="all",
        help="'all' or comma-separated key names",
    )
    p.add_argument(
        "--paid-keys",
        type=str,
        default="",
        help="'all' or comma-separated key names marked as paid (no cooldown)",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()
    args.output.parent.mkdir(parents=True, exist_ok=True)

    # Load env BEFORE importing gemini_parallel so AdvancedApiKeyManager sees keys.
    if args.env_file.exists():
        load_dotenv(args.env_file)
        log.info("loaded env from %s", args.env_file)
    else:
        log.warning("env file not found at %s (expecting keys already in env)", args.env_file)

    # Imported here so .env is loaded first.
    from gemini_parallel import (  # noqa: E402
        AdvancedApiKeyManager,
        GeminiSequentialProcessor,
    )

    # Build key manager. "all" is interpreted as "every GEMINI_API_KEY_* that is
    # explicitly declared in our project .env file", NOT every env var of that
    # shape currently visible in os.environ (see read_declared_key_names docstring).
    keylist_arg: Any
    if args.keylist.strip().lower() == "all":
        declared = read_declared_key_names(args.env_file)
        if not declared:
            log.error(
                "no GEMINI_API_KEY_* variables declared in %s; pass --keylist "
                "explicitly or populate the .env file",
                args.env_file,
            )
            sys.exit(1)
        keylist_arg = declared
        log.info("using %d keys declared in %s: %s", len(declared), args.env_file, declared)
    else:
        keylist_arg = [k.strip() for k in args.keylist.split(",") if k.strip()]

    paid_arg: Any = None
    if args.paid_keys.strip():
        if args.paid_keys.strip().lower() == "all":
            paid_arg = "all"
        else:
            paid_arg = [k.strip() for k in args.paid_keys.split(",") if k.strip()]

    key_manager_kwargs: dict[str, Any] = {"keylist_names": keylist_arg}
    if paid_arg is not None:
        key_manager_kwargs["paid_keys"] = paid_arg
    key_manager = AdvancedApiKeyManager(**key_manager_kwargs)

    status = key_manager.get_keys_status_summary()
    log.info("key manager loaded: %s", status)

    processor = GeminiSequentialProcessor(
        key_manager=key_manager,
        model_name=args.model,
        api_call_interval=args.api_interval,
    )

    generation_config = build_generation_config(thinking_level=args.thinking_level)

    # Resume support: skip samples already in output file.
    done_ids = load_done_sample_ids(args.output)
    if done_ids:
        log.info("resume: found %d samples already done in %s", len(done_ids), args.output)

    # Collect pending samples (up to --num-samples)
    pending: list[dict[str, Any]] = []
    for sample in iter_samples(args.input):
        if sample["sample_id"] in done_ids:
            continue
        pending.append(sample)
        if args.num_samples is not None and len(pending) >= args.num_samples:
            break
    log.info("%d pending samples to process", len(pending))

    if not pending:
        log.info("nothing to do; exiting")
        return

    prompts = build_prompts_for_batch(pending, generation_config)

    # Process sequentially (gemini_parallel internally rotates keys, enforces interval).
    t_start = time.time()
    ok = 0
    fail = 0
    with args.output.open("a", encoding="utf-8") as out:
        for i, (prompt_data, sample) in enumerate(zip(prompts, pending)):
            metadata, response, error = processor.process_single(prompt_data)
            parsed, parse_err = (None, None)
            if error is None and isinstance(response, str):
                parsed, parse_err = parse_teacher_response(response)
            elif error is None and response is not None:
                # Some code paths return a response object with .text
                text = getattr(response, "text", None)
                if text:
                    parsed, parse_err = parse_teacher_response(text)
                else:
                    parse_err = "response object has no text"

            # Gemini response_schema returns digit-string indices; canonical
            # disk format is int (matches TeacherResponse Literal[1..10] and
            # the Qwen teacher path). Coerce here, before fsync.
            parsed = coerce_indices_to_int(parsed)

            final_error = error or parse_err

            record = {
                "sample_id": sample["sample_id"],
                "user_id": sample["user_id"],
                "positive_business_id": sample["positive_business_id"],
                "metadata": metadata,
                "raw_response": response if isinstance(response, str) else None,
                "teacher_output": parsed,
                "error": final_error,
                "model": args.model,
                "thinking_level": args.thinking_level,
            }
            out.write(json.dumps(record, ensure_ascii=False) + "\n")
            out.flush()
            os.fsync(out.fileno())

            if final_error is None:
                ok += 1
            else:
                fail += 1

            if (i + 1) % 10 == 0 or (i + 1) == len(prompts):
                elapsed = time.time() - t_start
                rate = (i + 1) / elapsed if elapsed > 0 else 0.0
                log.info(
                    "progress %d/%d | ok=%d fail=%d | %.1f s elapsed | %.2f req/s",
                    i + 1,
                    len(prompts),
                    ok,
                    fail,
                    elapsed,
                    rate,
                )

    log.info("done: %d ok, %d failed, out=%s", ok, fail, args.output)


if __name__ == "__main__":
    main()

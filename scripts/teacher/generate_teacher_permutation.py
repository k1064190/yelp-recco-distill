#!/usr/bin/env python
# ABOUTME: Generate a second teacher pass over a user-specified candidate permutation
# ABOUTME: via the running vLLM HTTP serve, for Option-1 (PRP) position-bias debiasing.
"""Generate a teacher pass over a permuted candidate list (HTTP serve).

The existing ``philly_teacher_qwen35.jsonl`` is pass #1: the teacher reads
``sample.candidates`` in their stored order. This script produces pass #2 where
``candidates`` are reordered by a fixed permutation (default: reverse) before
the prompt is built, so that merging pass #1 + pass #2 via Borda count can
cancel the Qwen3.5 primacy bias that the position-bias analysis surfaced
(slot 1 top-1 rate 20.25% vs uniform 10%, χ² p=3.5e-74).

Key output detail
-----------------
The teacher output in pass #2 uses ``candidate_index`` *in prompt order*
(i.e. 1..10 over the permuted list). To recover the original-ordering
ranking we save the permutation as a list alongside the teacher output.
Specifically, ``permutation[i - 1]`` is the *original* slot id (1-based) of
the candidate shown to the teacher at prompt position ``i`` (1-based).

Example: ``permutation = [10, 9, ..., 1]``. If the teacher returns
``ranking = [1, 2, ...]``, the top-1 *original* slot is
``permutation[0] = 10``.

Transport
---------
Uses the vLLM OpenAI-compatible ``/v1/chat/completions`` endpoint with
``response_format={"type": "json_schema", ...}`` for guaranteed-parseable
JSON, matching what ``generate_teacher_qwen.py`` does in offline mode. The
teacher is assumed to already be serving — this script does not spawn it.

Usage
-----
    python \\
        scripts/teacher/generate_teacher_permutation.py \\
        --input  data/processed/philly_samples.jsonl \\
        --output data/teacher/philly_teacher_qwen35_perm_reverse.jsonl \\
        --base-url http://10.1.1.71:8100/v1 \\
        --concurrency 8 \\
        --permutation reverse

Resume-safe: rerun the same command to pick up where a crash left off.
"""

from __future__ import annotations

import argparse
import concurrent.futures as cf
import json
import logging
import sys
import threading
import time
from copy import deepcopy
from pathlib import Path
from typing import Any, Iterator

import requests

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from configs.teacher_prompt import (  # noqa: E402
    N_CANDIDATES,
    SYSTEM_INSTRUCTION as _DEFAULT_SYSTEM_INSTRUCTION,
    TeacherResponse,
    build_user_prompt,
)

# Overridable at CLI via --prompt-module. Keep the canonical variant as
# default so existing launch scripts keep working unchanged.
SYSTEM_INSTRUCTION = _DEFAULT_SYSTEM_INSTRUCTION

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("gen_teacher_perm")

_WRITE_LOCK = threading.Lock()


def iter_samples(path: Path) -> Iterator[dict[str, Any]]:
    """Stream preprocessed samples from a JSONL file.

    Args:
        path (Path): samples.jsonl produced by preprocess_yelp.py.

    Yields:
        dict: one sample record.
    """
    with path.open("r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                yield json.loads(line)


def load_done_sample_ids(output_path: Path) -> set[str]:
    """Collect sample_ids already present in an existing output file.

    Args:
        output_path (Path): teacher output JSONL; may not exist yet.

    Returns:
        set[str]: sample_ids whose record has ``error is None`` — those are
            considered successfully complete and will be skipped on resume.
    """
    done: set[str] = set()
    if not output_path.exists():
        return done
    with output_path.open("r", encoding="utf-8") as fh:
        for line in fh:
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


def make_permutation(kind: str, k: int = N_CANDIDATES) -> list[int]:
    """Build a 1-based permutation of the candidate slots.

    Args:
        kind (str): 'reverse' (prompt pos i = original slot k+1-i),
            or 'identity' (no reordering — pass #1 equivalent, for sanity).
        k (int): candidate count (default :const:`N_CANDIDATES`).

    Returns:
        list[int]: ``permutation`` such that ``permutation[i-1]`` is the
            1-based *original* slot of the candidate shown at prompt position
            ``i``.
    """
    if kind == "reverse":
        return list(range(k, 0, -1))
    if kind == "identity":
        return list(range(1, k + 1))
    raise ValueError(f"unknown permutation: {kind}")


def apply_permutation(sample: dict[str, Any], perm: list[int]) -> dict[str, Any]:
    """Return a shallow copy of ``sample`` with ``candidates`` reordered.

    Args:
        sample (dict): record with a ``candidates`` list of length
            ``len(perm)``.
        perm (list[int]): 1-based permutation from :func:`make_permutation`.

    Returns:
        dict: new dict where ``candidates[i]`` is the candidate that
            originally sat at slot ``perm[i]`` (1-based). Other keys shared.
    """
    cands = sample["candidates"]
    if len(cands) != len(perm):
        raise ValueError(
            f"sample {sample.get('sample_id')} has {len(cands)} candidates, "
            f"permutation has {len(perm)}"
        )
    out = dict(sample)  # shallow
    out["candidates"] = [cands[p - 1] for p in perm]
    return out


def parse_teacher_response(
    text: str,
) -> tuple[dict[str, Any] | None, str | None]:
    """Parse the raw response string into a JSON dict.

    Args:
        text (str): raw assistant content.

    Returns:
        tuple: ``(parsed_dict, error_message)``. Exactly one is ``None``.
    """
    if not text:
        return None, "empty response"
    text = text.strip()
    if text.startswith("```"):
        text = text.strip("`")
        if text.lower().startswith("json"):
            text = text[4:].lstrip()
    try:
        return json.loads(text), None
    except json.JSONDecodeError as e:
        return None, f"json decode failed: {e}"


def call_teacher(
    session: requests.Session,
    base_url: str,
    model: str,
    messages: list[dict[str, str]],
    schema: dict[str, Any],
    temperature: float,
    max_tokens: int,
    timeout: float,
    max_retries: int,
    enable_thinking: bool | None = None,
) -> tuple[dict[str, Any] | None, str | None]:
    """POST one chat completion with a JSON schema and return the JSON body.

    Args:
        session (requests.Session): reused per-thread for connection pooling.
        base_url (str): e.g. ``http://host:port/v1`` (no trailing slash).
        model (str): ``served-model-name`` of the vLLM serve.
        messages (list[dict]): OpenAI chat messages [system, user].
        schema (dict): JSON Schema for guided decoding.
        temperature (float): sampling temperature.
        max_tokens (int): completion budget.
        timeout (float): per-request seconds.
        max_retries (int): retry count on 5xx / network errors. Backoff is
            exponential: ``2 ** attempt`` seconds.
        enable_thinking: if not ``None``, pass ``chat_template_kwargs`` so
            Qwen3.5-style thinking-mode can be toggled. ``False`` disables CoT
            (recommended for structured output teachers).

    Returns:
        tuple: ``(response_json, error_message)``. Exactly one is ``None``.
    """
    url = base_url.rstrip("/") + "/chat/completions"
    payload = {
        "model": model,
        "messages": messages,
        "temperature": temperature,
        "max_tokens": max_tokens,
        "response_format": {
            "type": "json_schema",
            "json_schema": {"name": "TeacherResponse", "schema": schema},
        },
    }
    if enable_thinking is not None:
        payload["chat_template_kwargs"] = {"enable_thinking": enable_thinking}
    last_err = None
    for attempt in range(max_retries + 1):
        try:
            r = session.post(url, json=payload, timeout=timeout)
            if r.status_code == 200:
                return r.json(), None
            last_err = f"http {r.status_code}: {r.text[:200]}"
            if 500 <= r.status_code < 600:
                time.sleep(min(2 ** attempt, 10))
                continue
            return None, last_err  # 4xx: don't retry
        except requests.exceptions.RequestException as e:
            last_err = f"request exception: {e}"
            time.sleep(min(2 ** attempt, 10))
    return None, f"exhausted retries; last error: {last_err}"


def process_one(
    sample: dict[str, Any],
    perm: list[int],
    session: requests.Session,
    base_url: str,
    model: str,
    schema: dict[str, Any],
    temperature: float,
    max_tokens: int,
    timeout: float,
    max_retries: int,
    enable_thinking: bool | None = None,
) -> dict[str, Any]:
    """Run one teacher call for a permuted sample and assemble the record.

    Args:
        sample (dict): original sample (candidates in original order).
        perm (list[int]): permutation applied before prompting.
        session, base_url, model, schema, temperature, max_tokens, timeout,
            max_retries: see :func:`call_teacher`.

    Returns:
        dict: output record with keys
            ``sample_id``, ``user_id``, ``positive_business_id``, ``model``,
            ``permutation``, ``raw_response``, ``teacher_output``,
            ``metadata``, ``error``.
    """
    permuted = apply_permutation(sample, perm)
    messages = [
        {"role": "system", "content": SYSTEM_INSTRUCTION},
        {"role": "user", "content": build_user_prompt(permuted)},
    ]
    body, http_err = call_teacher(
        session=session,
        base_url=base_url,
        model=model,
        messages=messages,
        schema=schema,
        temperature=temperature,
        max_tokens=max_tokens,
        timeout=timeout,
        max_retries=max_retries,
        enable_thinking=enable_thinking,
    )
    if http_err is not None:
        return {
            "sample_id": sample["sample_id"],
            "user_id": sample["user_id"],
            "positive_business_id": sample["positive_business_id"],
            "model": model,
            "permutation": perm,
            "raw_response": None,
            "teacher_output": None,
            "metadata": None,
            "error": http_err,
        }

    choice = body["choices"][0]
    text = choice["message"]["content"]
    parsed, parse_err = parse_teacher_response(text)
    usage = body.get("usage") or {}
    metadata = {
        "prompt_tokens": usage.get("prompt_tokens"),
        "completion_tokens": usage.get("completion_tokens"),
        "finish_reason": choice.get("finish_reason"),
    }
    return {
        "sample_id": sample["sample_id"],
        "user_id": sample["user_id"],
        "positive_business_id": sample["positive_business_id"],
        "model": model,
        "permutation": perm,
        "raw_response": text,
        "teacher_output": parsed,
        "metadata": metadata,
        "error": parse_err,
    }


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments.

    Returns:
        argparse.Namespace: parsed arguments.
    """
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--input", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--base-url", type=str, default="http://10.1.1.71:8100/v1")
    p.add_argument("--model", type=str, default="qwen35-teacher")
    p.add_argument(
        "--permutation", type=str, default="reverse",
        choices=["reverse", "identity"],
        help="how to reorder candidates before prompting",
    )
    p.add_argument("--temperature", type=float, default=0.7)
    p.add_argument("--max-tokens", type=int, default=2048)
    p.add_argument("--timeout", type=float, default=180.0)
    p.add_argument("--max-retries", type=int, default=2)
    p.add_argument(
        "--concurrency", type=int, default=8,
        help="in-flight HTTP requests; serve stress-tested to 8",
    )
    p.add_argument("--max-samples", type=int, default=None,
                   help="process at most this many NEW samples")
    p.add_argument("--log-every", type=int, default=50)
    p.add_argument(
        "--prompt-module", type=str, default=None,
        help=(
            "Override SYSTEM_INSTRUCTION by importing it from a different "
            "module (e.g. 'configs.teacher_prompt_example'). build_user_prompt "
            "and TeacherResponse schema still come from configs.teacher_prompt."
        ),
    )
    p.add_argument(
        "--enable-thinking", action="store_true",
        help=(
            "Pass chat_template_kwargs={'enable_thinking': True}. Default is "
            "OFF so Qwen3.5 emits JSON directly without CoT preamble."
        ),
    )
    p.add_argument(
        "--no-enable-thinking", action="store_true",
        help=(
            "Explicitly disable thinking via chat_template_kwargs. Default "
            "behaviour (neither flag) is to omit the kwarg entirely so the "
            "server-side template default applies."
        ),
    )
    p.add_argument(
        "--shard-index", type=int, default=0,
        help="process only samples where idx %% shard_total == shard_index",
    )
    p.add_argument(
        "--shard-total", type=int, default=1,
        help="number of shards to split samples across (1 = no sharding)",
    )
    return p.parse_args()


def main() -> None:
    """Entry point: stream samples, concurrent HTTP calls, append to output."""
    global SYSTEM_INSTRUCTION
    args = parse_args()
    args.output.parent.mkdir(parents=True, exist_ok=True)

    if args.prompt_module:
        import importlib
        mod = importlib.import_module(args.prompt_module)
        SYSTEM_INSTRUCTION = mod.SYSTEM_INSTRUCTION
        log.info("loaded SYSTEM_INSTRUCTION from %s (%d chars)",
                 args.prompt_module, len(SYSTEM_INSTRUCTION))

    if args.enable_thinking and args.no_enable_thinking:
        raise SystemExit("--enable-thinking and --no-enable-thinking are mutually exclusive")
    if args.enable_thinking:
        enable_thinking: bool | None = True
    elif args.no_enable_thinking:
        enable_thinking = False
    else:
        enable_thinking = None

    perm = make_permutation(args.permutation)
    schema = TeacherResponse.model_json_schema()
    log.info("permutation kind=%s → %s", args.permutation, perm)
    log.info("teacher schema keys: %s",
             list(schema.get("properties", {}).keys()))
    log.info("enable_thinking override: %s (None=server default)", enable_thinking)

    done = load_done_sample_ids(args.output)
    log.info("resume: %d samples already done in %s", len(done), args.output)

    if args.shard_total < 1 or not (0 <= args.shard_index < args.shard_total):
        raise SystemExit(
            f"invalid shard: index={args.shard_index} total={args.shard_total}"
        )

    pending: list[dict[str, Any]] = []
    for idx, s in enumerate(iter_samples(args.input)):
        if args.shard_total > 1 and idx % args.shard_total != args.shard_index:
            continue
        if s["sample_id"] in done:
            continue
        pending.append(s)
        if args.max_samples is not None and len(pending) >= args.max_samples:
            break
    log.info(
        "%d pending samples (shard %d/%d)",
        len(pending), args.shard_index, args.shard_total,
    )
    if not pending:
        return

    session_local = threading.local()

    def get_session() -> requests.Session:
        """Return a per-thread requests session (connection pooling)."""
        sess = getattr(session_local, "s", None)
        if sess is None:
            sess = requests.Session()
            adapter = requests.adapters.HTTPAdapter(
                pool_connections=args.concurrency,
                pool_maxsize=args.concurrency,
                max_retries=0,  # we do our own retry
            )
            sess.mount("http://", adapter)
            sess.mount("https://", adapter)
            session_local.s = sess
        return sess

    t_start = time.time()
    n_ok = 0
    n_fail = 0

    def worker(sample: dict[str, Any]) -> dict[str, Any]:
        """Thread worker: one teacher call for one sample."""
        return process_one(
            sample=sample,
            perm=perm,
            session=get_session(),
            base_url=args.base_url,
            model=args.model,
            schema=schema,
            temperature=args.temperature,
            max_tokens=args.max_tokens,
            timeout=args.timeout,
            max_retries=args.max_retries,
            enable_thinking=enable_thinking,
        )

    with cf.ThreadPoolExecutor(max_workers=args.concurrency) as pool:
        futures = [pool.submit(worker, s) for s in pending]
        for i, fut in enumerate(cf.as_completed(futures), start=1):
            rec = fut.result()
            with _WRITE_LOCK, args.output.open("a", encoding="utf-8") as fh:
                fh.write(json.dumps(rec, ensure_ascii=False) + "\n")
                fh.flush()
            if rec["error"] is None:
                n_ok += 1
            else:
                n_fail += 1
            if i % args.log_every == 0 or i == len(pending):
                elapsed = time.time() - t_start
                rate = i / elapsed if elapsed > 0 else 0
                eta = (len(pending) - i) / rate if rate > 0 else float("inf")
                log.info(
                    "progress %d/%d  ok=%d fail=%d  %.2f req/s  eta %.0fs",
                    i, len(pending), n_ok, n_fail, rate, eta,
                )

    log.info("DONE: ok=%d fail=%d wall=%.1fs output=%s",
             n_ok, n_fail, time.time() - t_start, args.output)


if __name__ == "__main__":
    main()

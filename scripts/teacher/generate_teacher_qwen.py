#!/usr/bin/env python
# ABOUTME: Generate Teacher outputs (persona + rationales + ranking) from a
# ABOUTME: preprocessed Yelp sample JSONL using a local Qwen3.5 model via vLLM offline batch inference.

"""
Run a local Teacher LLM (default: Qwen/Qwen3.5-35B-A3B) on preprocessed Yelp
samples to produce structured persona / rationale / ranking outputs. Each
Teacher output is a JSON object matching the schema in configs.teacher_prompt.

Uses vLLM offline LLM class for batch inference with guided JSON decoding to
guarantee schema-conformant outputs. Processes samples in configurable chunks
with incremental disk writes for crash-safe resume.

This script is resumable: if the output file already contains lines for a
given sample_id with error=None, those samples are skipped on restart.

Example:
    $ python scripts/teacher/generate_teacher_qwen.py \\
        --input data/processed/philly_samples.jsonl \\
        --output data/teacher/philly_teacher_qwen35.jsonl \\
        --model Qwen/Qwen3.5-35B-A3B \\
        --tp 4 --batch-size 64

Env:
    Requires the matching environment (vLLM nightly + transformers
    main branch) for Qwen3.5 Gated DeltaNet architecture support.
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

from configs.teacher_prompt import (  # noqa: E402
    SYSTEM_INSTRUCTION,
    TeacherResponse,
    build_user_prompt,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("generate_teacher_qwen")


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
        response_text (str): raw text returned by the model.

    Returns:
        tuple: (parsed_dict_or_None, error_message_or_None)
    """
    if not response_text:
        return None, "empty response"
    text = response_text.strip()
    # vLLM guided_json should return pure JSON, but strip fences defensively.
    if text.startswith("```"):
        text = text.strip("`")
        if text.lower().startswith("json"):
            text = text[4:].lstrip()
    try:
        return json.loads(text), None
    except json.JSONDecodeError as e:
        return None, f"json decode failed: {e}"


def build_chat_messages(sample: dict[str, Any]) -> list[dict[str, str]]:
    """Build a chat message list from a preprocessed sample.

    Args:
        sample (dict): one record from samples.jsonl with keys
            history (list[dict]), candidates (list[dict]).

    Returns:
        list[dict]: OpenAI-style message list [system, user].
    """
    return [
        {"role": "system", "content": SYSTEM_INSTRUCTION},
        {"role": "user", "content": build_user_prompt(sample)},
    ]


def chunks(lst: list, n: int) -> Iterator[list]:
    """Split list into chunks of size n.

    Args:
        lst (list): input list.
        n (int): chunk size.

    Yields:
        list: sub-lists of up to n elements.
    """
    for i in range(0, len(lst), n):
        yield lst[i : i + n]


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments.

    Returns:
        argparse.Namespace: parsed arguments.
    """
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--input", type=Path, required=True, help="samples.jsonl")
    p.add_argument(
        "--output",
        type=Path,
        required=True,
        help="teacher output JSONL (append mode for resume)",
    )
    p.add_argument(
        "--num-samples",
        type=int,
        default=None,
        help="process at most this many NEW samples (after resume skip)",
    )
    p.add_argument(
        "--model",
        type=str,
        default="Qwen/Qwen3.5-35B-A3B",
        help="HuggingFace model name or local path (default: Qwen/Qwen3.5-35B-A3B)",
    )
    p.add_argument(
        "--tp",
        type=int,
        default=4,
        help="tensor parallel size (default: 4 for 4x GPU)",
    )
    p.add_argument(
        "--dtype",
        type=str,
        default="bfloat16",
        choices=["bfloat16", "float16", "auto"],
        help="model dtype (default: bfloat16)",
    )
    p.add_argument(
        "--max-model-len",
        type=int,
        default=8192,
        help="max context length for vLLM engine (default: 8192)",
    )
    p.add_argument(
        "--max-tokens",
        type=int,
        default=2048,
        help="max new tokens per generation (default: 2048)",
    )
    p.add_argument(
        "--temperature",
        type=float,
        default=0.7,
        help="sampling temperature (default: 0.7)",
    )
    p.add_argument(
        "--batch-size",
        type=int,
        default=64,
        help="number of samples per vLLM batch (default: 64). Larger = more "
        "efficient GPU utilization but longer between disk writes.",
    )
    p.add_argument(
        "--enable-thinking",
        action="store_true",
        default=False,
        help="enable model thinking mode (default: off for structured output)",
    )
    p.add_argument(
        "--gpu-memory-utilization",
        type=float,
        default=0.90,
        help="fraction of GPU memory for vLLM (default: 0.90)",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()
    args.output.parent.mkdir(parents=True, exist_ok=True)

    # Resume support: skip samples already in output file.
    done_ids = load_done_sample_ids(args.output)
    if done_ids:
        log.info("resume: found %d samples already done in %s", len(done_ids), args.output)

    # Collect pending samples (up to --num-samples).
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

    # Import vLLM here (heavy import, only needed if there's work to do).
    from vllm import LLM, SamplingParams  # noqa: E402

    # Detect which structured output API is available.
    # vLLM 0.19+ nightly uses StructuredOutputsParams(json=schema).
    # Older vLLM uses GuidedDecodingParams or guided_json kwarg.
    structured_output_mode = "unknown"
    StructuredOutputsParams = None
    GuidedDecodingParams = None

    try:
        from vllm.sampling_params import StructuredOutputsParams as _SOP
        StructuredOutputsParams = _SOP
        structured_output_mode = "structured_outputs_params"
    except ImportError:
        pass

    if structured_output_mode == "unknown":
        try:
            from vllm.sampling_params import GuidedDecodingParams as _GDP
            GuidedDecodingParams = _GDP
            structured_output_mode = "guided_cls"
        except ImportError:
            structured_output_mode = "guided_json_kwarg"

    log.info("structured output mode: %s", structured_output_mode)

    # Build JSON schema for guided decoding.
    teacher_schema = TeacherResponse.model_json_schema()
    log.info(
        "teacher JSON schema keys: %s",
        list(teacher_schema.get("properties", {}).keys()),
    )

    # Initialize vLLM engine.
    log.info(
        "loading model %s (tp=%d, dtype=%s, max_model_len=%d, gpu_mem=%.2f)",
        args.model,
        args.tp,
        args.dtype,
        args.max_model_len,
        args.gpu_memory_utilization,
    )
    t_load = time.time()

    # Build chat template kwargs to disable thinking mode for structured output.
    chat_template_kwargs = {}
    if not args.enable_thinking:
        chat_template_kwargs["enable_thinking"] = False

    llm = LLM(
        model=args.model,
        tensor_parallel_size=args.tp,
        dtype=args.dtype,
        max_model_len=args.max_model_len,
        gpu_memory_utilization=args.gpu_memory_utilization,
        trust_remote_code=True,
        enforce_eager=True,
    )
    log.info("model loaded in %.1f s", time.time() - t_load)

    # Build sampling params with structured JSON output.
    base_kwargs = {
        "temperature": args.temperature,
        "max_tokens": args.max_tokens,
    }
    if structured_output_mode == "structured_outputs_params":
        base_kwargs["structured_outputs"] = StructuredOutputsParams(
            json=teacher_schema
        )
    elif structured_output_mode == "guided_cls":
        base_kwargs["guided_decoding"] = GuidedDecodingParams(json=teacher_schema)
    else:
        base_kwargs["guided_json"] = teacher_schema

    sampling_params = SamplingParams(**base_kwargs)

    # Process in batches with incremental disk writes.
    t_start = time.time()
    total_ok = 0
    total_fail = 0
    batch_num = 0

    for batch in chunks(pending, args.batch_size):
        batch_num += 1
        batch_t = time.time()

        # Build conversations for this batch.
        conversations = [build_chat_messages(s) for s in batch]

        # Run vLLM batch inference.
        try:
            outputs = llm.chat(
                conversations,
                sampling_params,
                chat_template_kwargs=chat_template_kwargs,
            )
        except Exception as e:
            log.error("vLLM batch %d failed: %s", batch_num, e)
            # Write error records for the whole batch so resume skips nothing.
            with args.output.open("a", encoding="utf-8") as out:
                for sample in batch:
                    record = {
                        "sample_id": sample["sample_id"],
                        "user_id": sample["user_id"],
                        "positive_business_id": sample["positive_business_id"],
                        "metadata": None,
                        "raw_response": None,
                        "teacher_output": None,
                        "error": f"vllm batch error: {e}",
                        "model": args.model,
                    }
                    out.write(json.dumps(record, ensure_ascii=False) + "\n")
                total_fail += len(batch)
            continue

        # Write results to disk.
        ok_batch = 0
        fail_batch = 0
        with args.output.open("a", encoding="utf-8") as out:
            for sample, output in zip(batch, outputs):
                response_text = output.outputs[0].text if output.outputs else ""
                parsed, parse_err = parse_teacher_response(response_text)

                record = {
                    "sample_id": sample["sample_id"],
                    "user_id": sample["user_id"],
                    "positive_business_id": sample["positive_business_id"],
                    "metadata": {
                        "prompt_tokens": len(output.prompt_token_ids)
                        if output.prompt_token_ids
                        else None,
                        "completion_tokens": sum(
                            len(o.token_ids) for o in output.outputs
                        )
                        if output.outputs
                        else None,
                        "finish_reason": output.outputs[0].finish_reason
                        if output.outputs
                        else None,
                    },
                    "raw_response": response_text,
                    "teacher_output": parsed,
                    "error": parse_err,
                    "model": args.model,
                }
                out.write(json.dumps(record, ensure_ascii=False) + "\n")

                if parse_err is None:
                    ok_batch += 1
                else:
                    fail_batch += 1

            out.flush()
            os.fsync(out.fileno())

        total_ok += ok_batch
        total_fail += fail_batch
        elapsed = time.time() - t_start
        batch_elapsed = time.time() - batch_t
        processed = total_ok + total_fail
        rate = processed / elapsed if elapsed > 0 else 0.0

        log.info(
            "batch %d done (%d samples, %.1f s) | "
            "progress %d/%d | ok=%d fail=%d | "
            "%.1f s elapsed | %.2f samples/s",
            batch_num,
            len(batch),
            batch_elapsed,
            processed,
            len(pending),
            total_ok,
            total_fail,
            elapsed,
            rate,
        )

    log.info(
        "DONE: %d ok, %d failed out of %d total, output=%s, wall=%.1f s",
        total_ok,
        total_fail,
        len(pending),
        args.output,
        time.time() - t_start,
    )


if __name__ == "__main__":
    main()

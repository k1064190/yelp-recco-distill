#!/usr/bin/env python
# ABOUTME: W4A16 GPTQ post-training quantization of the merged Qwen3-4B
# ABOUTME: student with in-domain calibration data from the SFT train split.

"""
Quantize the merged Qwen3-4B student (``ckpt/student-merged``) to W4A16 using
``llm-compressor``'s GPTQ recipe, then save a compressed-tensors checkpoint
that ``vllm serve --quantization compressed-tensors`` can load directly.

Calibration policy (see ../../.../plans/drifting-questing-meadow.md §2):

* The calibration samples are drawn from the **same deterministic train
  split** that ``train_student.py`` fits on, not from the eval split. This
  matches two constraints at once:
    1. The activation distributions at quantization time come from the
       actual domain (Yelp restaurant recommendation JSON output), which
       gives better accuracy recovery than generic text like C4 or
       UltraChat.
    2. The eval split remains untouched, so downstream latency benchmarks
       and the LLM-as-a-Judge evaluation on the eval split do not leak
       calibration data.
* Calibration samples are rendered as full conversations (system + user +
  assistant) via the model's chat template, then tokenized with the same
  tokenizer that serves the model at inference time. This is the format
  the llm-compressor docs recommend for SFT-style models.

Example:
    $ python scripts/quantize/quantize_w4a16.py \\
        --model ckpt/student-merged \\
        --samples data/processed/philly_samples.jsonl \\
        --teacher data/teacher/philly_teacher.jsonl \\
        --output ckpt/student-w4a16 \\
        --num-calib 128
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.train.train_student import (  # noqa: E402
    build_training_example,
    load_and_filter,
    split_examples,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("quantize_w4a16")


def build_calibration_dataset(
    samples_path: Path,
    teacher_path: Path,
    tokenizer,
    num_calib: int,
    max_seq_length: int,
    eval_ratio: float = 0.9,
):
    """Produce a tokenized HF Dataset for GPTQ calibration.

    Joins teacher outputs with source samples, applies the same deterministic
    90/10 split that ``train_student.py`` uses, keeps the train half, then
    turns each kept record into a full ChatML conversation text, truncates
    to ``max_seq_length`` tokens, and returns a ``datasets.Dataset`` with
    ``input_ids`` and ``attention_mask`` columns — the exact shape
    ``llmcompressor.oneshot`` expects.

    Args:
        samples_path (Path): processed samples JSONL.
        teacher_path (Path): teacher output JSONL.
        tokenizer: a Hugging Face tokenizer (must have ``apply_chat_template``).
        num_calib (int): maximum calibration sample count (truncated from
            the train split if there are more available).
        max_seq_length (int): per-sample token truncation limit.
        eval_ratio (float): train fraction used in the deterministic split.
            Must match train_student.py's default to avoid eval leakage.

    Returns:
        datasets.Dataset: tokenized calibration dataset with columns
            ``input_ids`` (list[int]) and ``attention_mask`` (list[int]).
    """
    from datasets import Dataset

    examples, _stats = load_and_filter(samples_path, teacher_path)
    train_exs, _eval_exs = split_examples(examples, ratio=eval_ratio)
    log.info(
        "calibration pool: %d train-split examples available, will use up to %d",
        len(train_exs),
        num_calib,
    )
    selected = train_exs[:num_calib]

    def _to_tokenized(joined: dict) -> dict:
        """Render one joined record as tokenized ChatML text.

        Args:
            joined (dict): one element from ``load_and_filter`` (sample + teacher).

        Returns:
            dict: tokenized fields — input_ids (list[int]),
                attention_mask (list[int]).
        """
        ex = build_training_example(joined["sample"], joined["teacher"])
        messages = ex["prompt"] + ex["completion"]
        text = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=False
        )
        # add_special_tokens=False because apply_chat_template already
        # embeds any BOS/EOS the model needs.
        return tokenizer(
            text,
            padding=False,
            truncation=True,
            max_length=max_seq_length,
            add_special_tokens=False,
        )

    rows = [_to_tokenized(ex) for ex in selected]
    return Dataset.from_list(rows)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--model",
        type=Path,
        default=PROJECT_ROOT / "ckpt/student-merged",
        help="LoRA-merged Qwen3-4B checkpoint to quantize",
    )
    p.add_argument(
        "--samples",
        type=Path,
        default=PROJECT_ROOT / "data/processed/philly_samples.jsonl",
    )
    p.add_argument(
        "--teacher",
        type=Path,
        default=PROJECT_ROOT / "data/teacher/philly_teacher.jsonl",
    )
    p.add_argument(
        "--output",
        type=Path,
        default=PROJECT_ROOT / "ckpt/student-w4a16",
    )
    p.add_argument("--num-calib", type=int, default=128)
    p.add_argument("--max-seq-length", type=int, default=2048)
    p.add_argument("--eval-ratio", type=float, default=0.9)
    p.add_argument(
        "--dampening-frac",
        type=float,
        default=0.01,
        help="GPTQ Hessian dampening; default 0.01 is fine for 4B models",
    )
    p.add_argument(
        "--skip-sanity",
        action="store_true",
        help="skip the post-quant vLLM smoke-test generation",
    )
    return p.parse_args()


def main() -> int:
    args = parse_args()

    if not args.model.exists():
        log.error("merged model path not found: %s", args.model)
        return 2

    # Heavy imports deferred so --help is cheap.
    import torch
    from llmcompressor import oneshot
    from llmcompressor.modifiers.quantization import GPTQModifier
    from transformers import AutoModelForCausalLM, AutoTokenizer

    log.info("loading merged model from %s", args.model)
    tokenizer = AutoTokenizer.from_pretrained(str(args.model), use_fast=True)
    model = AutoModelForCausalLM.from_pretrained(
        str(args.model),
        dtype=torch.bfloat16,
        device_map="auto",
    )

    log.info("building in-domain calibration dataset")
    calib_ds = build_calibration_dataset(
        samples_path=args.samples,
        teacher_path=args.teacher,
        tokenizer=tokenizer,
        num_calib=args.num_calib,
        max_seq_length=args.max_seq_length,
        eval_ratio=args.eval_ratio,
    )
    log.info(
        "calibration dataset: %d rows, avg seq len ~%d tokens",
        len(calib_ds),
        sum(len(r) for r in calib_ds["input_ids"]) // max(len(calib_ds), 1),
    )

    recipe = [
        GPTQModifier(
            targets="Linear",
            scheme="W4A16",
            ignore=["lm_head"],
            dampening_frac=args.dampening_frac,
        )
    ]

    log.info("applying GPTQ W4A16 oneshot quantization...")
    oneshot(
        model=model,
        dataset=calib_ds,
        recipe=recipe,
        max_seq_length=args.max_seq_length,
        num_calibration_samples=len(calib_ds),
    )

    args.output.mkdir(parents=True, exist_ok=True)
    log.info("saving compressed-tensors checkpoint to %s", args.output)
    model.save_pretrained(str(args.output), save_compressed=True)
    tokenizer.save_pretrained(str(args.output))

    # ---- Sanity check: the saved checkpoint must load into vLLM and emit
    # a non-empty completion. This catches the
    # "compressed-tensors/config.json version drift" failure mode before
    # the downstream benchmark discovers it.
    if args.skip_sanity:
        log.info("--skip-sanity set; not verifying vLLM load")
        return 0

    log.info("sanity-checking quantized model with a short vLLM generate call")
    # Force FLASH_ATTN backend to avoid the flashinfer decode-wrapper
    # assertion triggered by compressed-tensors W4A16 on Ampere. Hard
    # override (not setdefault) because the shell env may already export
    # FLASHINFER. Must be set before the vllm import so the backend
    # selector reads the flag. See the portfolio README.
    os.environ["VLLM_ATTENTION_BACKEND"] = "FLASH_ATTN"
    try:
        from vllm import LLM, SamplingParams
    except Exception as e:  # pragma: no cover - vLLM import can fail on odd boxes
        log.warning("vLLM import failed, skipping sanity generate: %s", e)
        return 0

    # Release training-side GPU memory before loading vLLM. vLLM allocates
    # its own KV cache pool and will OOM if the bf16 copy is still resident.
    del model
    torch.cuda.empty_cache()

    llm = LLM(
        model=str(args.output),
        quantization="compressed-tensors",
        dtype="float16",
        gpu_memory_utilization=0.5,
        max_model_len=512,
        enforce_eager=True,
    )
    smoke_prompt = (
        "<|im_start|>user\nReturn the JSON {\"ok\": true}.<|im_end|>\n"
        "<|im_start|>assistant\n"
    )
    out = llm.generate([smoke_prompt], SamplingParams(max_tokens=32, temperature=0.0))
    text = out[0].outputs[0].text if out and out[0].outputs else ""
    log.info("sanity generate output: %r", text)
    if not text or len(text) < 3:
        log.error(
            "W4A16 model produced empty/near-empty output — likely a "
            "compressed-tensors format mismatch. Check config.json for "
            "quantization_config key and consider HF load_in_4bit fallback."
        )
        return 3

    log.info("W4A16 quantization complete and verified: %s", args.output)
    return 0


if __name__ == "__main__":
    sys.exit(main())

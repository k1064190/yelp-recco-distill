#!/usr/bin/env python
# ABOUTME: bitsandbytes NF4 quantization of the merged Qwen3-4B student model.
# ABOUTME: Calibration-free 4-bit NormalFloat, saves HF-loadable checkpoint.

"""
Quantize the merged Qwen3-4B student to bitsandbytes NF4 (4-bit NormalFloat).

Unlike GPTQ (``quantize_w4a16.py``), NF4 is a calibration-free method that
quantizes weights at load time using optimal quantization bins for normally-
distributed weights (Dettmers et al., "QLoRA", NeurIPS 2023). This gives a
fast, data-free quantization path that trades a small accuracy delta for zero
calibration cost.

The checkpoint is saved in the standard HF format with a ``quantization_config``
entry in ``config.json``, so downstream tools (``eval_metrics.py``,
``transformers.AutoModelForCausalLM.from_pretrained``) can load it directly.

Example:
    $ CUDA_VISIBLE_DEVICES=2 python scripts/quantize/quantize_nf4.py \\
        --model ckpt/student-merged \\
        --output ckpt/student-nf4
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("quantize_nf4")


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments.

    Returns:
        argparse.Namespace: parsed arguments with model, output, compute_dtype,
            double_quant, and skip_sanity fields.
    """
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--model",
        type=Path,
        default=PROJECT_ROOT / "ckpt/student-merged",
        help="LoRA-merged Qwen3-4B checkpoint to quantize",
    )
    p.add_argument(
        "--output",
        type=Path,
        default=PROJECT_ROOT / "ckpt/student-nf4",
        help="output directory for the NF4 checkpoint",
    )
    p.add_argument(
        "--compute-dtype",
        type=str,
        default="bf16",
        choices=["bf16", "fp16"],
        help="compute dtype for dequantized activations (default bf16)",
    )
    p.add_argument(
        "--double-quant",
        action="store_true",
        default=True,
        help="enable double quantization (quantize the quantization constants, saves ~0.4 bits/param)",
    )
    p.add_argument(
        "--no-double-quant",
        action="store_true",
        help="disable double quantization",
    )
    p.add_argument(
        "--skip-sanity",
        action="store_true",
        help="skip post-quantization generation sanity check",
    )
    return p.parse_args()


def get_disk_size_mb(path: Path) -> float:
    """Compute total size of all files under a directory in MB.

    Args:
        path (Path): directory to measure.

    Returns:
        float: total size in megabytes.
    """
    total = sum(f.stat().st_size for f in path.rglob("*") if f.is_file())
    return total / (1024 * 1024)


def main() -> int:
    """Load merged model with NF4 quantization, save checkpoint, sanity check.

    Returns:
        int: exit code (0 = success, 2 = input error, 3 = sanity failure).
    """
    args = parse_args()

    if not args.model.exists():
        log.error("merged model path not found: %s", args.model)
        return 2

    double_quant = args.double_quant and not args.no_double_quant

    # Heavy imports deferred so --help is cheap.
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

    dtype_map = {"bf16": torch.bfloat16, "fp16": torch.float16}
    compute_dtype = dtype_map[args.compute_dtype]

    log.info("loading merged model from %s with NF4 quantization", args.model)
    log.info(
        "config: quant_type=nf4, compute_dtype=%s, double_quant=%s",
        args.compute_dtype,
        double_quant,
    )

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=compute_dtype,
        bnb_4bit_use_double_quant=double_quant,
    )

    tokenizer = AutoTokenizer.from_pretrained(str(args.model), use_fast=True)
    model = AutoModelForCausalLM.from_pretrained(
        str(args.model),
        quantization_config=bnb_config,
        device_map="auto",
    )

    log.info("model loaded successfully, saving to %s", args.output)
    args.output.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(str(args.output))
    tokenizer.save_pretrained(str(args.output))

    disk_mb = get_disk_size_mb(args.output)
    log.info("NF4 checkpoint size: %.1f MB (%.2f GB)", disk_mb, disk_mb / 1024)

    # ---- Sanity check: the saved model must generate non-empty output.
    if args.skip_sanity:
        log.info("--skip-sanity set; skipping generation check")
        return 0

    log.info("sanity-checking NF4 model with a short generate call")
    smoke_messages = [
        {"role": "user", "content": 'Return the JSON {"ok": true}.'},
    ]
    prompt_text = tokenizer.apply_chat_template(
        smoke_messages, tokenize=False, add_generation_prompt=True
    )
    inputs = tokenizer(prompt_text, return_tensors="pt").to(model.device)

    with torch.no_grad():
        out = model.generate(
            **inputs,
            max_new_tokens=32,
            do_sample=False,
            pad_token_id=tokenizer.pad_token_id,
        )
    gen_ids = out[0][inputs["input_ids"].shape[1]:]
    text = tokenizer.decode(gen_ids, skip_special_tokens=True)
    log.info("sanity generate output: %r", text)

    if not text or len(text) < 3:
        log.error(
            "NF4 model produced empty/near-empty output — "
            "check bitsandbytes installation and GPU compatibility"
        )
        return 3

    log.info("NF4 quantization complete and verified: %s", args.output)
    log.info(
        "summary: input=%.2f GB, output=%.2f GB, compression=%.1fx",
        get_disk_size_mb(args.model) / 1024,
        disk_mb / 1024,
        get_disk_size_mb(args.model) / max(disk_mb, 1),
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())

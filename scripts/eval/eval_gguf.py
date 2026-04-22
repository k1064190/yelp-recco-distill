#!/usr/bin/env python
# ABOUTME: Evaluate a GGUF-quantized student model using llama-cpp-python.
# ABOUTME: Computes the same Recall/MRR/tau metrics as eval_metrics.py.

"""
Evaluate a GGUF-quantized Qwen3-4B student on the eval split.

Uses llama-cpp-python for inference instead of HF transformers. Computes
the same metrics as ``eval_metrics.py``:
  - Recall@{1,5,10} and MRR@10 vs positive_business_id
  - Top-1 agreement and Kendall tau vs teacher ranking
  - JSON parse success rate

Example:
    $ CUDA_VISIBLE_DEVICES=3 python scripts/eval/eval_gguf.py \\
        --gguf ckpt/student-gguf-q4km/student-q4-k-m.gguf \\
        --tag gguf-q4km
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from configs.teacher_prompt import SYSTEM_INSTRUCTION, build_user_prompt  # noqa: E402
from scripts.eval.eval_metrics import (  # noqa: E402
    candidates_for_examples,
    extract_student_ranking,
    format_summary_table,
    metrics_against_positive,
    metrics_against_teacher,
    positives_for_examples,
    teacher_rankings_for_examples,
)
from scripts.train.train_student import load_and_filter, split_examples  # noqa: E402

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("eval_gguf")


def build_chatml_prompt(sample: dict[str, Any]) -> str:
    """Build a ChatML-formatted prompt string for a single sample.

    Uses the same system instruction and user prompt format as the HF
    eval pipeline, but renders to raw text since llama-cpp-python
    doesn't use a tokenizer's apply_chat_template.

    Args:
        sample (dict): one processed sample record with history,
            candidates, etc.

    Returns:
        str: ChatML-formatted prompt ending with <|im_start|>assistant\\n.
    """
    user_text = build_user_prompt(sample)
    return (
        f"<|im_start|>system\n{SYSTEM_INSTRUCTION}<|im_end|>\n"
        f"<|im_start|>user\n{user_text}<|im_end|>\n"
        f"<|im_start|>assistant\n"
    )


def generate_gguf_rankings(
    llm,
    examples: list[dict[str, Any]],
    max_tokens: int,
) -> tuple[list[list[str]], list[str]]:
    """Run the GGUF model on eval examples and parse rankings.

    Sequential generation (no batching in llama-cpp-python CPU/single-GPU
    mode). Each sample's output is parsed via ``extract_student_ranking``.

    Args:
        llm: a llama_cpp.Llama instance.
        examples (list[dict]): joined eval records.
        max_tokens (int): cap on generated tokens per sample.

    Returns:
        tuple: (rankings, raw_texts) — rankings[i] is a valid 10-element
            list or empty list. raw_texts[i] is the decoded output.
    """
    rankings: list[list[str]] = []
    raw_texts: list[str] = []
    cand_sets = candidates_for_examples(examples)

    for i, ex in enumerate(examples):
        prompt = build_chatml_prompt(ex["sample"])
        output = llm(
            prompt,
            max_tokens=max_tokens,
            temperature=0.0,
            top_p=1.0,
            stop=["<|im_end|>", "<|endoftext|>"],
            echo=False,
        )
        text = output["choices"][0]["text"] if output["choices"] else ""
        raw_texts.append(text)
        ranking = extract_student_ranking(text, cand_sets[i])
        rankings.append(ranking or [])

        log.info(
            "generated %d / %d (valid=%s)",
            i + 1,
            len(examples),
            "yes" if ranking else "NO",
        )

    return rankings, raw_texts


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments.

    Returns:
        argparse.Namespace: parsed args with gguf, samples, teacher,
            eval_ratio, max_tokens, n_gpu_layers, ctx_size, tag, output fields.
    """
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--gguf",
        type=Path,
        required=True,
        help="path to the GGUF model file",
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
        "--eval-ratio",
        type=float,
        default=0.9,
        help="train fraction (keep in sync with train_student.py)",
    )
    p.add_argument("--max-tokens", type=int, default=1024)
    p.add_argument(
        "--n-gpu-layers",
        type=int,
        default=-1,
        help="number of layers to offload to GPU (-1 = all)",
    )
    p.add_argument(
        "--ctx-size",
        type=int,
        default=4096,
        help="context size in tokens",
    )
    p.add_argument(
        "--max-samples",
        type=int,
        default=None,
        help="cap eval samples (for smoke tests)",
    )
    p.add_argument(
        "--tag",
        type=str,
        default="gguf-q4km",
        help="label for the output file",
    )
    p.add_argument(
        "--output",
        type=Path,
        default=None,
        help="output JSON path (default data/results/eval_<tag>.json)",
    )
    p.add_argument(
        "--raw-out",
        type=Path,
        default=None,
        help="optional path to dump raw model text outputs for debugging",
    )
    return p.parse_args()


def main() -> int:
    """Load GGUF model, run eval on eval split, report metrics.

    Returns:
        int: exit code (0 = success, 2 = input error).
    """
    args = parse_args()
    out_path = args.output or (
        PROJECT_ROOT / "data/results" / f"eval_{args.tag}.json"
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)

    if not args.gguf.exists():
        log.error("GGUF model not found: %s", args.gguf)
        return 2

    # ---- 1. Load + split -------------------------------------------------------
    examples, _stats = load_and_filter(args.samples, args.teacher)
    train_exs, eval_exs = split_examples(examples, ratio=args.eval_ratio)
    eval_examples = eval_exs
    if args.max_samples:
        eval_examples = eval_examples[: args.max_samples]
    log.info("evaluating on %d eval-split examples", len(eval_examples))
    if not eval_examples:
        log.error("no examples to evaluate; aborting")
        return 2

    # ---- 2. Teacher baseline ---------------------------------------------------
    teacher_rankings = teacher_rankings_for_examples(eval_examples)
    positives = positives_for_examples(eval_examples)
    teacher_vs_positive = metrics_against_positive(teacher_rankings, positives)
    log.info(
        "teacher vs positive: R@1=%.3f R@5=%.3f MRR=%.3f",
        teacher_vs_positive["recall@1"],
        teacher_vs_positive["recall@5"],
        teacher_vs_positive["mrr@10"],
    )

    summary: dict[str, Any] = {
        "tag": args.tag,
        "split": "eval",
        "n_samples": len(eval_examples),
        "model": str(args.gguf),
        "positive_metrics": {"teacher": teacher_vs_positive},
    }

    # ---- 3. Load GGUF model ----------------------------------------------------
    from llama_cpp import Llama

    log.info("loading GGUF model from %s", args.gguf)
    llm = Llama(
        model_path=str(args.gguf),
        n_gpu_layers=args.n_gpu_layers,
        n_ctx=args.ctx_size,
        verbose=False,
    )
    log.info("GGUF model loaded (n_gpu_layers=%d, n_ctx=%d)", args.n_gpu_layers, args.ctx_size)

    # ---- 4. Student generation + parse -----------------------------------------
    student_rankings, raw_texts = generate_gguf_rankings(
        llm,
        eval_examples,
        max_tokens=args.max_tokens,
    )
    valid = sum(1 for r in student_rankings if r)
    summary["parsing"] = {
        "total": len(student_rankings),
        "valid": valid,
        "valid_rate": valid / len(student_rankings) if student_rankings else 0.0,
    }
    log.info(
        "parsing: %d/%d (%.1f%%) valid rankings",
        valid,
        len(student_rankings),
        100 * valid / len(student_rankings) if student_rankings else 0,
    )

    # ---- 5. Metrics ------------------------------------------------------------
    student_vs_positive = metrics_against_positive(student_rankings, positives)
    summary["positive_metrics"]["student"] = student_vs_positive
    log.info(
        "student vs positive: R@1=%.3f R@5=%.3f MRR=%.3f",
        student_vs_positive["recall@1"],
        student_vs_positive["recall@5"],
        student_vs_positive["mrr@10"],
    )

    teacher_agreement = metrics_against_teacher(student_rankings, teacher_rankings)
    summary["teacher_agreement"] = teacher_agreement
    log.info(
        "student vs teacher: top1=%.3f tau=%.3f (n_valid=%d)",
        teacher_agreement["top1_agreement"],
        teacher_agreement["kendall_tau_mean"],
        teacher_agreement["kendall_tau_valid_n"],
    )

    # ---- 6. Persist ------------------------------------------------------------
    print(format_summary_table(summary))
    out_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False))
    log.info("wrote summary to %s", out_path)

    if args.raw_out:
        args.raw_out.parent.mkdir(parents=True, exist_ok=True)
        args.raw_out.write_text(
            "\n---\n".join(f"[{i}] {t}" for i, t in enumerate(raw_texts)),
            encoding="utf-8",
        )
        log.info("wrote raw outputs to %s", args.raw_out)

    return 0


if __name__ == "__main__":
    sys.exit(main())

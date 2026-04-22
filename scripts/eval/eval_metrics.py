#!/usr/bin/env python
# ABOUTME: Classical retrieval metrics (Recall@k, MRR, rank correlation) for the
# ABOUTME: Yelp KD pipeline — index-based student/teacher vs positive ground truth.

"""
Compute classical retrieval metrics on a trained student model.

The student is asked to produce a ranked list of 10 candidate indices (1-based)
from a per-user history prompt. At eval time, indices are mapped back to
business_ids via the ordered candidate list. We score three things on the eval
split:

  1. Recall@k (k=1,5,10) and MRR@10 against the Yelp-observed next visit
     (``positive_business_id``). The positive is always one of the 10
     candidates (see scripts/data/preprocess_yelp.py), so Recall@10 for any
     schema-valid ranking is trivially 1.0 — the informative signals are
     Recall@1, Recall@5, and MRR.

  2. Teacher baseline on the same eval split, computed from the already-
     stored teacher rankings in ``philly_teacher.jsonl``. No new Gemini
     calls. The teacher prompt does NOT leak ``positive_business_id`` (see
     configs/teacher_prompt.py), so the teacher's Recall@k is a legitimate
     upper bound for what a black-box-KD student can achieve.

  3. Student-vs-teacher agreement: top-1 match rate and mean Kendall tau of
     the full 10-element ranking (distillation fidelity).

Outputs:
  - ``data/results/eval_<tag>.json``  structured summary
  - stdout                            markdown table for README pasting

Example:
    $ python scripts/eval/eval_metrics.py \\
        --model ckpt/student-merged --tag merged \\
        --batch-size 4 --max-new-tokens 1024
"""

from __future__ import annotations

import argparse
import json
import logging
import re
import sys
from pathlib import Path
from typing import Any

# Project root so we can import scripts.* and configs.*
PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from configs.teacher_prompt import SYSTEM_INSTRUCTION, build_user_prompt  # noqa: E402
from scripts.train.train_student import load_and_filter, split_examples  # noqa: E402

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("eval_metrics")


# ---------- Pure metric helpers (unit-tested) ----------


def metrics_against_positive(
    rankings: list[list[str]],
    positives: list[str],
) -> dict[str, float]:
    """Compute Recall@{1,5,10}, MRR@10, and NDCG@{5,10} against positive_business_id.

    NDCG: single-label case reduces to ``1 / log2(rank+1)`` if the positive
    is within top-k, else 0. IDCG = 1 (ideal ordering puts positive at
    rank 1). So NDCG@k is recall@k weighted by rank-position discount.

    Args:
        rankings (list[list[str]]): per-sample predicted ranking of candidate
            business_ids, best-fit first. Invalid rankings should be passed
            as empty lists so their contribution to every metric is 0.
        positives (list[str]): per-sample ground-truth positive business_id.
            Must be the same length as ``rankings``.

    Returns:
        dict: keys recall@1, recall@5, recall@10, mrr@10, ndcg@5, ndcg@10,
            n_evaluated. All metrics are in [0, 1]; n_evaluated == len(rankings).
    """
    import math

    if not rankings:
        raise ValueError("empty rankings list")
    if len(rankings) != len(positives):
        raise ValueError(
            f"length mismatch: {len(rankings)} rankings vs {len(positives)} positives"
        )
    n = len(rankings)
    r1 = r5 = r10 = 0
    mrr = 0.0
    ndcg5 = 0.0
    ndcg10 = 0.0
    for ranking, positive in zip(rankings, positives):
        if not ranking or positive not in ranking:
            continue
        rank = ranking.index(positive) + 1  # 1-indexed
        if rank <= 1:
            r1 += 1
        if rank <= 5:
            r5 += 1
        if rank <= 10:
            r10 += 1
        mrr += 1.0 / rank
        gain = 1.0 / math.log2(rank + 1)  # IDCG = 1.0 at rank 1
        if rank <= 5:
            ndcg5 += gain
        if rank <= 10:
            ndcg10 += gain
    return {
        "recall@1": r1 / n,
        "recall@5": r5 / n,
        "recall@10": r10 / n,
        "mrr@10": mrr / n,
        "ndcg@5": ndcg5 / n,
        "ndcg@10": ndcg10 / n,
        "n_evaluated": n,
    }


def kendall_tau(a: list[str], b: list[str]) -> float | None:
    """Compute Kendall tau rank correlation between two orderings.

    Uses the basic tau (not tau-b) formula: (C - D) / (n*(n-1)/2) where C/D
    are counts of concordant/discordant pairs. Requires ``a`` and ``b`` to
    be permutations of the same set with no ties.

    Args:
        a (list[str]): first ordering of n distinct items.
        b (list[str]): second ordering of n distinct items. Must be a
            permutation of ``a``.

    Returns:
        float | None: tau in [-1, 1], or None if the sets differ or n < 2.
    """
    if len(a) < 2 or set(a) != set(b):
        return None
    if len(a) != len(b):
        return None
    pos_a = {x: i for i, x in enumerate(a)}
    pos_b = {x: i for i, x in enumerate(b)}
    items = list(pos_a.keys())
    n = len(items)
    concordant = 0
    discordant = 0
    for i in range(n):
        for j in range(i + 1, n):
            di = pos_a[items[i]] - pos_a[items[j]]
            dj = pos_b[items[i]] - pos_b[items[j]]
            sign = di * dj
            if sign > 0:
                concordant += 1
            elif sign < 0:
                discordant += 1
    total = n * (n - 1) // 2
    if total == 0:
        return None
    return (concordant - discordant) / total


def metrics_against_teacher(
    student_rankings: list[list[str]],
    teacher_rankings: list[list[str]],
) -> dict[str, float]:
    """Compute top-1 agreement rate and mean Kendall tau vs teacher.

    A pair (student, teacher) contributes to tau only when both are non-empty
    and both rank exactly the same candidate set (a prerequisite for a
    well-defined Kendall tau). Samples where the student output was invalid
    are silently excluded from tau but still counted in top-1 agreement
    (as a miss).

    Args:
        student_rankings (list[list[str]]): per-sample student ranking
            (empty list if the student's output was invalid).
        teacher_rankings (list[list[str]]): per-sample teacher ranking, same
            length as ``student_rankings``.

    Returns:
        dict: keys top1_agreement (float in [0, 1] over all samples),
            kendall_tau_mean (mean over valid pairs only), kendall_tau_valid_n
            (number of pairs contributing to tau).
    """
    if len(student_rankings) != len(teacher_rankings):
        raise ValueError(
            f"length mismatch: {len(student_rankings)} vs {len(teacher_rankings)}"
        )
    n = len(student_rankings)
    if n == 0:
        return {"top1_agreement": 0.0, "kendall_tau_mean": 0.0, "kendall_tau_valid_n": 0}
    top1 = 0
    taus: list[float] = []
    for s, t in zip(student_rankings, teacher_rankings):
        if s and t and s[0] == t[0]:
            top1 += 1
        tau = kendall_tau(s, t)
        if tau is not None:
            taus.append(tau)
    return {
        "top1_agreement": top1 / n,
        "kendall_tau_mean": sum(taus) / len(taus) if taus else 0.0,
        "kendall_tau_valid_n": len(taus),
    }


def parse_json_ranking(text: str) -> dict | None:
    """Robustly extract a JSON object from a (possibly noisy) model output.

    Handles bare JSON, JSON wrapped in a ```json ... ``` markdown fence, JSON
    followed by trailing reasoning text, and bracketed noise. Returns None
    on any parse failure (including truncation).

    Args:
        text (str): raw decoded generation string.

    Returns:
        dict | None: parsed top-level object, or None on any failure.
    """
    # Prefer a markdown-fenced JSON block if present.
    fence_match = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.DOTALL)
    if fence_match:
        try:
            return json.loads(fence_match.group(1))
        except json.JSONDecodeError:
            pass
    # Find the outermost balanced {...}.
    start = text.find("{")
    if start < 0:
        return None
    depth = 0
    end = None
    for i in range(start, len(text)):
        c = text[i]
        if c == "{":
            depth += 1
        elif c == "}":
            depth -= 1
            if depth == 0:
                end = i + 1
                break
    if end is None:
        return None
    try:
        return json.loads(text[start:end])
    except json.JSONDecodeError:
        return None


def extract_student_ranking(
    text: str, candidates: list[dict[str, Any]]
) -> list[str] | None:
    """Extract a valid index-based ranking and convert to business_ids.

    The ranking must parse as JSON, contain a top-level ``ranking`` list of
    integers that form a permutation of {1, ..., N}. The indices are then
    mapped to business_ids via the ordered candidate list.

    Args:
        text (str): raw decoded generation string.
        candidates (list[dict]): ordered candidate list from the preprocessed
            sample. Each entry must have a ``business_id`` key.

    Returns:
        list[str] | None: business_id ranking if valid, else None.
    """
    obj = parse_json_ranking(text)
    if obj is None:
        return None
    ranking = obj.get("ranking")
    if not isinstance(ranking, list):
        return None
    n = len(candidates)
    if not all(isinstance(x, int) for x in ranking):
        return None
    if len(ranking) != n:
        return None
    if set(ranking) != set(range(1, n + 1)):
        return None
    return [candidates[idx - 1]["business_id"] for idx in ranking]


# ---------- Data extraction from joined examples ----------


def teacher_rankings_for_examples(
    examples: list[dict[str, Any]],
) -> list[list[str]]:
    """Extract per-example teacher ranking, mapping indices to business_ids.

    Teacher rankings are stored as lists of 1-based candidate indices. This
    function converts them to business_id lists using the ordered candidate
    list from each sample.

    Args:
        examples (list[dict]): joined records from ``load_and_filter`` —
            each has a ``teacher`` entry whose ``teacher_output.ranking`` is
            a list of 1-based int indices, and a ``sample`` entry with the
            ordered candidate list.

    Returns:
        list[list[str]]: per-sample teacher ranking as business_ids.
    """
    result: list[list[str]] = []
    for ex in examples:
        ranking_indices = ex["teacher"]["teacher_output"]["ranking"]
        candidates = ex["sample"]["candidates"]
        result.append([candidates[idx - 1]["business_id"] for idx in ranking_indices])
    return result


def positives_for_examples(examples: list[dict[str, Any]]) -> list[str]:
    """Return ground-truth positive_business_id per example.

    Args:
        examples (list[dict]): joined records.

    Returns:
        list[str]: positive business_id per sample.
    """
    return [ex["sample"]["positive_business_id"] for ex in examples]


def candidates_for_examples(
    examples: list[dict[str, Any]],
) -> list[list[dict[str, Any]]]:
    """Return the ordered candidate list per example.

    The order matches the prompt numbering (1-based), so candidate index i
    maps to candidates[i-1].

    Args:
        examples (list[dict]): joined records.

    Returns:
        list[list[dict]]: one ordered candidate list per sample.
    """
    return [ex["sample"]["candidates"] for ex in examples]


# ---------- Student generation ----------


def generate_student_rankings(
    model,
    tokenizer,
    examples: list[dict[str, Any]],
    batch_size: int,
    max_new_tokens: int,
    max_prompt_length: int = 4096,
) -> tuple[list[list[str]], list[str]]:
    """Run the student model on eval examples and parse rankings.

    Batched greedy decoding with left-padding. Each sample's generation is
    parsed via ``extract_student_ranking`` which converts 1-based index
    rankings to business_id lists; failed parses are recorded as empty lists
    so downstream metrics attribute them 0 contribution.

    Args:
        model: loaded HF causal LM (expected to be on CUDA, dtype bf16).
        tokenizer: matching tokenizer with ``padding_side='left'``.
        examples (list[dict]): joined eval records.
        batch_size (int): per-forward batch size for generation.
        max_new_tokens (int): cap on generated tokens per sample.
        max_prompt_length (int): truncate prompts to this length (in tokens)
            before padding; matches train_student.py ``max_length``.

    Returns:
        tuple: (rankings, raw_texts) — rankings[i] is a valid 10-element
            business_id list or an empty list if parsing failed. raw_texts[i]
            is the decoded generation for debugging.
    """
    import torch

    model.eval()
    rankings: list[list[str]] = []
    raw_texts: list[str] = []
    cand_lists = candidates_for_examples(examples)

    for batch_start in range(0, len(examples), batch_size):
        batch = examples[batch_start : batch_start + batch_size]
        batch_cands = cand_lists[batch_start : batch_start + batch_size]

        messages_batch = []
        for ex in batch:
            user_text = build_user_prompt(ex["sample"])
            messages_batch.append(
                [
                    {"role": "system", "content": SYSTEM_INSTRUCTION},
                    {"role": "user", "content": user_text},
                ]
            )
        # enable_thinking=False suppresses Qwen3.5 thinking mode
        # (chain-of-thought preamble before the actual JSON response).
        # Without this, 9B+ models emit "Thinking Process:..." text
        # that fails JSON parsing despite correct SFT training.
        chat_kwargs = dict(tokenize=False, add_generation_prompt=True)
        try:
            # Qwen3.5 tokenizers accept enable_thinking; others may not.
            tokenizer.apply_chat_template(
                [{"role": "user", "content": "test"}],
                enable_thinking=False, **chat_kwargs,
            )
            chat_kwargs["enable_thinking"] = False
        except TypeError:
            pass
        text_batch = [
            tokenizer.apply_chat_template(m, **chat_kwargs)
            for m in messages_batch
        ]
        enc = tokenizer(
            text_batch,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=max_prompt_length,
        ).to(model.device)

        with torch.no_grad():
            out = model.generate(
                input_ids=enc["input_ids"],
                attention_mask=enc["attention_mask"],
                max_new_tokens=max_new_tokens,
                do_sample=False,
                pad_token_id=tokenizer.pad_token_id,
            )

        prompt_len = enc["input_ids"].shape[1]
        for i, full in enumerate(out):
            gen_ids = full[prompt_len:]
            text = tokenizer.decode(gen_ids, skip_special_tokens=True)
            raw_texts.append(text)
            ranking = extract_student_ranking(text, batch_cands[i])
            rankings.append(ranking or [])

        log.info(
            "generated %d / %d",
            min(batch_start + batch_size, len(examples)),
            len(examples),
        )

    return rankings, raw_texts


# ---------- Reporting ----------


def _random_mrr10() -> float:
    """Expected MRR@10 under a uniform-random ranking of 10 candidates.

    Returns:
        float: sum_{r=1..10} (1/r) / 10 ≈ 0.2929.
    """
    return sum(1.0 / r for r in range(1, 11)) / 10.0


def format_summary_table(summary: dict[str, Any]) -> str:
    """Render the metrics summary as a markdown table for the README.

    Args:
        summary (dict): full metric summary produced by ``main``.

    Returns:
        str: multi-line markdown snippet with a main table plus footnotes.
    """
    positive = summary.get("positive_metrics", {})
    lines = [
        "| Row | Recall@1 | Recall@5 | Recall@10 | MRR@10 | NDCG@5 | NDCG@10 |",
        "|---|---|---|---|---|---|---|",
        f"| Random (1/10) | {1/10:.3f} | {5/10:.3f} | {10/10:.3f} | {_random_mrr10():.3f} | — | — |",
    ]
    if "teacher" in positive:
        t = positive["teacher"]
        lines.append(
            f"| Teacher (upper bound) | {t['recall@1']:.3f} | {t['recall@5']:.3f} "
            f"| {t['recall@10']:.3f} | {t['mrr@10']:.3f} "
            f"| {t.get('ndcg@5', 0):.3f} | {t.get('ndcg@10', 0):.3f} |"
        )
    if "student" in positive:
        s = positive["student"]
        lines.append(
            f"| Student ({summary['tag']}) | {s['recall@1']:.3f} | {s['recall@5']:.3f} "
            f"| {s['recall@10']:.3f} | {s['mrr@10']:.3f} "
            f"| {s.get('ndcg@5', 0):.3f} | {s.get('ndcg@10', 0):.3f} |"
        )
    lines.append("")
    if "teacher_agreement" in summary:
        ag = summary["teacher_agreement"]
        lines.append(
            f"**Student vs Teacher (distillation fidelity)**: "
            f"top-1 agreement {ag['top1_agreement']:.3f}, "
            f"mean Kendall tau {ag['kendall_tau_mean']:.3f} "
            f"(n_valid={ag['kendall_tau_valid_n']})"
        )
    if "parsing" in summary:
        p = summary["parsing"]
        lines.append(
            f"**Parsing**: {p['valid']}/{p['total']} "
            f"({p['valid_rate']:.1%}) schema-valid student rankings"
        )
    return "\n".join(lines)


# ---------- Orchestration ----------


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--model",
        type=str,
        default=str(PROJECT_ROOT / "ckpt/student-merged"),
        help="student model path or HF id",
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
        "--split",
        type=str,
        default="eval",
        choices=["eval", "train", "all"],
        help="which deterministic split to evaluate on (default eval)",
    )
    p.add_argument(
        "--eval-ratio",
        type=float,
        default=0.9,
        help="train fraction in the hash split — keep in sync with train_student.py",
    )
    p.add_argument("--batch-size", type=int, default=4)
    p.add_argument("--max-new-tokens", type=int, default=1024)
    p.add_argument(
        "--max-prompt-length",
        type=int,
        default=4096,
        help="tokenizer truncation cap on the prompt (matches training max_length)",
    )
    p.add_argument(
        "--max-samples",
        type=int,
        default=None,
        help="cap eval samples (for smoke tests)",
    )
    p.add_argument(
        "--attn-impl",
        type=str,
        default="flash_attention_2",
        choices=["sdpa", "flash_attention_2", "eager"],
    )
    p.add_argument(
        "--dtype",
        type=str,
        default="bf16",
        choices=["bf16", "fp16"],
        help=(
            "compute dtype. Use 'bf16' for vanilla Qwen3-4B / LoRA-merged "
            "checkpoints (matches training precision). Use 'fp16' for "
            "compressed-tensors W4A16 models — compressed-tensors dequant "
            "kernels require fp16 compute on Ampere (see plan §3)."
        ),
    )
    p.add_argument(
        "--tag",
        type=str,
        default="merged",
        help="label for the output file, e.g. 'merged' or 'w4a16'",
    )
    p.add_argument(
        "--output",
        type=Path,
        default=None,
        help="output JSON summary path (default data/results/eval_<tag>.json)",
    )
    p.add_argument(
        "--teacher-only",
        action="store_true",
        help="skip student generation; report teacher baseline only",
    )
    p.add_argument(
        "--raw-out",
        type=Path,
        default=None,
        help="optional path to dump raw student text outputs for debugging",
    )
    return p.parse_args()


def main() -> int:
    args = parse_args()
    out_path = args.output or (
        PROJECT_ROOT / "data/results" / f"eval_{args.tag}.json"
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)

    # ---- 1. Load + split -----------------------------------------------------------
    examples, _stats = load_and_filter(args.samples, args.teacher)
    if args.split == "all":
        eval_examples = examples
    else:
        train_exs, eval_exs = split_examples(examples, ratio=args.eval_ratio)
        eval_examples = train_exs if args.split == "train" else eval_exs
    if args.max_samples:
        eval_examples = eval_examples[: args.max_samples]
    log.info("evaluating on %d %s-split examples", len(eval_examples), args.split)
    if not eval_examples:
        log.error("no examples to evaluate; aborting")
        return 2

    # ---- 2. Teacher baseline vs positive -------------------------------------------
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
        "split": args.split,
        "n_samples": len(eval_examples),
        "model": args.model,
        "positive_metrics": {"teacher": teacher_vs_positive},
    }

    if args.teacher_only:
        print(format_summary_table(summary))
        out_path.write_text(
            json.dumps(summary, indent=2, ensure_ascii=False)
        )
        log.info("wrote teacher-only summary to %s", out_path)
        return 0

    # ---- 3. Load student model -----------------------------------------------------
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    dtype_map = {"bf16": torch.bfloat16, "fp16": torch.float16}
    compute_dtype = dtype_map[args.dtype]
    log.info("loading student model from %s (dtype=%s)", args.model, args.dtype)
    tokenizer = AutoTokenizer.from_pretrained(
        args.model, use_fast=True, padding_side="left"
    )
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        dtype=compute_dtype,
        attn_implementation=args.attn_impl,
        device_map="cuda",
    )
    model.config.use_cache = True  # inference, not training

    # ---- 4. Student generation + parse ---------------------------------------------
    student_rankings, raw_texts = generate_student_rankings(
        model,
        tokenizer,
        eval_examples,
        batch_size=args.batch_size,
        max_new_tokens=args.max_new_tokens,
        max_prompt_length=args.max_prompt_length,
    )
    valid = sum(1 for r in student_rankings if r)
    summary["parsing"] = {
        "total": len(student_rankings),
        "valid": valid,
        "valid_rate": valid / len(student_rankings),
    }
    log.info("parsing: %d/%d (%.1f%%) valid rankings", valid, len(student_rankings), 100 * valid / len(student_rankings))

    # ---- 5. Student metrics --------------------------------------------------------
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

    # ---- 6. Persist ----------------------------------------------------------------
    print(format_summary_table(summary))
    out_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False))
    log.info("wrote summary to %s", out_path)

    if args.raw_out:
        args.raw_out.parent.mkdir(parents=True, exist_ok=True)
        args.raw_out.write_text(
            "\n---\n".join(f"[{i}] {t}" for i, t in enumerate(raw_texts)),
            encoding="utf-8",
        )
        log.info("wrote raw student outputs to %s", args.raw_out)

    return 0


if __name__ == "__main__":
    sys.exit(main())

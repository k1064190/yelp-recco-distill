#!/usr/bin/env python
# ABOUTME: Retrieval metrics via vLLM serve with guided JSON decoding.
# ABOUTME: Fair comparison between base and SFT models under constrained output.

"""
Eval retrieval metrics using a vLLM OpenAI-compatible endpoint with
response_format=json_schema to enforce structured output.

Unlike eval_metrics.py (HF transformers free generation), this script
forces ALL models to produce schema-valid JSON via guided decoding.
This enables fair comparison between base models (no SFT) and fine-tuned
models — the question becomes "does SFT improve ranking quality?" rather
than "can the model produce valid JSON?".

Example:
    $ python scripts/eval/eval_metrics_vllm.py \\
        --url http://localhost:8200/v1 \\
        --model-name student-fp16 \\
        --teacher data/teacher/philly_teacher_qwen35_http_identity.jsonl \\
        --tag v3-sft-vllm
"""

from __future__ import annotations

import argparse
import json
import logging
import statistics
import sys
import time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from configs.teacher_prompt import (  # noqa: E402
    SYSTEM_INSTRUCTION as _DEFAULT_SYSTEM_INSTRUCTION,
    TeacherResponse,
    build_user_prompt,
)

# Module-level hook; main() may override via ``--prompt-module`` (importlib).
SYSTEM_INSTRUCTION = _DEFAULT_SYSTEM_INSTRUCTION
from scripts.eval.eval_metrics import (  # noqa: E402
    candidates_for_examples,
    extract_student_ranking,
    kendall_tau,
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
log = logging.getLogger("eval_vllm")


def _build_payload(
    messages: list[dict],
    model_name: str,
    max_tokens: int,
    guided: bool,
    schema: dict,
    enable_thinking: bool = False,
) -> dict:
    """Build a chat-completions payload.

    Qwen3.5 is a thinking model — vLLM's chat template emits a `<think>`
    preamble unless `chat_template_kwargs.enable_thinking=False` is passed at
    the TOP LEVEL of the request body (not nested under `extra_body`). With
    guided JSON the grammar mask preempts the thinking text but the model
    still wastes internal tokens; without guided it produces zero valid JSON.
    We always disable thinking for consistent cross-variant eval.
    """
    payload = {
        "model": model_name,
        "messages": messages,
        "max_tokens": max_tokens,
        "temperature": 0,
        "chat_template_kwargs": {"enable_thinking": enable_thinking},
    }
    if guided:
        payload["response_format"] = {
            "type": "json_schema",
            "json_schema": {
                "name": "TeacherResponse",
                "schema": schema,
            },
        }
    return payload


def _generate_one(
    idx: int,
    ex: dict,
    cands: list[dict],
    url: str,
    model_name: str,
    max_tokens: int,
    guided: bool,
    schema: dict,
    request_timeout: int,
    enable_thinking: bool = False,
) -> dict:
    """Generate one sample; return a per-record dict with raw + parsed fields.

    Returns:
        dict with keys ``index``, ``sample_id``, ``raw_text``, ``ranking`` (list
        of business_ids or None), ``parsed_output`` (dict with
        ``ranking`` / ``persona`` / ``rationales`` when parseable, else None),
        ``finish_reason``, ``output_tokens`` (int or None), ``error``.
    """
    import requests

    user_text = build_user_prompt(ex["sample"])
    messages = [
        {"role": "system", "content": SYSTEM_INSTRUCTION},
        {"role": "user", "content": user_text},
    ]
    payload = _build_payload(
        messages, model_name, max_tokens, guided, schema,
        enable_thinking=enable_thinking,
    )

    record = {
        "index": idx,
        "sample_id": ex["sample"]["sample_id"],
        "raw_text": "",
        "ranking": None,
        "parsed_output": None,
        "finish_reason": None,
        "output_tokens": None,
        "error": None,
    }
    try:
        resp = requests.post(
            f"{url}/chat/completions", json=payload, timeout=request_timeout,
        )
        data = resp.json()
        if "choices" not in data:
            record["error"] = data.get("message", str(data))[:500]
            return record
        ch = data["choices"][0]
        record["raw_text"] = ch["message"]["content"] or ""
        record["finish_reason"] = ch.get("finish_reason")
        if "usage" in data and isinstance(data["usage"], dict):
            record["output_tokens"] = data["usage"].get("completion_tokens")
    except Exception as e:
        record["error"] = str(e)[:500]
        return record

    ranking = extract_student_ranking(record["raw_text"], cands)
    record["ranking"] = ranking or None

    if ranking:
        try:
            parsed = json.loads(record["raw_text"])
            record["parsed_output"] = {
                "ranking": parsed.get("ranking"),
                "persona": parsed.get("persona"),
                "rationales": parsed.get("rationales"),
            }
        except Exception:
            pass

    return record


def generate_rankings_vllm(
    examples: list[dict[str, Any]],
    cand_lists: list[list[dict[str, Any]]],
    url: str,
    model_name: str,
    max_tokens: int = 1536,
    guided: bool = True,
    concurrency: int = 8,
    request_timeout: int = 240,
    enable_thinking: bool = False,
    stream_path: Path | None = None,
    stream_meta: dict | None = None,
) -> tuple[list[list[str]], list[dict]]:
    """Generate rankings concurrently via vLLM OpenAI-compatible endpoint.

    Args:
        examples: joined records from load_and_filter.
        cand_lists: per-sample candidate list.
        url: vLLM OpenAI base URL.
        model_name: served model name.
        max_tokens: max output tokens per request.
        guided: if True, force TeacherResponse JSON schema via response_format.
        concurrency: number of parallel in-flight HTTP requests. vLLM's
            continuous batching amortises the teacher forward passes, so
            client-side parallelism dominates wall-clock speedup.
        request_timeout: per-request HTTP read timeout in seconds.
        stream_path: when provided, append each record as JSONL as it
            completes (out-of-order, includes ``index`` for downstream sort).
            Enables in-flight inspection during long runs. File is truncated
            on start.
        stream_meta: extra fields (``tag``, ``eval_tag``, ``guided_decoding``)
            folded into every streamed line to match the final ``_raw.jsonl``
            shape written by :func:`write_raw_jsonl`.

    Returns:
        tuple of (rankings, records):
          - rankings: list of length N, each an ordered list of business_ids
            or ``[]`` on parse failure.
          - records: list of length N with per-sample raw + parsed fields
            (see :func:`_generate_one`).
    """
    schema = TeacherResponse.model_json_schema()
    records: list[dict] = [None] * len(examples)
    meta = stream_meta or {}
    stream_fp = stream_path.open("w", encoding="utf-8") if stream_path else None

    try:
        with ThreadPoolExecutor(max_workers=concurrency) as pool:
            futures = {
                pool.submit(
                    _generate_one, i, ex, cands, url, model_name, max_tokens,
                    guided, schema, request_timeout, enable_thinking,
                ): i
                for i, (ex, cands) in enumerate(zip(examples, cand_lists))
            }
            done_count = 0
            valid_count = 0
            for fut in as_completed(futures):
                rec = fut.result()
                records[rec["index"]] = rec
                done_count += 1
                if rec["ranking"]:
                    valid_count += 1
                if stream_fp is not None:
                    ex = examples[rec["index"]]
                    line = {
                        "index": rec["index"],
                        "sample_id": rec["sample_id"],
                        "user_id": ex["sample"].get("user_id"),
                        "positive_business_id": ex["sample"].get("positive_business_id"),
                        "model": model_name,
                        "eval_tag": meta.get("tag"),
                        "guided_decoding": guided,
                        "raw_response": rec.get("raw_text", ""),
                        "teacher_output": rec.get("parsed_output"),
                        "finish_reason": rec.get("finish_reason"),
                        "output_tokens": rec.get("output_tokens"),
                        "error": rec.get("error"),
                    }
                    stream_fp.write(json.dumps(line, ensure_ascii=False) + "\n")
                    stream_fp.flush()
                if done_count % 20 == 0 or done_count == len(examples):
                    log.info(
                        "generated %d / %d (valid: %d, %.0f%%)",
                        done_count, len(examples), valid_count,
                        valid_count / done_count * 100,
                    )
    finally:
        if stream_fp is not None:
            stream_fp.close()

    rankings = [(r["ranking"] or []) for r in records]
    return rankings, records


def token_distribution(records: list[dict]) -> dict:
    """Aggregate per-sample ``output_tokens`` into a descriptive summary.

    Only samples with a populated ``output_tokens`` (vLLM ``usage`` block
    present) contribute. Used to compare how verbose different variants are
    under the same generation budget.
    """
    lens = [r["output_tokens"] for r in records if r.get("output_tokens") is not None]
    if not lens:
        return {"n": 0}
    sorted_lens = sorted(lens)
    def pct(p):
        k = max(0, min(len(sorted_lens) - 1, int(round(p / 100 * (len(sorted_lens) - 1)))))
        return sorted_lens[k]
    return {
        "n": len(lens),
        "mean": round(statistics.mean(lens), 1),
        "stdev": round(statistics.stdev(lens), 1) if len(lens) > 1 else 0.0,
        "min": min(lens),
        "p50": pct(50),
        "p95": pct(95),
        "p99": pct(99),
        "max": max(lens),
    }


def position_bias(
    records: list[dict],
    examples: list[dict],
    cand_lists: list[list[dict]],
) -> dict:
    """Top-1 slot distribution + GT-conditional recall@1, matching ``analyze_position_bias.py``.

    Returns a dict with:
      - ``top1_slot_counts``: length-10 list of counts (slot 1 is first-shown).
      - ``slot1_top1_rate``: fraction of top-1 predictions landing in slot 1.
      - ``chi2_uniform``: χ² statistic vs uniform expectation.
      - ``chi2_pvalue``: p-value (requires scipy; None if unavailable).
      - ``recall_by_gt_position``: recall@1 conditional on GT slot, length 10.
      - ``recall_support``: per-slot denominator for the above, length 10.
      - ``n_valid``: number of samples with a parseable ranking.
    """
    K = 10
    top1_slots: list[int] = []
    hits_by_gt: Counter[int] = Counter()
    totals_by_gt: Counter[int] = Counter()

    for rec, ex, cands in zip(records, examples, cand_lists):
        parsed = rec.get("parsed_output")
        if not parsed:
            continue
        ranking = parsed.get("ranking")
        if not (isinstance(ranking, list) and len(ranking) == K and
                all(isinstance(r, int) and 1 <= r <= K for r in ranking) and
                len(set(ranking)) == K):
            continue
        top1 = ranking[0]
        top1_slots.append(top1)

        # GT slot: find the candidate whose business_id matches positive_business_id
        pos_bid = ex["sample"].get("positive_business_id")
        gt_slot = next(
            (idx + 1 for idx, c in enumerate(cands) if c.get("business_id") == pos_bid),
            None,
        )
        if gt_slot is not None:
            totals_by_gt[gt_slot] += 1
            if top1 == gt_slot:
                hits_by_gt[gt_slot] += 1

    if not top1_slots:
        return {"n_valid": 0}

    slot_counts = [sum(1 for s in top1_slots if s == k) for k in range(1, K + 1)]
    expected = len(top1_slots) / K
    chi2 = sum((c - expected) ** 2 / expected for c in slot_counts) if expected else 0.0
    try:
        from scipy.stats import chisquare
        chi2_scipy, pval = chisquare(slot_counts)
        chi2 = float(chi2_scipy)
        pval = float(pval)
    except ImportError:
        pval = None

    recall = [
        round(hits_by_gt[g] / totals_by_gt[g], 4) if totals_by_gt[g] else None
        for g in range(1, K + 1)
    ]
    return {
        "n_valid": len(top1_slots),
        "top1_slot_counts": slot_counts,
        "slot1_top1_rate": round(slot_counts[0] / len(top1_slots), 4),
        "chi2_uniform": round(chi2, 3),
        "chi2_pvalue": pval if pval is None else round(pval, 6),
        "recall_by_gt_position": recall,
        "recall_support": [totals_by_gt[g] for g in range(1, K + 1)],
    }


def write_raw_jsonl(
    path: Path,
    records: list[dict],
    examples: list[dict],
    tag: str,
    model_name: str,
    guided: bool,
) -> None:
    """Write per-sample raw + parsed outputs in teacher-JSONL-compatible format.

    Mirrors the shape of ``data/teacher/philly_teacher_qwen35_*.jsonl`` so
    the standard ``scripts/teacher/analyze_position_bias.py`` can ingest it
    directly if desired (uses ``teacher_output.ranking``, ``error``,
    ``sample_id``).
    """
    with path.open("w", encoding="utf-8") as f:
        for rec, ex in zip(records, examples):
            out = {
                "sample_id": rec["sample_id"],
                "user_id": ex["sample"].get("user_id"),
                "positive_business_id": ex["sample"].get("positive_business_id"),
                "model": model_name,
                "eval_tag": tag,
                "guided_decoding": guided,
                "raw_response": rec.get("raw_text", ""),
                "teacher_output": rec.get("parsed_output"),
                "finish_reason": rec.get("finish_reason"),
                "output_tokens": rec.get("output_tokens"),
                "error": rec.get("error"),
            }
            f.write(json.dumps(out, ensure_ascii=False) + "\n")


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments.

    Returns:
        argparse.Namespace: parsed arguments.
    """
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--url", type=str, required=True,
                    help="vLLM OpenAI base URL (e.g. http://localhost:8200/v1)")
    p.add_argument("--model-name", type=str, required=True,
                    help="served model name for the API request")
    p.add_argument("--samples", type=Path,
                    default=PROJECT_ROOT / "data/processed/philly_samples.jsonl")
    p.add_argument("--teacher", type=Path,
                    default=PROJECT_ROOT / "data/teacher/philly_teacher.jsonl")
    p.add_argument("--split", type=str, default="eval",
                    choices=["eval", "train", "all"])
    p.add_argument("--eval-ratio", type=float, default=0.9)
    p.add_argument("--max-tokens", type=int, default=1024)
    p.add_argument("--max-samples", type=int, default=None)
    p.add_argument("--tag", type=str, default="vllm",
                    help="label for the output file")
    p.add_argument("--output", type=Path,
                    default=PROJECT_ROOT / "data/results")
    p.add_argument(
        "--no-guided-json", action="store_true",
        help=(
            "disable response_format=json_schema. The model must emit valid "
            "TeacherResponse JSON on its own. Use this to measure the raw "
            "JSON-producing capability of a model (e.g. teacher baseline "
            "without the guided safety net)."
        ),
    )
    p.add_argument(
        "--concurrency", type=int, default=8,
        help=(
            "number of parallel in-flight HTTP requests. vLLM's continuous "
            "batching lets the server absorb concurrency up to --max-num-seqs; "
            "client-side parallelism dominates wall-clock speedup. Default 8."
        ),
    )
    p.add_argument(
        "--request-timeout", type=int, default=240,
        help="per-request HTTP read timeout in seconds.",
    )
    p.add_argument(
        "--enable-thinking", action="store_true",
        help=(
            "enable Qwen3.5 thinking mode (chat_template_kwargs.enable_thinking=True). "
            "Default is OFF so bench results are thinking-free. Turn ON to match "
            "the v3 teacher data-generation setup (which did NOT pass the kwarg, "
            "so thinking ran by default) for fair slot-1 bias comparison."
        ),
    )
    p.add_argument(
        "--prompt-module", type=str, default="configs.teacher_prompt",
        help=(
            "import path to a module exporting SYSTEM_INSTRUCTION. Defaults to "
            "`configs.teacher_prompt`. Used to A/B different system prompts "
            "against the same teacher serve (e.g. "
            "`configs.teacher_prompt_example` for a JSON-example variant)."
        ),
    )
    return p.parse_args()


def main() -> int:
    """Run vLLM-based evaluation with guided JSON decoding.

    Returns:
        int: exit code (0 on success).
    """
    args = parse_args()
    args.output.mkdir(parents=True, exist_ok=True)

    # Module-level SYSTEM_INSTRUCTION override — lets callers A/B different
    # prompt variants without editing the config in-place (matters when
    # multiple concurrent processes would otherwise collide on the same file).
    if args.prompt_module and args.prompt_module != "configs.teacher_prompt":
        import importlib
        pm = importlib.import_module(args.prompt_module)
        global SYSTEM_INSTRUCTION
        SYSTEM_INSTRUCTION = pm.SYSTEM_INSTRUCTION
        log.info("loaded SYSTEM_INSTRUCTION from %s (%d chars)",
                 args.prompt_module, len(SYSTEM_INSTRUCTION))

    # Load and split data
    examples, stats = load_and_filter(args.samples, args.teacher)
    log.info("join+filter stats: %s", stats)

    if args.split == "eval":
        _, examples = split_examples(examples, ratio=args.eval_ratio)
    elif args.split == "train":
        examples, _ = split_examples(examples, ratio=args.eval_ratio)

    if args.max_samples:
        examples = examples[:args.max_samples]

    log.info("evaluating on %d %s-split examples", len(examples), args.split)

    # Teacher baselines (from stored file, no API calls)
    teacher_ranks = teacher_rankings_for_examples(examples)
    positives = positives_for_examples(examples)
    cand_lists = candidates_for_examples(examples)

    teacher_pos = metrics_against_positive(teacher_ranks, positives)
    log.info(
        "teacher vs positive: R@1=%.3f R@5=%.3f MRR=%.3f",
        teacher_pos["recall@1"], teacher_pos["recall@5"], teacher_pos["mrr@10"],
    )

    # Student generation via vLLM (guided or unconstrained)
    guided = not args.no_guided_json
    log.info(
        "generating via %s (model=%s, guided_json=%s, concurrency=%d, thinking=%s)",
        args.url, args.model_name, guided, args.concurrency, args.enable_thinking,
    )
    raw_path = args.output / f"eval_{args.tag}_raw.jsonl"
    t0 = time.perf_counter()
    student_ranks, records = generate_rankings_vllm(
        examples, cand_lists, args.url, args.model_name, args.max_tokens,
        guided=guided, concurrency=args.concurrency,
        request_timeout=args.request_timeout,
        enable_thinking=args.enable_thinking,
        stream_path=raw_path,
        stream_meta={"tag": args.tag},
    )
    elapsed = time.perf_counter() - t0
    log.info("generation done in %.1fs", elapsed)

    # Metrics
    student_pos = metrics_against_positive(student_ranks, positives)
    agreement = metrics_against_teacher(student_ranks, teacher_ranks)

    valid = sum(1 for r in student_ranks if r)
    total = len(student_ranks)

    # Distribution + bias analyses (new)
    tok_dist = token_distribution(records)
    pos_bias = position_bias(records, examples, cand_lists)

    # Failed-sample debug list reconstructed from records.
    failed_samples = [
        {"index": r["index"], "sample_id": r["sample_id"],
         "raw_text": (r.get("raw_text") or "")[:500], "error": r.get("error")}
        for r in records
        if not r.get("ranking") and (r.get("raw_text") or r.get("error"))
    ]

    # Print results
    print()
    print(f"| Model | R@1 | R@5 | R@10 | MRR | NDCG@5 | NDCG@10 | τ | Parse |")
    print(f"|-------|-----|-----|------|-----|--------|---------|---|-------|")
    print(
        f"| Teacher | {teacher_pos['recall@1']:.3f} | {teacher_pos['recall@5']:.3f} "
        f"| {teacher_pos['recall@10']:.3f} | {teacher_pos['mrr@10']:.3f} "
        f"| {teacher_pos.get('ndcg@5', 0):.3f} | {teacher_pos.get('ndcg@10', 0):.3f} | — | — |"
    )
    print(
        f"| {args.tag} | {student_pos['recall@1']:.3f} | {student_pos['recall@5']:.3f} "
        f"| {student_pos['recall@10']:.3f} | {student_pos['mrr@10']:.3f} "
        f"| {student_pos.get('ndcg@5', 0):.3f} | {student_pos.get('ndcg@10', 0):.3f} "
        f"| {agreement['kendall_tau_mean']:.3f} | {valid}/{total} "
        f"({valid/total*100:.1f}%) |"
    )
    print()
    print(
        f"**Student vs Teacher**: top-1 agreement {agreement['top1_agreement']:.3f}, "
        f"mean Kendall tau {agreement['kendall_tau_mean']:.3f} "
        f"(n_valid={agreement['kendall_tau_valid_n']})"
    )

    # Save
    result = {
        "tag": args.tag,
        "model_name": args.model_name,
        "url": args.url,
        "n_samples": total,
        "split": args.split,
        "guided_decoding": guided,
        "enable_thinking": args.enable_thinking,
        "concurrency": args.concurrency,
        "parsing": {"total": total, "valid": valid, "valid_rate": valid / total},
        "positive_metrics": {
            "teacher": teacher_pos,
            "student": student_pos,
        },
        "teacher_agreement": agreement,
        "token_distribution": tok_dist,
        "position_bias": pos_bias,
        "generation_time_s": round(elapsed, 1),
    }
    if failed_samples:
        result["failed_samples"] = failed_samples[:20]  # cap at 20
        log.info("%d samples failed parsing, saved first %d for debug",
                 len(failed_samples), min(len(failed_samples), 20))

    out_path = args.output / f"eval_{args.tag}.json"
    out_path.write_text(json.dumps(result, indent=2, ensure_ascii=False))

    # Raw per-sample JSONL (teacher-JSONL-compatible shape). Streamed
    # out-of-order during generation (see stream_path); rewrite sorted by
    # index so downstream consumers (analyze_position_bias.py) see records
    # aligned with `samples` in eval_<tag>.json.
    write_raw_jsonl(raw_path, records, examples, args.tag, args.model_name, guided)
    log.info("wrote raw JSONL to %s", raw_path)

    # Quick console line for position bias
    if pos_bias.get("n_valid", 0) > 0:
        log.info(
            "position bias: slot1=%.1f%% (n=%d), χ² p=%s",
            pos_bias["slot1_top1_rate"] * 100, pos_bias["n_valid"],
            pos_bias.get("chi2_pvalue"),
        )
    if tok_dist.get("n", 0) > 0:
        log.info(
            "output tokens: mean=%.0f p50=%d p95=%d max=%d",
            tok_dist["mean"], tok_dist["p50"], tok_dist["p95"], tok_dist["max"],
        )
    log.info("saved results to %s", out_path)
    return 0


if __name__ == "__main__":
    sys.exit(main())

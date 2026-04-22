#!/usr/bin/env python
# ABOUTME: Merge two teacher passes over different candidate permutations via
# ABOUTME: Borda count to cancel LLM position bias (Option-1 / PRP).
"""Merge two teacher passes into a Borda-aggregated ranking.

Inputs
------
- ``--pass1``: ``philly_teacher_qwen35.jsonl`` — candidates in original order.
  Records have ``teacher_output.ranking`` whose ints are 1-based indices into
  the *original* ``candidates`` list.
- ``--pass2``: ``philly_teacher_qwen35_perm_reverse.jsonl`` — candidates in
  reversed order, produced by ``scripts/teacher/generate_teacher_permutation.py``.
  Records have an extra ``permutation`` field; the ``teacher_output.ranking``
  ints are 1-based indices into the *prompt-time* candidate list, which you
  translate back with ``original_slot = permutation[prompt_slot - 1]``.

Algorithm
---------
For each ``sample_id`` present and strict-valid in both files:

1. Translate pass #2's ``ranking`` to original slot ids using the stored
   ``permutation``. Call the two per-slot 0-based rank vectors ``rank1[s]``
   and ``rank2[s]`` (``s`` is an original slot, rank 0 is best).
2. Borda score per slot: ``B[s] = (K - 1 - rank1[s]) + (K - 1 - rank2[s])``.
   Best ≈ 2(K-1). Worst = 0.
3. New ranking: sort slots by ``B[s]`` descending. Ties are broken by pass
   #1's rank (primary teacher signal wins ties, matching common PRP usage).
4. Emit a record mirroring the original teacher schema
   (``persona``, ``rationales``, ``ranking``) so existing consumers keep
   working. The merged ``persona`` and ``rationales`` are copied from pass
   #1 (pass #2's text is paraphrased but lower-signal because its rationale
   ordering is permuted — we only use its ranking).

Diagnostics
-----------
Per merged record we also store:
- ``permutation_consistency.top1_agreement``: bool — did both passes pick
  the same original slot as top-1?
- ``permutation_consistency.kendall_tau``: float — rank correlation between
  the two pass-level rankings over original slots. Close to +1 means the
  teacher was content-consistent across permutations; close to 0 or negative
  means position bias dominated.
- ``permutation_consistency.borda_scores``: list[int] of length K, indexed
  by slot-1 (``borda_scores[0]`` = slot 1).

Usage
-----
    python \\
        scripts/teacher/merge_teacher_permutations.py \\
        --pass1 data/teacher/philly_teacher_qwen35.jsonl \\
        --pass2 data/teacher/philly_teacher_qwen35_perm_reverse.jsonl \\
        --output data/teacher/philly_teacher_qwen35_borda.jsonl

Only samples strict-valid in *both* passes are written. The script also
prints summary stats (merge counts, mean τ, top-1 agreement rate).
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from configs.teacher_prompt import N_CANDIDATES  # noqa: E402

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("merge_teacher")

K = N_CANDIDATES


def is_strict_valid_ranking(r: Any) -> bool:
    """Return True iff ``r`` is a length-K permutation of ``{1..K}``.

    Args:
        r (Any): the ``ranking`` field candidate.

    Returns:
        bool: validity.
    """
    if not isinstance(r, list) or len(r) != K:
        return False
    if not all(isinstance(x, int) and 1 <= x <= K for x in r):
        return False
    return len(set(r)) == K


def load_valid(path: Path) -> dict[str, dict[str, Any]]:
    """Read a teacher jsonl and return ``sample_id -> record`` for strict-valid.

    Args:
        path (Path): teacher jsonl file.

    Returns:
        dict: keys are sample_ids; values are whole records. Only records
            with ``error is None`` and strict-valid ranking are kept.
    """
    out: dict[str, dict[str, Any]] = {}
    with path.open("r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue
            if rec.get("error") is not None:
                continue
            out_field = rec.get("teacher_output")
            if not isinstance(out_field, dict):
                continue
            if not is_strict_valid_ranking(out_field.get("ranking")):
                continue
            out[rec["sample_id"]] = rec
    return out


def ranking_to_rank_vector(ranking: list[int], k: int = K) -> list[int]:
    """Convert a 1-based ranking (best-to-worst) to a 0-based rank vector.

    Args:
        ranking (list[int]): e.g. ``[9, 5, ...]`` — slot 9 is best.
        k (int): number of slots.

    Returns:
        list[int]: ``ranks`` of length ``k`` where ``ranks[s - 1]`` is the
            0-based rank position of slot ``s`` (0 = best).
    """
    ranks = [0] * k
    for pos, slot in enumerate(ranking):
        ranks[slot - 1] = pos
    return ranks


def translate_pass2(ranking: list[int], permutation: list[int]) -> list[int]:
    """Translate pass-2 ranking from prompt-slot space to original-slot space.

    Args:
        ranking (list[int]): 1-based indices into the *permuted* candidate
            list (as returned by the teacher in pass 2).
        permutation (list[int]): 1-based permutation such that
            ``permutation[i - 1]`` is the original slot of the candidate shown
            at prompt position ``i``.

    Returns:
        list[int]: new ranking in the *original* slot space, best-to-worst.
    """
    return [permutation[x - 1] for x in ranking]


def kendall_tau(a: list[int], b: list[int]) -> float:
    """Kendall tau-b over two equal-length integer sequences.

    Args:
        a (list[int]): first sequence; interpreted as rank assignments.
        b (list[int]): second sequence aligned to ``a``.

    Returns:
        float: τ in ``[-1, 1]``. Uses tau-b via
            ``scipy.stats.kendalltau``; ``scipy`` is already a project
            dependency (see :mod:`scripts.analyze_position_bias`).
    """
    from scipy.stats import kendalltau
    res = kendalltau(a, b)
    return float(res.statistic) if res.statistic == res.statistic else 0.0


def borda_merge(ranking1: list[int], ranking2: list[int]) -> list[int]:
    """Aggregate two original-slot rankings into one via Borda count.

    Args:
        ranking1 (list[int]): pass 1 ranking, 1-based original-slot order.
        ranking2 (list[int]): pass 2 ranking, 1-based original-slot order
            (already translated via :func:`translate_pass2`).

    Returns:
        list[int]: merged ranking, 1-based, length K. Ties broken by pass 1
            rank (stable preference for the primary teacher signal).
    """
    rank1 = ranking_to_rank_vector(ranking1)
    rank2 = ranking_to_rank_vector(ranking2)
    # Borda points: top gets K-1, bottom gets 0.
    borda = [
        (K - 1 - rank1[s - 1]) + (K - 1 - rank2[s - 1])
        for s in range(1, K + 1)
    ]
    # Sort slots by descending Borda; tie-break by ascending pass-1 rank.
    slots = list(range(1, K + 1))
    slots.sort(key=lambda s: (-borda[s - 1], rank1[s - 1]))
    return slots


def merge_one(rec1: dict[str, Any], rec2: dict[str, Any]) -> dict[str, Any]:
    """Build the merged record for one sample.

    Args:
        rec1 (dict): pass 1 record (identity permutation).
        rec2 (dict): pass 2 record (non-identity permutation, has
            ``permutation`` field).

    Returns:
        dict: merged record with the same top-level shape as pass 1 plus
            a ``permutation_consistency`` diagnostics block.
    """
    ranking1 = rec1["teacher_output"]["ranking"]
    ranking2_prompt = rec2["teacher_output"]["ranking"]
    perm = rec2["permutation"]
    ranking2 = translate_pass2(ranking2_prompt, perm)

    merged_ranking = borda_merge(ranking1, ranking2)

    rank1 = ranking_to_rank_vector(ranking1)
    rank2 = ranking_to_rank_vector(ranking2)
    borda_scores = [
        (K - 1 - rank1[s - 1]) + (K - 1 - rank2[s - 1])
        for s in range(1, K + 1)
    ]
    tau = kendall_tau(rank1, rank2)
    top1_agree = ranking1[0] == ranking2[0]

    merged_output = {
        "persona": rec1["teacher_output"].get("persona"),
        "rationales": rec1["teacher_output"].get("rationales"),
        "ranking": merged_ranking,
    }

    return {
        "sample_id": rec1["sample_id"],
        "user_id": rec1["user_id"],
        "positive_business_id": rec1["positive_business_id"],
        "model": rec1.get("model"),
        "teacher_output": merged_output,
        "error": None,
        "merge": {
            "method": "borda_count_k=2",
            "source_passes": [
                {"path": "pass1", "permutation": list(range(1, K + 1))},
                {"path": "pass2", "permutation": perm},
            ],
            "pass1_ranking": ranking1,
            "pass2_ranking_original_space": ranking2,
            "borda_scores": borda_scores,
            "top1_agreement": top1_agree,
            "kendall_tau": round(tau, 4),
        },
    }


def parse_args() -> argparse.Namespace:
    """Parse CLI args.

    Returns:
        argparse.Namespace: parsed arguments.
    """
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--pass1", type=Path, required=True)
    p.add_argument("--pass2", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    return p.parse_args()


def main() -> None:
    """Entry point: load, merge, write, summarize."""
    args = parse_args()
    args.output.parent.mkdir(parents=True, exist_ok=True)

    log.info("loading pass1 %s", args.pass1)
    p1 = load_valid(args.pass1)
    log.info("pass1 strict-valid: %d", len(p1))

    log.info("loading pass2 %s", args.pass2)
    p2 = load_valid(args.pass2)
    log.info("pass2 strict-valid: %d", len(p2))

    shared = sorted(set(p1) & set(p2))
    log.info("shared sample_ids: %d", len(shared))

    n_agree = 0
    taus: list[float] = []
    with args.output.open("w", encoding="utf-8") as fh:
        for sid in shared:
            merged = merge_one(p1[sid], p2[sid])
            fh.write(json.dumps(merged, ensure_ascii=False) + "\n")
            if merged["merge"]["top1_agreement"]:
                n_agree += 1
            taus.append(merged["merge"]["kendall_tau"])

    def _mean(xs: list[float]) -> float:
        """Arithmetic mean; 0.0 for empty list."""
        return sum(xs) / len(xs) if xs else 0.0

    log.info("DONE: %d merged records → %s", len(shared), args.output)
    if shared:
        log.info(
            "top-1 agreement: %d / %d = %.3f",
            n_agree, len(shared), n_agree / len(shared),
        )
        log.info("mean Kendall tau (pass1 vs pass2): %.4f", _mean(taus))
        pos_taus = [t for t in taus if t > 0]
        log.info(
            "tau>0 fraction: %d / %d = %.3f (content-consistent share)",
            len(pos_taus), len(taus), len(pos_taus) / len(taus),
        )


if __name__ == "__main__":
    main()

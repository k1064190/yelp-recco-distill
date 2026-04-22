#!/usr/bin/env python
# ABOUTME: Build counterfactual teacher outputs (ranking shuffled / rationale
# ABOUTME: deranged / persona replaced) for validating the listwise judge.

"""
Generate perturbed teacher outputs for Judge validation.

Motivation
----------
The first listwise-judge pass scored the teacher at Groundedness 4.96 /
Personalization 5.00 on 48/50 samples, with zero variance on
Personalization. That ceiling saturation makes it impossible to tell
whether:
  (a) the teacher really is that good and the judge is well-calibrated, or
  (b) the judge is a 5-stamp machine that cannot discriminate.

This script answers (a) vs (b) with a controlled perturbation set: we
take the N=50 teacher outputs and inject known, targeted quality
defects, then re-run the same judge. If the judge's scores drop on the
perturbed outputs, (a) holds. If the scores stay at 5, (b) holds and
the judge is not a useful evaluator.

Perturbation set
----------------
All three are deterministic (seed = hash(sample_id + kind)) so results
are reproducible and the same sample gets the same synthetic defect
across re-runs.

  P1 ranking_shuffled
      persona + rationales preserved; the ranking list is re-permuted
      to a random non-identity ordering. The persona/rationale semantic
      content still fits the candidates, but the ranking order is no
      longer consistent with the rationale tone. Expected: Personalization
      should drop sharply; Groundedness should be largely unchanged.

  P2 rationale_swapped (derangement)
      persona + ranking preserved; rationale TEXT is reassigned across
      candidates via a derangement (permutation with no fixed points).
      So rationale entry "candidate_index=3" now carries the reason
      text originally written about candidate 7. Both axes should
      collapse: Groundedness because the reason now references fields
      of the wrong candidate (category / rating / name mismatch), and
      Personalization because the persona<->rationale<->ranking triangle
      is scrambled.

  P3 persona_replaced
      rationales + ranking preserved; persona is replaced with a generic
      template that has no tie to the user's history. Groundedness
      should be roughly unchanged (rationales still cite real fields);
      Personalization should drop because the persona no longer
      captures the history pattern.

Output
------
Three JSON files in the per-backend shape that ``judge_listwise.py``
already accepts::

    data/inference_samples/teacher-p1-ranking-shuffled.json
    data/inference_samples/teacher-p2-rationale-swapped.json
    data/inference_samples/teacher-p3-persona-replaced.json

Each file has ``backend`` set to the perturbation tag, and a ``samples``
list with the same flat schema as the generator (``sample_id``,
``positive_business_id``, ``output_text`` = JSON of the perturbed
teacher dict, ``parsed_ranking``, ``recovered_business_ids``,
``json_parse_ok=True``, ``output_tokens=None``).

Example::

    $ python scripts/teacher/perturb_teacher_outputs.py \\
        --source data/inference_samples/teacher.json \\
        --samples data/processed/philly_samples.jsonl \\
        --out-dir data/inference_samples
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import random
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("perturb_teacher")


# ---------- Perturbation constants ----------


GENERIC_PERSONA = (
    "The user enjoys trying various restaurants and places, including "
    "different cuisines and casual dining spots."
)
"""Generic placeholder persona for P3. Deliberately bland and
history-agnostic so the judge's Personalization score should drop if
it is paying any attention to history<->persona alignment."""


P1_TAG = "teacher-p1-ranking-shuffled"
P2_TAG = "teacher-p2-rationale-swapped"
P3_TAG = "teacher-p3-persona-replaced"


# ---------- RNG + permutation helpers ----------


def _rng_for_sample(sample_id: str, kind: str) -> random.Random:
    """Derive a deterministic ``random.Random`` from (sample_id, kind).

    Args:
        sample_id (str): the eval sample id (used as the primary seed).
        kind (str): perturbation kind label (so P1 and P2 on the same
            sample get independent random streams).

    Returns:
        random.Random: seeded RNG. The seed is the first 16 hex digits
            of ``sha256(f"{kind}:{sample_id}")`` interpreted as int.
    """
    h = hashlib.sha256(f"{kind}:{sample_id}".encode("utf-8")).hexdigest()
    return random.Random(int(h[:16], 16))


def _shuffled_non_identity(seq: list[Any], rng: random.Random) -> list[Any]:
    """Return a random permutation of ``seq`` that is not the identity.

    Args:
        seq (list): input list (treated as ordered; not mutated).
        rng (random.Random): seeded RNG.

    Returns:
        list: permutation different from ``seq``. For length <= 1 the
            input is returned unchanged. For length 2 the single
            non-identity permutation (swap) is returned. For larger
            inputs we shuffle with up to 50 attempts; if we never draw
            a non-identity permutation (vanishingly rare) we fall back
            to swapping the first two elements.
    """
    if len(seq) <= 1:
        return list(seq)
    orig = list(seq)
    if len(seq) == 2:
        return [orig[1], orig[0]]
    for _ in range(50):
        perm = orig[:]
        rng.shuffle(perm)
        if perm != orig:
            return perm
    perm = orig[:]
    perm[0], perm[1] = perm[1], perm[0]
    return perm


def _derangement(seq: list[Any], rng: random.Random) -> list[Any]:
    """Return a permutation of ``seq`` with no fixed points.

    A derangement is a permutation where ``perm[i] != seq[i]`` for
    every ``i``. For |seq| >= 2 the probability that a uniform random
    permutation is a derangement is ~1/e ~= 0.37, so rejection
    sampling is efficient.

    Args:
        seq (list): input list. Must have length >= 2 for a non-trivial
            derangement to exist; length 0/1 returns the input.
        rng (random.Random): seeded RNG.

    Returns:
        list: a derangement of ``seq``. On the vanishingly rare case
            that 200 rejection samples all fail to be derangements, we
            fall back to a one-step cyclic shift which is always a
            derangement.
    """
    if len(seq) <= 1:
        return list(seq)
    orig = list(seq)
    for _ in range(200):
        perm = orig[:]
        rng.shuffle(perm)
        if all(a != b for a, b in zip(perm, orig)):
            return perm
    return orig[1:] + orig[:1]


# ---------- Per-output perturbations ----------


def perturb_p1_ranking_shuffled(
    teacher_output: dict[str, Any],
    rng: random.Random,
) -> dict[str, Any]:
    """Apply P1: shuffle the ranking list to a non-identity permutation.

    Args:
        teacher_output (dict): parsed output with keys ``persona``,
            ``rationales``, ``ranking``.
        rng (random.Random): seeded RNG.

    Returns:
        dict: same shape as input; only ``ranking`` is altered.
    """
    new_ranking = _shuffled_non_identity(list(teacher_output.get("ranking") or []), rng)
    return {
        "persona": teacher_output.get("persona", ""),
        "rationales": list(teacher_output.get("rationales") or []),
        "ranking": new_ranking,
    }


def perturb_p2_rationale_swapped(
    teacher_output: dict[str, Any],
    rng: random.Random,
) -> dict[str, Any]:
    """Apply P2: redistribute rationale reason texts via a derangement.

    The rationale entry for each ``candidate_index`` keeps its
    ``candidate_index`` field, but its ``reason`` is replaced with the
    reason text originally attached to a different candidate. The
    derangement guarantees no rationale keeps its original text.

    Args:
        teacher_output (dict): parsed output.
        rng (random.Random): seeded RNG.

    Returns:
        dict: same shape as input; ``rationales[i].reason`` is swapped
            across positions while ``ranking`` and ``persona`` are
            preserved.
    """
    rationales = list(teacher_output.get("rationales") or [])
    if len(rationales) < 2:
        # Degenerate: nothing to swap.
        return {
            "persona": teacher_output.get("persona", ""),
            "rationales": rationales,
            "ranking": list(teacher_output.get("ranking") or []),
        }

    indexed = sorted(
        [(r.get("candidate_index"), r.get("reason", "")) for r in rationales
         if isinstance(r.get("candidate_index"), int)],
        key=lambda p: p[0],
    )
    if not indexed:
        return teacher_output

    reasons = [reason for _, reason in indexed]
    deranged = _derangement(reasons, rng)
    new_rationales = [
        {"candidate_index": idx, "reason": new_reason}
        for (idx, _), new_reason in zip(indexed, deranged)
    ]
    return {
        "persona": teacher_output.get("persona", ""),
        "rationales": new_rationales,
        "ranking": list(teacher_output.get("ranking") or []),
    }


def perturb_p3_persona_replaced(
    teacher_output: dict[str, Any],
    _rng: random.Random,
) -> dict[str, Any]:
    """Apply P3: replace the persona string with a generic template.

    Args:
        teacher_output (dict): parsed output.
        _rng (random.Random): unused; signature matches the other
            perturbation callables.

    Returns:
        dict: same shape with ``persona`` replaced by ``GENERIC_PERSONA``.
    """
    return {
        "persona": GENERIC_PERSONA,
        "rationales": list(teacher_output.get("rationales") or []),
        "ranking": list(teacher_output.get("ranking") or []),
    }


PERTURBATIONS: dict[str, tuple[str, Callable[[dict[str, Any], random.Random], dict[str, Any]]]] = {
    "p1": (P1_TAG, perturb_p1_ranking_shuffled),
    "p2": (P2_TAG, perturb_p2_rationale_swapped),
    "p3": (P3_TAG, perturb_p3_persona_replaced),
}


# ---------- samples.jsonl helper ----------


def _load_candidates_by_sample_id(samples_path: Path) -> dict[str, list[str]]:
    """Index samples.jsonl so we can map candidate_index -> business_id.

    Args:
        samples_path (Path): the preprocessed samples JSONL.

    Returns:
        dict: {sample_id -> ordered list of business_ids}. Position ``i``
            of the list corresponds to candidate_index ``i + 1``.
    """
    out: dict[str, list[str]] = {}
    with samples_path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue
            sid = rec.get("sample_id")
            cands = rec.get("candidates") or []
            if not sid or not isinstance(cands, list):
                continue
            out[sid] = [c.get("business_id") for c in cands]
    return out


def _recovered_bids_from_ranking(
    ranking: list[int],
    candidate_bids: list[str],
) -> list[str | None]:
    """Convert a 1-based ranking list into a list of business_ids.

    Args:
        ranking (list[int]): 1-based candidate indices (best -> worst).
        candidate_bids (list[str]): business_ids ordered by candidate_index.

    Returns:
        list[str | None]: business_ids in ranking order; positions that
            reference an out-of-bounds candidate index get ``None``.
    """
    out: list[str | None] = []
    for idx in ranking:
        if not isinstance(idx, int) or not (1 <= idx <= len(candidate_bids)):
            out.append(None)
            continue
        out.append(candidate_bids[idx - 1])
    return out


# ---------- I/O ----------


def build_perturbed_cache(
    source_cache: dict[str, Any],
    backend_tag: str,
    kind: str,
    perturb_fn: Callable[[dict[str, Any], random.Random], dict[str, Any]],
    candidates_by_sid: dict[str, list[str]],
) -> dict[str, Any]:
    """Build a per-backend cache JSON for one perturbation type.

    Args:
        source_cache (dict): loaded teacher per-backend JSON
            (``{"backend": "teacher", "samples": [...]}``).
        backend_tag (str): tag to embed in the output (e.g.
            ``"teacher-p1-ranking-shuffled"``) and used as the
            ``model_tag`` when the judge reads this file.
        kind (str): short tag used for RNG seeding (P1 / P2 / P3).
        perturb_fn (callable): ``(teacher_output, rng) -> perturbed``.
        candidates_by_sid (dict): map from sample_id to its ordered
            candidate business_ids. Needed to re-derive
            ``recovered_business_ids`` after ranking permutations.

    Returns:
        dict: per-backend JSON ready to write to disk.
    """
    new_samples: list[dict[str, Any]] = []
    skipped_unparseable = 0
    for s in source_cache.get("samples", []):
        sid = s.get("sample_id")
        if not sid:
            continue
        out_text = s.get("output_text") or ""
        try:
            parsed = json.loads(out_text)
        except json.JSONDecodeError:
            skipped_unparseable += 1
            continue
        if not isinstance(parsed, dict):
            skipped_unparseable += 1
            continue

        rng = _rng_for_sample(sid, kind)
        perturbed = perturb_fn(parsed, rng)
        new_text = json.dumps(perturbed, ensure_ascii=False)
        new_ranking = perturbed.get("ranking") or []
        cand_bids = candidates_by_sid.get(sid) or []
        new_bids = _recovered_bids_from_ranking(new_ranking, cand_bids)

        new_samples.append({
            "sample_id": sid,
            "positive_business_id": s.get("positive_business_id"),
            "prompt_preview": s.get("prompt_preview"),
            "output_text": new_text,
            "parsed_ranking": new_ranking,
            "recovered_business_ids": new_bids,
            "json_parse_ok": True,
            "output_tokens": None,
            "latency_sec": None,
            "perturbation_kind": kind,
        })
    if skipped_unparseable:
        log.warning(
            "skipped %d unparseable source records for %s",
            skipped_unparseable, backend_tag,
        )
    return {
        "backend": backend_tag,
        "model_path_or_url": f"synthetic-perturbation:{kind}",
        "dtype": "synthetic",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "source_cache_basename": source_cache.get("_source_basename"),
        "samples": new_samples,
    }


def write_all_perturbations(
    source_cache_path: Path,
    samples_path: Path,
    out_dir: Path,
    kinds: list[str],
) -> dict[str, Path]:
    """Generate and write the requested perturbation JSONs.

    Args:
        source_cache_path (Path): teacher per-backend JSON
            (``data/inference_samples/teacher.json``).
        samples_path (Path): preprocessed samples JSONL (needed for
            candidate_index -> business_id mapping).
        out_dir (Path): output directory.
        kinds (list[str]): subset of ``["p1", "p2", "p3"]``.

    Returns:
        dict: {kind -> output path}.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    source = json.loads(source_cache_path.read_text(encoding="utf-8"))
    source["_source_basename"] = source_cache_path.name
    cand_by_sid = _load_candidates_by_sample_id(samples_path)

    written: dict[str, Path] = {}
    for k in kinds:
        if k not in PERTURBATIONS:
            raise ValueError(f"unknown perturbation kind {k!r}")
        tag, fn = PERTURBATIONS[k]
        doc = build_perturbed_cache(source, tag, k, fn, cand_by_sid)
        out_path = out_dir / f"{tag}.json"
        out_path.write_text(json.dumps(doc, indent=2, ensure_ascii=False))
        log.info(
            "%s -> %s (%d samples)",
            tag, out_path, len(doc["samples"]),
        )
        written[k] = out_path
    return written


# ---------- CLI ----------


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--source",
        type=Path,
        default=PROJECT_ROOT / "data/inference_samples/teacher.json",
        help="source teacher cache JSON (per-backend shape)",
    )
    p.add_argument(
        "--samples",
        type=Path,
        default=PROJECT_ROOT / "data/processed/philly_samples.jsonl",
    )
    p.add_argument(
        "--out-dir",
        type=Path,
        default=PROJECT_ROOT / "data/inference_samples",
    )
    p.add_argument(
        "--kinds",
        type=str,
        default="p1,p2,p3",
        help="comma-separated subset of {p1, p2, p3}",
    )
    return p.parse_args()


def main() -> int:
    args = parse_args()
    kinds = [k.strip() for k in args.kinds.split(",") if k.strip()]
    write_all_perturbations(args.source, args.samples, args.out_dir, kinds)
    return 0


if __name__ == "__main__":
    sys.exit(main())

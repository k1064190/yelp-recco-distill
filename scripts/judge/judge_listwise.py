#!/usr/bin/env python
# ABOUTME: Listwise LLM-as-a-Judge absolute scoring on (Groundedness,
# ABOUTME: Logicality & Personalization) axes for Yelp recommendation outputs.

"""
Listwise LLM-as-a-Judge for the Yelp distillation pipeline.

For each (eval sample, model checkpoint) pair, the judge sees:

  - the user's visit history (ground-truth Yelp data),
  - the 10 candidate places shown to the model (ground-truth Yelp data),
  - the model's full listwise output: persona + 10 rationales + ranking.

The judge returns absolute 1-5 scores on two independent axes:

  GROUNDEDNESS
      Are the rationales factually faithful to the candidate's real Yelp
      fields (name, categories, avg_stars, attributes)? Are there any
      hallucinations (invented menu items, wrong price tier, fake ratings)?

  LOGICALITY & PERSONALIZATION
      Does the persona capture the user's history pattern? Does each
      rationale connect that persona to the candidate's fields? Does the
      ranking order follow from the rationales?

Both scores must be accompanied by an evidence string citing the specific
rationale number and the specific Yelp field (or history item) that drove
the score. The evidence strings are not aggregated; they exist for sample
auditing and for any future bias-probe analysis.

Bias probes (position, verbosity, self-enhancement, rubric order, score ID)
are intentionally **not** run here -- they are deferred for this project.

Model outputs are read from a pre-generated inference cache produced by
``scripts/eval/generate_inference_samples.py`` (the consolidated
``data/inference_samples/all_backends_merged.json``). The cache must contain
every requested (sample_id, model_tag) pair; the judge does not run
inference itself.

The judge backend is Gemini 3 Flash Preview, called directly through the
``google-genai`` Python SDK with a single paid API key read from ``.env``
(``GOOGLE_API_KEY``, falling back to ``GEMINI_API_KEY``). Output is appended
to a JSONL file so a throttled run can be resumed without re-judging
completed pairs.

Example::

    $ python scripts/judge/judge_listwise.py \\
        --inference-cache data/inference_samples/all_backends_merged.json \\
        --models teacher,v2-sft,v2-sft-w4a16 \\
        --n 50 \\
        --raw data/results/judge_listwise_raw.jsonl \\
        --summary data/results/judge_listwise_summary.json
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

from pydantic import BaseModel, Field  # noqa: E402

from configs.teacher_prompt import (  # noqa: E402
    _format_candidate_item,
    _format_history_item,
)
from scripts.train.train_student import load_and_filter, split_examples  # noqa: E402


def pick_eval_samples(
    eval_exs: list[dict[str, Any]],
    n: int,
) -> list[dict[str, Any]]:
    """Return ``n`` deterministically-spaced eval examples.

    Mirrors the selection rule in ``scripts.generate_inference_samples.pick_samples``
    so that the judge consumes exactly the same sample_ids that the
    inference cache was generated for. Kept as a local helper rather
    than imported from that module to stay independent of its I/O paths.

    Args:
        eval_exs (list[dict]): full eval split from ``split_examples``.
        n (int): desired count. If ``n`` exceeds the split size the full
            split is returned.

    Returns:
        list[dict]: ``[eval_exs[0], eval_exs[step], eval_exs[2*step], ...]``
            where ``step = len(eval_exs) // n``. For ``n = len(eval_exs)``
            the full split is returned in original order.
    """
    if n <= 0 or not eval_exs:
        return []
    if len(eval_exs) <= n:
        return eval_exs
    step = len(eval_exs) // n
    return [eval_exs[i * step] for i in range(n)]

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("judge_listwise")


# ---------- Judge schema ----------


SCORE_MIN = 1
SCORE_MAX = 10


RUBRIC_VERSION = "v3"
"""Bump whenever ``JUDGE_SYSTEM_LISTWISE`` or ``JUDGE_USER_TEMPLATE``
changes in a way that alters the scoring distribution. Each verdict
record embeds this tag so the analysis scripts can separate runs by
rubric generation.

History:
    v1 -- original rubric, 2026-04-15. Teacher pass saturated
          (G=4.96, P=5.00). Counterfactual validation showed the rubric
          did not enforce rationale<->candidate field alignment (P2
          groundedness null, Delta=-0.04) or persona<->history coherence
          (P3 personalization null, every Delta=0).
    v2 -- rewrite shipped 2026-04-15 evening. Forces per-rationale
          candidate-field citation and persona-history pattern
          enumeration. Lifted P2 groundedness and P3 personalization
          above null in validation (6/6 probe cells pass), but teacher
          baseline still saturated at G=4.94 / P=5.00 on the 1-5 scale
          because the top tier rewarded "no errors" rather than "rich
          citations". Student discrimination headroom was too narrow.
    v3 -- rewrite shipped 2026-04-22. Expands to 3 independent axes
          (Groundedness, Personalization, Ranking Coherence) on a 1-10
          scale with quality gates at the top tiers (10 requires
          avg_fields >= 3.0 with attribute citations; 10 on
          Personalization requires non-trivial inference from history).
          Ranking Coherence pulls R_reverse out of Personalization so
          ranking-vs-rationale internal consistency is scored
          independently. Teacher baseline observed at G=8.6 / P=9.3 /
          RC=10 on 50 samples -- variance unlocked as intended.

          Calibration iterations (same RUBRIC_VERSION tag "v3",
          distinguished by pre-fix backup files in _legacy/):
            iter0 (2026-04-22 02:28): cap=4 for cross-candidate (c) leak.
              Validation: P1xRC pass Δ=-7.24, P3xP pass Δ=-2.08. BUT
              P2xG inverted (Δ=+0.44) and P2xP null (Δ=-0.12) because
              the cap at 4 was too gentle -- Gemini 3 Flash catches the
              obvious P2 derangements but misses subtle category/rating
              overlaps, so only ~20 % of samples got depressed G, not
              enough to shift the mean. Backup:
              judge_listwise.py.bak-rubric-v3-iter0-cap4-20260422.
            iter1 (2026-04-22, current): cap=2 for (c) leak. Restores
              v2-era cap strictness so even a single caught swap drives
              G to 2, producing a meaningful mean drop on P2.
"""


class ListwiseVerdict(BaseModel):
    """Structured judge output for one (sample, model) listwise judgment.

    Attributes:
        groundedness (int): 1-10 score for factual fidelity to the provided
            Yelp data AND citation depth. 10 requires every rationale to
            cite 3+ distinct fields including at least one named attribute
            on at least half the rationales; 8 = "correct and basic"; 1 =
            most rationales talk about the wrong candidate entirely.
        groundedness_evidence (str): must report G_exact, G_wrong, G_hallu,
            G_vague, avg_fields_per_rationale, G_attr_N counts plus one
            concrete justification line for the chosen score.
        personalization (int): 1-10 score for the persona's specificity to
            this user's history and the rationales' anchoring to that
            persona. 10 requires 2+ non-trivial inferences plus every
            rationale linked; 8 = solid specific persona; 4 or below =
            generic persona that could fit any user.
        personalization_evidence (str): must report P_specific,
            P_nontrivial, R_link counts plus one concrete justification line.
        ranking_coherence (int): 1-10 score for internal consistency
            between the model's rationale tones and its own ranking order.
            Measures the model's coherence, NOT retrieval correctness. 10
            requires zero inversions plus a monotonic tone-ordering match;
            2 means the ranking systematically inverts the rationales.
        ranking_coherence_evidence (str): must report the per-candidate
            rationale tone labels ([+,+,~,-,...] for candidates 1..N),
            R_reverse, R_top3_positive, and R_bottom3_negative counts plus
            one concrete justification line.
    """

    groundedness: int = Field(..., ge=SCORE_MIN, le=SCORE_MAX)
    groundedness_evidence: str = Field(
        ...,
        description=(
            "Report G_exact / G_wrong / G_hallu / G_vague counts, "
            "avg_fields_per_rationale (mean distinct fields cited), and "
            "G_attr_N (rationales citing a named attribute). One concrete "
            "example line explaining the chosen tier."
        ),
    )
    personalization: int = Field(..., ge=SCORE_MIN, le=SCORE_MAX)
    personalization_evidence: str = Field(
        ...,
        description=(
            "Report P_specific count (history-bound observations), "
            "P_nontrivial count (non-obvious inferences), and R_link count "
            "(rationales linked to persona). One concrete example line."
        ),
    )
    ranking_coherence: int = Field(..., ge=SCORE_MIN, le=SCORE_MAX)
    ranking_coherence_evidence: str = Field(
        ...,
        description=(
            "List rationale tone per candidate [+, ~, -] in candidate-index "
            "order, report R_reverse / R_top3_positive / R_bottom3_negative "
            "counts, and give one concrete example line for the chosen tier."
        ),
    )


JUDGE_SYSTEM_LISTWISE = (
    "You are a strict, evidence-driven evaluator of place recommendation "
    "systems. You score the full listwise output of a single model on "
    "THREE independent axes, each on a 1-10 scale:\n"
    "  - Groundedness: rationale #k cites fields that actually belong to "
    "candidate #k, with depth and specificity.\n"
    "  - Personalization: persona identifies specific, non-trivial "
    "patterns in the user's history; rationales anchor to that persona.\n"
    "  - Ranking Coherence: ranking order matches rationale tone -- "
    "positively-described candidates appear ABOVE negatively-described "
    "ones in the ranking. This measures the MODEL'S OWN CONSISTENCY, "
    "not whether the ranking matches ground truth.\n\n"
    "You MUST perform three explicit verification steps before scoring, "
    "described in the user prompt. Use the FULL 1-10 range: 10 is "
    "reserved for outputs that are genuinely exceptional (rich, "
    "multi-field citations; non-trivial persona inferences; perfectly "
    "coherent ranking). 8 is the typical score for 'correct but basic'. "
    "Do NOT default to 7 or 8 to play it safe -- that is noise, not "
    "evaluation.\n\n"
    "Writing fluency and length must NOT substitute for these checks: a "
    "well-written rationale that cites the wrong candidate's fields is "
    "ungrounded, a fluent persona that could fit any user is not "
    "personalised, and a confident rationale ranked against its own tone "
    "is incoherent. Each axis is scored INDEPENDENTLY -- a strong persona "
    "does not rescue an incoherent ranking, and vice versa."
)


JUDGE_USER_TEMPLATE = """\
You are judging a place-recommendation listwise output for ONE user.

[USER VISIT HISTORY] ({num_history} places, chronological)
{history_block}

[CANDIDATE PLACES] ({num_candidates} places -- this is the ground-truth Yelp data the model saw)
{candidates_block}

[MODEL OUTPUT]
Persona:
  {persona}

Rationales (numbered by candidate index):
{rationales_block}

Ranking (best -> worst, 1-based candidate indices):
  {ranking}

=======================================================================
VERIFICATION STEP 1 -- Groundedness (required, do NOT skip)
=======================================================================

For EACH rationale #k (k = 1 .. N), locate candidate #k in the
[CANDIDATE PLACES] block (same 1-based index -- rationale #3 must be
checked against candidate #3, not some other one). Silently answer:

  (a) Does the rationale cite at least one specific field (name,
      categories, avg_stars / rating tier, named attribute, or hours)
      from candidate #k?
  (b) If yes, does the cited field value actually appear in candidate
      #k's data? ("highly rated dessert spot" for avg_stars=2.5 does
      NOT match.)
  (c) Does the rationale mention a name / cuisine / category belonging
      to a DIFFERENT candidate? (Cross-candidate leak: rationale #1 talks
      about "London Grill" but candidate #1 is "Heung Fa Chun Sweet
      House".)
  (d) How many DISTINCT fields are cited in this rationale? Count: name
      (1), categories (1 per distinct category label), rating (1), each
      named attribute separately (e.g., "has patio" + "dog-friendly" =
      2), hours (1). "This Japanese restaurant" = 1 field. "4.5-star
      Japanese restaurant with a patio" = 3 fields.

Compute:
  G_exact    = rationales that pass (a) AND (b) AND do not fail (c).
  G_wrong    = rationales that fail (c) -- cross-candidate leak.
  G_hallu    = rationales that cite a field value not matching candidate #k.
  G_vague    = rationales that cite no specific field at all (pure filler).
  avg_fields = mean of (d) across rationales.
  G_attr_N   = rationales that cite at least one NAMED ATTRIBUTE
               (not just name / category / rating).

GROUNDEDNESS ladder (1-10, pick the HIGHEST tier whose preconditions hold):

 10 = G_exact = N AND G_wrong = 0 AND G_hallu = 0 AND G_vague = 0 AND
      avg_fields >= 3.0 AND G_attr_N >= ceil(N/2).
      "Exceptional: rich, multi-field citations with specific attributes."
  9 = G_exact = N AND 0 errors AND 0 vague AND avg_fields >= 2.0.
      "Very good: every rationale cites 2+ fields correctly."
  8 = G_exact = N AND 0 errors AND 0 vague AND avg_fields >= 1.5.
      "Solid baseline: all rationales correct but mostly single-field."
  7 = G_exact >= N-1 AND G_wrong = 0 AND G_hallu = 0 AND G_vague <= 1.
      "One rationale shallow or vague, rest correct."
  6 = G_exact >= N-2 AND G_wrong = 0 AND G_hallu = 0 AND G_vague <= 2.
      "Two rationales shallow, rest correct."
  5 = G_exact >= ceil(2N/3) AND (G_wrong + G_hallu) <= 1 AND G_vague <= 3.
      "Mostly correct with one factual slip."
  4 = G_exact >= ceil(N/2) AND (G_wrong + G_hallu) <= 2.
      "Half correct, small mis-attribution."
  3 = (G_wrong + G_hallu) = 3 OR G_exact < ceil(N/2) with G_vague >= 4.
      "Substantial grounding failures."
  2 = (G_wrong + G_hallu) >= ceil(N/2).
      "Half or more rationales mis-attributed or hallucinated."
  1 = Most rationales talk about entirely wrong candidates.

CAP: a single clear case of (c) caps Groundedness at 2, because factual
fidelity to THIS candidate is the point of the axis. To reach 9-10 you
need BOTH zero errors AND demonstrable depth.

=======================================================================
VERIFICATION STEP 2 -- Personalization (required, do NOT skip)
=======================================================================

Read the persona text. List (silently) every SPECIFIC, history-bound
observation. A specific observation names or clearly alludes to one of:
  - a cuisine that appears in the user's history,
  - a venue or chain mentioned in the history,
  - a preference pattern the history supports (e.g. "prefers casual"
    when most history entries are casual),
  - a review sentiment pattern (e.g. "dislikes crowded places",
    "values whiskey selection" drawn from review wording).
Generic statements like "enjoys various restaurants" count as 0.

Compute:
  P_specific   = count of specific, history-bound observations.
  P_nontrivial = count of observations that require INFERENCE, not just
                 surface frequency. Trivial: "likes Asian food" when
                 history has 8/10 Asian entries (near-tautology).
                 Non-trivial: "prefers neighborhood gastropubs over
                 chain bars despite living in a chain-heavy area"
                 (inferred from review sentiment + venue style, not
                 a direct count).
  R_link       = rationales whose reason explicitly connects the
                 candidate to at least one specific persona observation
                 (not to a generic claim).

PERSONALIZATION ladder (1-10):

 10 = P_specific >= 5 AND R_link = N AND P_nontrivial >= 2.
      "Exceptional: persona demonstrates real user-model insight."
  9 = P_specific >= 4 AND R_link >= N-1 AND P_nontrivial >= 1.
      "Very good: specific persona with at least one inference."
  8 = P_specific >= 3 AND R_link >= N-1.
      "Solid: specific persona, rationales anchor to it."
  7 = P_specific >= 2 AND R_link >= ceil(2N/3).
      "Persona moderately specific, two-thirds of rationales linked."
  6 = P_specific >= 2 AND R_link >= ceil(N/2).
      "Persona moderately specific, half of rationales linked."
  5 = P_specific = 1, OR (P_specific >= 2 AND R_link < ceil(N/2)).
      "One specific observation, or persona-rationale disconnect."
  4 = P_specific = 0 AND R_link >= ceil(N/3).
      "Generic persona; some rationales still tie to history features."
  3 = P_specific = 0 AND R_link < ceil(N/3).
      "Generic persona, rationales don't anchor."
  2 = Persona mentions NO history pattern at all.
  1 = Persona contradicts the history (claims preferences it cannot support).

CAP: P_specific = 0 caps Personalization at 4. Generic filler that
could apply to any user is not personalisation regardless of rationale
craftsmanship.

=======================================================================
VERIFICATION STEP 3 -- Ranking Coherence (required, do NOT skip)
=======================================================================

This axis measures whether the MODEL'S OWN RANKING is consistent with
the MODEL'S OWN RATIONALES. It does NOT reward correct retrieval (that
is measured separately by R@1 / MRR). It rewards internal consistency.

For EACH rationale, silently label its tone:
  +  positive / strong endorsement ("excellent match", "clear fit",
     "user would love this").
  ~  neutral ("acceptable option", "not outstanding but reasonable").
  -  negative / weak fit ("poor match", "user rarely visits this
     category", "3.0 stars is low for this cuisine").

Then read the RANKING (best -> worst, 1-based candidate indices). Compute:

  R_reverse          = number of ordered pairs (candidate_i above
                       candidate_j in the ranking) where tone(i) = '-'
                       AND tone(j) = '+'. I.e. a negatively-toned
                       candidate ranked above a positively-toned one.
  R_top3_positive    = among the TOP 3 ranked candidates, how many have
                       tone = '+'.
  R_bottom3_negative = among the BOTTOM 3 ranked candidates, how many
                       have tone = '-'.

RANKING COHERENCE ladder (1-10):

 10 = R_reverse = 0 AND R_top3_positive = 3 AND R_bottom3_negative >= 2.
      "Perfect monotonic agreement between tone and rank."
  9 = R_reverse = 0 AND R_top3_positive >= 2 AND R_bottom3_negative >= 2.
      "Ranking strongly follows rationale tone."
  8 = R_reverse <= 1 AND R_top3_positive >= 2.
      "One minor inversion; top of ranking still tracks positive tone."
  7 = R_reverse <= 2.
      "Couple of inversions but no systemic disagreement."
  6 = R_reverse <= 3, OR R_top3_positive = 1 with R_reverse <= 4.
      "Several inversions."
  5 = R_reverse <= ceil(N/2).
      "Ranking partially decoupled from tone."
  4 = R_reverse > ceil(N/2) but not systematic reversal.
      "Ranking often inverts rationale tone."
  3 = Ranking largely ignores rationale tone.
  2 = Ranking systematically inverted: positively-toned candidates
      cluster at the BOTTOM of the ranking.
  1 = Ranking appears random relative to rationale tone.

EDGE CASE: if ALL rationales share the same tone (e.g. all '+') then
R_reverse is mechanically 0. In that case, score based on whether the
ranking still reflects the RELATIVE STRENGTH of endorsements (stronger
phrasings ranked higher). If the model's ranking is random over equally
positive rationales, score 5-6, not 10.

=======================================================================
OUTPUT FORMAT
=======================================================================

Return JSON matching the provided schema. Score all three axes on 1-10.
Each evidence string MUST report the counts you computed PLUS one
concrete example line explaining the chosen tier.

  groundedness_evidence example:
    "G_exact=10, G_wrong=0, G_hallu=0, G_vague=0, avg_fields=1.8,
     G_attr_N=3. All rationales correctly cite fields but most use
     name+category only; only rationales #2, #5, #9 cite attributes
     (patio, dog-friendly, wifi). Score: 8 -- correct and basic, missing
     the attribute depth needed for 9-10."

  personalization_evidence example:
    "P_specific=4 (Indian/Jamaican/African cuisines from history #2,4,5;
     sandwich preference from #7,9; hygiene sensitivity from #3,7;
     nightlife from #1,6,8). P_nontrivial=1 (hygiene sensitivity is
     inferred from review sentiment, not a surface count). R_link=10/10.
     Score: 9."

  ranking_coherence_evidence example:
    "Tones by candidate index 1..10: [+,~,+,+,~,~,-,~,-,+].
     Ranking: [10,3,4,1,5,6,2,8,9,7].
     R_reverse=1 (candidate 7, tone '-', ranked at position 10 which is
     fine; but candidate 9 tone '-' ranked above 7 tone '-' -- no
     inversion). Re-checking: no '-' ranked above '+'.
     R_top3_positive=3 (candidates 10,3,4 all '+').
     R_bottom3_negative=2 (candidates 9 and 7 are '-', candidate 8 is
     '~'). Score: 10 -- perfect."

REMINDERS:
- Use the FULL 1-10 range. 10 is rare -- reserved for truly exceptional
  outputs. 8 is "correct and basic". Do not cluster at 7-8.
- Fluency, confidence, and length do NOT substitute for the verification
  counts above.
- Each axis is scored INDEPENDENTLY. A strong Groundedness score does
  not rescue a Personalization or Ranking Coherence failure.

Do not output anything other than the JSON object.\
"""


# ---------- Prompt rendering ----------


def _format_rationale_line(idx: int, reason: str) -> str:
    """Render one rationale line keyed by candidate index.

    Args:
        idx (int): 1-based candidate index.
        reason (str): the rationale text.

    Returns:
        str: ``"  N. <reason>"`` with leading two-space indent so the
            block stays visually aligned under the [MODEL OUTPUT] header.
    """
    text = (reason or "").strip().replace("\n", " ")
    return f"  {idx}. {text}"


def _build_rationales_block(rationales: list[dict[str, Any]] | None) -> str:
    """Format the listwise rationales block in candidate-index order.

    Args:
        rationales (list[dict] | None): each entry has
            ``candidate_index`` (1-based int) and ``reason`` (str). Order
            of the input list is normalized to ascending candidate_index.

    Returns:
        str: ``"  1. ...\\n  2. ...\\n ..."`` with one line per rationale,
            or ``"  (none)"`` if the list is empty / missing.
    """
    if not rationales:
        return "  (none)"
    pairs: list[tuple[int, str]] = []
    for r in rationales:
        idx = r.get("candidate_index")
        if not isinstance(idx, int):
            continue
        pairs.append((idx, r.get("reason", "")))
    pairs.sort(key=lambda p: p[0])
    return "\n".join(_format_rationale_line(i, reason) for i, reason in pairs)


def _build_ranking_block(ranking: list[Any] | None) -> str:
    """Render the ranking list as a comma-separated single line.

    Args:
        ranking (list | None): ordered candidate indices (best -> worst).

    Returns:
        str: ``"1, 4, 7, ..."`` or ``"(none)"``.
    """
    if not ranking:
        return "(none)"
    return ", ".join(str(x) for x in ranking)


def build_history_block(sample: dict[str, Any]) -> str:
    """Render the [USER VISIT HISTORY] block matching teacher_prompt format.

    Args:
        sample (dict): preprocessed sample with ``history`` list of dicts.

    Returns:
        str: numbered bullet lines, one per history entry, or ``"(none)"``.
    """
    history = sample.get("history", []) or []
    if not history:
        return "(none)"
    return "\n".join(_format_history_item(i + 1, h) for i, h in enumerate(history))


def build_candidates_block(sample: dict[str, Any]) -> str:
    """Render the [CANDIDATE PLACES] block matching teacher_prompt format.

    Args:
        sample (dict): preprocessed sample with ``candidates`` list of dicts.

    Returns:
        str: numbered bullet lines, one per candidate, or ``"(none)"``.
    """
    candidates = sample.get("candidates", []) or []
    if not candidates:
        return "(none)"
    return "\n".join(_format_candidate_item(i + 1, c) for i, c in enumerate(candidates))


def build_judge_prompt_listwise(
    sample: dict[str, Any],
    model_output: dict[str, Any],
) -> str:
    """Render the listwise judge user-prompt for one (sample, model) pair.

    Args:
        sample (dict): preprocessed sample with ``history`` and
            ``candidates`` (the ground-truth Yelp data the judge anchors on).
        model_output (dict): parsed model output with keys ``persona``,
            ``rationales`` (list of {candidate_index, reason}), ``ranking``
            (list of 1-based ints).

    Returns:
        str: fully-formatted judge user prompt ready to send to Gemini.
    """
    history = sample.get("history", []) or []
    candidates = sample.get("candidates", []) or []
    persona = (model_output.get("persona") or "").strip() or "(empty)"

    return JUDGE_USER_TEMPLATE.format(
        num_history=len(history),
        history_block=build_history_block(sample),
        num_candidates=len(candidates),
        candidates_block=build_candidates_block(sample),
        persona=persona,
        rationales_block=_build_rationales_block(model_output.get("rationales")),
        ranking=_build_ranking_block(model_output.get("ranking")),
    )


# ---------- Inference cache I/O ----------


def _merge_per_backend_into(
    indexed: dict[str, dict[str, Any]],
    backend_tag: str,
    flat_samples: list[dict[str, Any]],
) -> None:
    """Merge a per-backend JSON (flat samples) into the indexed structure.

    Args:
        indexed (dict): {sample_id -> {"positive_business_id", "by_backend": {...}}}
            accumulator. Mutated in place.
        backend_tag (str): backend key to insert under ``by_backend``.
        flat_samples (list[dict]): sample records from a per-backend
            JSON, each with top-level keys like ``output_text``,
            ``parsed_ranking``, ``recovered_business_ids``.
    """
    for rec in flat_samples:
        sid = rec.get("sample_id")
        if not sid:
            continue
        entry = indexed.setdefault(
            sid,
            {
                "sample_id": sid,
                "positive_business_id": rec.get("positive_business_id"),
                "by_backend": {},
            },
        )
        entry["by_backend"][backend_tag] = {
            k: v for k, v in rec.items() if k not in {"sample_id", "positive_business_id"}
        }
        # Prefer non-None positive_business_id if the first record lacked it.
        if entry.get("positive_business_id") is None:
            entry["positive_business_id"] = rec.get("positive_business_id")


def _load_one_cache(cache_path: Path) -> dict[str, dict[str, Any]]:
    """Load a single inference-cache JSON in either consolidated or
    per-backend shape and return it indexed by sample_id.

    Args:
        cache_path (Path): JSON produced by
            ``scripts/eval/generate_inference_samples.py``. Two shapes are
            accepted:
              - consolidated: top-level dict with ``samples: [{sample_id,
                positive_business_id, by_backend: {tag: {...}}}]``
                (``all_backends_merged.json``).
              - per-backend: top-level dict with ``backend: <tag>`` and
                ``samples: [{sample_id, positive_business_id, output_text,
                ...}]`` (``teacher.json``, ``v2-sft.json``, ...).

    Returns:
        dict: {sample_id -> entry} in the indexed shape.
    """
    raw = json.loads(cache_path.read_text(encoding="utf-8"))
    samples = raw.get("samples")
    if not isinstance(samples, list):
        raise ValueError(
            f"inference cache {cache_path} has no 'samples' list (keys={list(raw)})"
        )

    indexed: dict[str, dict[str, Any]] = {}
    backend_tag = raw.get("backend")
    if backend_tag and samples and "by_backend" not in (samples[0] or {}):
        # Per-backend shape: rehydrate into the indexed-by-sample_id layout.
        _merge_per_backend_into(indexed, backend_tag, samples)
    else:
        # Consolidated shape: already indexed-ish, just keyed by sample_id.
        for entry in samples:
            sid = entry.get("sample_id")
            if not sid:
                continue
            indexed[sid] = entry
    return indexed


def load_inference_cache(
    cache_paths: Path | list[Path],
) -> dict[str, dict[str, Any]]:
    """Load one or more inference-cache JSONs and return a merged index.

    Accepts either a single ``Path`` (the common case -- consolidated
    ``all_backends_merged.json``) or a list of paths (per-backend files
    to stitch together). Per-sample ``by_backend`` maps from later files
    override earlier ones for overlapping backend tags.

    Args:
        cache_paths (Path | list[Path]): one or more inference caches.

    Returns:
        dict: {sample_id -> {"positive_business_id", "by_backend":
            {tag: {...}}}}. The keys of ``by_backend`` are the union of
            tags across the input files.
    """
    if isinstance(cache_paths, Path):
        return _load_one_cache(cache_paths)

    merged: dict[str, dict[str, Any]] = {}
    for p in cache_paths:
        one = _load_one_cache(p)
        for sid, entry in one.items():
            target = merged.setdefault(
                sid,
                {
                    "sample_id": sid,
                    "positive_business_id": entry.get("positive_business_id"),
                    "by_backend": {},
                },
            )
            if target.get("positive_business_id") is None:
                target["positive_business_id"] = entry.get("positive_business_id")
            for tag, bk in (entry.get("by_backend") or {}).items():
                target["by_backend"][tag] = bk
    return merged


def parse_model_output(output_text: str) -> dict[str, Any] | None:
    """Best-effort parse of an inference cache ``output_text`` into the
    listwise output schema.

    Args:
        output_text (str): the model's raw response. May start with a
            ``"```json"`` fence; we strip that defensively.

    Returns:
        dict | None: ``{"persona", "rationales", "ranking"}`` on success
            or ``None`` on JSON parse failure. The function does not
            re-validate the inner schema (that is the judge's job -- the
            judge will see whatever fields are present).
    """
    text = (output_text or "").strip()
    if not text:
        return None
    if text.startswith("```"):
        text = text.strip("`")
        if text.lower().startswith("json"):
            text = text[4:].lstrip()
    try:
        obj = json.loads(text)
    except json.JSONDecodeError:
        return None
    if not isinstance(obj, dict):
        return None
    return obj


# ---------- Resume support ----------


def load_done_keys(raw_path: Path) -> set[tuple[str, str]]:
    """Load (sample_id, model_tag) pairs that already have a verdict on disk.

    Only verdicts whose ``error`` is None count as done -- errored records
    are re-attempted on resume so a transient quota failure does not
    permanently lose the slot.

    Args:
        raw_path (Path): append-only JSONL of verdict records.

    Returns:
        set: {(sample_id, model_tag), ...}.
    """
    if not raw_path.exists():
        return set()
    done: set[tuple[str, str]] = set()
    with raw_path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue
            if rec.get("error") is not None:
                continue
            sid = rec.get("sample_id")
            tag = rec.get("model_tag")
            if sid and tag:
                done.add((sid, tag))
    return done


# ---------- Judge call via google-genai ----------


class ListwiseJudgeClient:
    """Direct google-genai client for listwise judge calls.

    Reads ``GOOGLE_API_KEY`` (or ``GEMINI_API_KEY`` as fallback) from the
    project ``.env`` and calls ``client.models.generate_content`` with
    structured-output config (Pydantic ``ListwiseVerdict`` as
    ``response_schema``). Transient 429/503 errors are retried with
    exponential backoff.

    Args:
        env_file (Path): path to the project ``.env`` from which the API
            key is loaded.
        model_name (str): Gemini model identifier (e.g.
            ``gemini-3-flash-preview``).
        api_call_interval (float): minimum seconds to sleep between
            successive calls. Set to 0 on paid tier with no TPM pressure.
        max_output_tokens (int): judge generation cap.
        max_retries (int): number of transient-error retries before
            surfacing the error in the verdict record.
    """

    _RETRY_STATUS = (429, 500, 502, 503, 504)

    def __init__(
        self,
        env_file: Path,
        model_name: str,
        api_call_interval: float,
        max_output_tokens: int = 1024,
        max_retries: int = 3,
    ):
        import os

        from dotenv import load_dotenv

        if env_file.exists():
            load_dotenv(env_file)

        api_key = os.environ.get("GOOGLE_API_KEY") or os.environ.get("GEMINI_API_KEY")
        if not api_key:
            raise RuntimeError(
                f"no GOOGLE_API_KEY (or GEMINI_API_KEY) in {env_file}; cannot run judge"
            )

        from google import genai
        from google.genai import types as genai_types

        self._genai_types = genai_types
        self._client = genai.Client(api_key=api_key)
        self._model_name = model_name
        self._api_call_interval = api_call_interval
        self._max_retries = max_retries
        self._last_call_ts: float = 0.0

        self._gen_config = genai_types.GenerateContentConfig(
            temperature=0.0,
            response_mime_type="application/json",
            response_schema=ListwiseVerdict,
            system_instruction=JUDGE_SYSTEM_LISTWISE,
            thinking_config=genai_types.ThinkingConfig(
                thinking_level=genai_types.ThinkingLevel.MINIMAL,
            ),
            max_output_tokens=max_output_tokens,
        )

    def _throttle(self) -> None:
        """Sleep so that calls are spaced at least ``api_call_interval`` apart."""
        import time

        if self._api_call_interval <= 0:
            return
        elapsed = time.monotonic() - self._last_call_ts
        wait = self._api_call_interval - elapsed
        if wait > 0:
            time.sleep(wait)
        self._last_call_ts = time.monotonic()

    def _generate_with_retry(self, prompt_text: str) -> tuple[str | None, str | None]:
        """Call generate_content with exponential backoff on transient errors.

        Returns:
            tuple: (text, error). Exactly one is non-None.
        """
        import time

        last_err: str | None = None
        for attempt in range(self._max_retries + 1):
            self._throttle()
            try:
                response = self._client.models.generate_content(
                    model=self._model_name,
                    contents=prompt_text,
                    config=self._gen_config,
                )
            except Exception as e:  # google.genai.errors.APIError and friends
                status = getattr(e, "code", None) or getattr(e, "status_code", None)
                last_err = f"{type(e).__name__}: {e}"
                if status in self._RETRY_STATUS and attempt < self._max_retries:
                    backoff = 2.0 ** attempt
                    time.sleep(backoff)
                    continue
                return None, last_err

            text = getattr(response, "text", None) or ""
            if not text:
                # Empty completion — treat as non-retryable "empty response".
                return None, "empty response"
            return text, None

        return None, last_err or "unknown error"

    def judge_one(
        self,
        sample_id: str,
        model_tag: str,
        sample: dict[str, Any],
        model_output: dict[str, Any],
    ) -> dict[str, Any]:
        """Run one listwise judge call and return a parsed verdict record.

        Args:
            sample_id (str): the eval sample id being judged.
            model_tag (str): which model produced ``model_output`` (used
                only for record-keeping; the judge prompt itself does not
                reveal the model identity to avoid trivial preference
                leakage).
            sample (dict): preprocessed sample (history + candidates).
            model_output (dict): parsed listwise output to score.

        Returns:
            dict: verdict record with keys ``sample_id``, ``model_tag``,
                ``rubric_version``, ``groundedness``, ``personalization``,
                ``groundedness_evidence``, ``personalization_evidence``,
                ``error``. Score fields are None on error.
        """
        prompt_text = build_judge_prompt_listwise(sample, model_output)

        record: dict[str, Any] = {
            "sample_id": sample_id,
            "model_tag": model_tag,
            "rubric_version": RUBRIC_VERSION,
            "groundedness": None,
            "personalization": None,
            "ranking_coherence": None,
            "groundedness_evidence": None,
            "personalization_evidence": None,
            "ranking_coherence_evidence": None,
            "error": None,
        }

        text, err = self._generate_with_retry(prompt_text)
        if err is not None:
            record["error"] = err
            return record

        text = (text or "").strip()
        if text.startswith("```"):
            text = text.strip("`")
            if text.lower().startswith("json"):
                text = text[4:].lstrip()

        try:
            parsed = json.loads(text)
        except json.JSONDecodeError as e:
            record["error"] = f"json decode failed: {e}"
            return record

        try:
            verdict = ListwiseVerdict.model_validate(parsed)
        except Exception as e:  # pydantic.ValidationError
            record["error"] = f"schema validation failed: {e}"
            return record

        record.update(
            {
                "groundedness": verdict.groundedness,
                "personalization": verdict.personalization,
                "ranking_coherence": verdict.ranking_coherence,
                "groundedness_evidence": verdict.groundedness_evidence,
                "personalization_evidence": verdict.personalization_evidence,
                "ranking_coherence_evidence": verdict.ranking_coherence_evidence,
            }
        )
        return record


# ---------- Aggregation ----------


def bootstrap_mean_ci(
    values: list[float],
    n_resamples: int = 10000,
    alpha: float = 0.05,
    seed: int = 0,
) -> tuple[float, float, float]:
    """Bootstrap percentile CI for the mean of a 1-D numeric sample.

    Args:
        values (list[float]): observations. NaNs and Nones must be
            filtered by the caller before reaching here.
        n_resamples (int): bootstrap iteration count (default 10000).
        alpha (float): significance level (default 0.05 -> 95% CI).
        seed (int): RNG seed for reproducibility.

    Returns:
        tuple: (mean, ci_low, ci_high). Returns (NaN, NaN, NaN) if the
            input is empty.
    """
    import numpy as np

    arr = np.asarray(values, dtype=float)
    if arr.size == 0:
        return float("nan"), float("nan"), float("nan")
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, arr.size, size=(n_resamples, arr.size))
    means = arr[idx].mean(axis=1)
    lo = float(np.quantile(means, alpha / 2))
    hi = float(np.quantile(means, 1 - alpha / 2))
    return float(arr.mean()), lo, hi


def aggregate_per_model(
    verdicts: list[dict[str, Any]],
) -> dict[str, dict[str, Any]]:
    """Aggregate raw verdict records into a per-model summary.

    Args:
        verdicts (list[dict]): raw verdict records as written to the
            JSONL file. Records with ``error != None`` are excluded from
            score statistics but still counted in ``n_errors``.

    Returns:
        dict: {model_tag -> {n_total, n_scored, n_errors,
            groundedness: {mean, ci95_low, ci95_high, n},
            personalization: {mean, ci95_low, ci95_high, n},
            ranking_coherence: {mean, ci95_low, ci95_high, n}}}.
            ``ranking_coherence`` is always present; for v1/v2 records it
            will have ``n=0`` / NaN stats because those rubrics did not
            emit this axis.
    """
    by_model: dict[str, list[dict[str, Any]]] = {}
    for v in verdicts:
        tag = v.get("model_tag") or "unknown"
        by_model.setdefault(tag, []).append(v)

    summary: dict[str, dict[str, Any]] = {}
    for tag, recs in by_model.items():
        n_total = len(recs)
        scored = [r for r in recs if r.get("error") is None]
        n_errors = n_total - len(scored)

        g_vals = [
            float(r["groundedness"])
            for r in scored
            if r.get("groundedness") is not None
        ]
        p_vals = [
            float(r["personalization"])
            for r in scored
            if r.get("personalization") is not None
        ]
        rc_vals = [
            float(r["ranking_coherence"])
            for r in scored
            if r.get("ranking_coherence") is not None
        ]
        g_mean, g_lo, g_hi = bootstrap_mean_ci(g_vals)
        p_mean, p_lo, p_hi = bootstrap_mean_ci(p_vals)
        rc_mean, rc_lo, rc_hi = bootstrap_mean_ci(rc_vals)

        summary[tag] = {
            "n_total": n_total,
            "n_scored": len(scored),
            "n_errors": n_errors,
            "groundedness": {
                "mean": g_mean,
                "ci95_low": g_lo,
                "ci95_high": g_hi,
                "n": len(g_vals),
            },
            "personalization": {
                "mean": p_mean,
                "ci95_low": p_lo,
                "ci95_high": p_hi,
                "n": len(p_vals),
            },
            "ranking_coherence": {
                "mean": rc_mean,
                "ci95_low": rc_lo,
                "ci95_high": rc_hi,
                "n": len(rc_vals),
            },
        }
    return summary


# ---------- Main ----------


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--samples",
        type=Path,
        default=PROJECT_ROOT / "data/processed/philly_samples.jsonl",
    )
    p.add_argument(
        "--teacher",
        type=Path,
        default=PROJECT_ROOT / "data/teacher/philly_teacher_qwen35.jsonl",
        help="teacher JSONL used by load_and_filter to define the eval split",
    )
    p.add_argument(
        "--inference-cache",
        type=Path,
        action="append",
        default=None,
        help=(
            "inference cache JSON from generate_inference_samples.py. "
            "May be passed multiple times to stitch per-backend files "
            "(data/inference_samples/teacher.json, v2-sft.json, ...); "
            "the consolidated all_backends_merged.json is also accepted. "
            "Defaults to all_backends_merged.json when omitted."
        ),
    )
    p.add_argument(
        "--models",
        type=str,
        default="teacher,v2-sft,v2-sft-w4a16",
        help="comma-separated model_tag values to score (must exist in cache.by_backend)",
    )
    p.add_argument("--env-file", type=Path, default=PROJECT_ROOT / ".env")
    p.add_argument("--judge-model", type=str, default="gemini-3-flash-preview")
    p.add_argument("--api-call-interval", type=float, default=4.0)
    p.add_argument(
        "--max-output-tokens",
        type=int,
        default=3072,
        help=(
            "judge generation cap. Default 3072 for rubric v3 (up from "
            "2048 on v2) because v3 adds a third axis (Ranking Coherence) "
            "with its own evidence string that enumerates per-candidate "
            "tone labels on top of the v2 counts."
        ),
    )
    p.add_argument("--eval-ratio", type=float, default=0.9)
    p.add_argument("--n", type=int, default=50)
    p.add_argument(
        "--raw",
        type=Path,
        default=None,
        help=(
            "append-only JSONL for raw verdict records. Defaults to "
            "data/results/judge_listwise_raw_{RUBRIC_VERSION}.jsonl so "
            "rubric versions do not mix in one file."
        ),
    )
    p.add_argument(
        "--summary",
        type=Path,
        default=None,
        help=(
            "per-model aggregate summary JSON. Defaults to "
            "data/results/judge_listwise_summary_{RUBRIC_VERSION}.json."
        ),
    )
    p.add_argument(
        "--dry-run",
        action="store_true",
        help="resolve eval split + cache lookups + prompt rendering, then exit "
        "without calling Gemini. Useful to verify N coverage and prompt size.",
    )
    return p.parse_args()


def main() -> int:
    args = parse_args()
    if args.raw is None:
        args.raw = PROJECT_ROOT / f"data/results/judge_listwise_raw_{RUBRIC_VERSION}.jsonl"
    if args.summary is None:
        args.summary = PROJECT_ROOT / f"data/results/judge_listwise_summary_{RUBRIC_VERSION}.json"
    args.raw.parent.mkdir(parents=True, exist_ok=True)
    args.summary.parent.mkdir(parents=True, exist_ok=True)
    log.info("rubric version: %s  raw: %s", RUBRIC_VERSION, args.raw)

    requested_models = [m.strip() for m in args.models.split(",") if m.strip()]
    if not requested_models:
        log.error("no models requested; pass --models")
        return 2

    cache_paths: list[Path] = args.inference_cache or [
        PROJECT_ROOT / "data/inference_samples/all_backends_merged.json"
    ]

    # 1. Eval split (uses the same hash bucketing as train_student.py).
    examples, _stats = load_and_filter(args.samples, args.teacher)
    _train, eval_exs = split_examples(examples, ratio=args.eval_ratio)
    log.info("eval pool: %d records (target N=%d)", len(eval_exs), args.n)
    if not eval_exs:
        log.error("eval split is empty; aborting")
        return 2
    eval_exs = pick_eval_samples(eval_exs, args.n)
    eval_by_id = {ex["sample_id"]: ex for ex in eval_exs}

    # 2. Inference cache (one or more files).
    cache = load_inference_cache(cache_paths)
    log.info(
        "inference cache covers %d sample_ids from %d file(s)",
        len(cache), len(cache_paths),
    )

    missing_in_cache = [sid for sid in eval_by_id if sid not in cache]
    if missing_in_cache:
        log.error(
            "%d/%d eval samples are absent from the inference cache "
            "(first 5 missing: %s). Re-run scripts/eval/generate_inference_samples.py "
            "with at least --n-samples %d before judging.",
            len(missing_in_cache), len(eval_by_id), missing_in_cache[:5], args.n,
        )
        return 3

    # Verify each requested model_tag exists for at least one cached sample.
    available_tags: set[str] = set()
    for entry in cache.values():
        for tag in (entry.get("by_backend") or {}):
            available_tags.add(tag)
    unknown = [m for m in requested_models if m not in available_tags]
    if unknown:
        log.error(
            "requested models %s not present in cache (available: %s)",
            unknown, sorted(available_tags),
        )
        return 3

    # 3. Resume support: skip pairs already scored cleanly.
    done = load_done_keys(args.raw)
    if done:
        log.info("resume: %d (sample_id, model_tag) pairs already scored", len(done))

    # 4. Optional dry-run: resolve everything but skip Gemini.
    if args.dry_run:
        n_pairs = 0
        n_parse_fail = 0
        max_prompt_chars = 0
        for ex in eval_exs:
            sid = ex["sample_id"]
            entry = cache[sid]
            for tag in requested_models:
                bk = (entry.get("by_backend") or {}).get(tag)
                if bk is None:
                    continue
                parsed = parse_model_output(bk.get("output_text", ""))
                if parsed is None:
                    n_parse_fail += 1
                    continue
                prompt = build_judge_prompt_listwise(ex["sample"], parsed)
                max_prompt_chars = max(max_prompt_chars, len(prompt))
                n_pairs += 1
        log.info(
            "dry-run: %d judgeable pairs, %d cached outputs failed JSON parse, "
            "max prompt = %d chars",
            n_pairs, n_parse_fail, max_prompt_chars,
        )
        return 0

    # 5. Judge client.
    judge = ListwiseJudgeClient(
        env_file=args.env_file,
        model_name=args.judge_model,
        api_call_interval=args.api_call_interval,
        max_output_tokens=args.max_output_tokens,
    )

    # 6. Per-pair judging loop with append-on-each-call resumability.
    n_pairs_total = len(eval_exs) * len(requested_models)
    n_done_now = 0
    n_skipped_resume = 0
    n_parse_fail = 0
    n_judge_err = 0
    n_scored = 0

    with args.raw.open("a", encoding="utf-8") as f_raw:
        for ex in eval_exs:
            sid = ex["sample_id"]
            sample = ex["sample"]
            entry = cache[sid]

            for tag in requested_models:
                if (sid, tag) in done:
                    n_skipped_resume += 1
                    continue
                bk = (entry.get("by_backend") or {}).get(tag)
                if bk is None:
                    rec = {
                        "sample_id": sid,
                        "model_tag": tag,
                        "groundedness": None,
                        "personalization": None,
                        "groundedness_evidence": None,
                        "personalization_evidence": None,
                        "error": "missing in cache",
                    }
                    f_raw.write(json.dumps(rec, ensure_ascii=False) + "\n")
                    f_raw.flush()
                    n_judge_err += 1
                    continue

                parsed = parse_model_output(bk.get("output_text", ""))
                if parsed is None:
                    rec = {
                        "sample_id": sid,
                        "model_tag": tag,
                        "groundedness": None,
                        "personalization": None,
                        "groundedness_evidence": None,
                        "personalization_evidence": None,
                        "error": "model output parse failed",
                    }
                    f_raw.write(json.dumps(rec, ensure_ascii=False) + "\n")
                    f_raw.flush()
                    n_parse_fail += 1
                    continue

                rec = judge.judge_one(
                    sample_id=sid,
                    model_tag=tag,
                    sample=sample,
                    model_output=parsed,
                )
                f_raw.write(json.dumps(rec, ensure_ascii=False) + "\n")
                f_raw.flush()
                n_done_now += 1
                if rec["error"] is None:
                    n_scored += 1
                else:
                    n_judge_err += 1
                if n_done_now % 10 == 0:
                    log.info(
                        "progress: %d new (%d scored, %d judge_err) of %d total pairs",
                        n_done_now, n_scored, n_judge_err, n_pairs_total,
                    )

    log.info(
        "wrote %d new verdicts -> %s "
        "(resume_skipped=%d, parse_fail=%d, judge_err=%d, scored=%d)",
        n_done_now, args.raw, n_skipped_resume, n_parse_fail, n_judge_err, n_scored,
    )

    # 7. Aggregate from full raw file (so resumed runs reflect the union).
    all_verdicts: list[dict[str, Any]] = []
    with args.raw.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                all_verdicts.append(json.loads(line))
            except json.JSONDecodeError:
                continue

    summary = {
        "config": {
            "n_eval_samples": len(eval_exs),
            "models": requested_models,
            "judge_model": args.judge_model,
            "inference_cache": [str(p) for p in cache_paths],
        },
        "per_model": aggregate_per_model(all_verdicts),
    }
    args.summary.write_text(json.dumps(summary, indent=2, ensure_ascii=False))
    log.info("wrote summary -> %s", args.summary)

    for tag, s in summary["per_model"].items():
        g = s["groundedness"]
        p = s["personalization"]
        log.info(
            "%-20s groundedness=%.2f [%.2f, %.2f]  personalization=%.2f [%.2f, %.2f]  n_scored=%d/%d",
            tag,
            g["mean"], g["ci95_low"], g["ci95_high"],
            p["mean"], p["ci95_low"], p["ci95_high"],
            s["n_scored"], s["n_total"],
        )
    return 0


if __name__ == "__main__":
    sys.exit(main())

# ABOUTME: Teacher prompt template and structured response schema for LLM-based
# ABOUTME: synthetic data distillation on Yelp place recommendation task.

"""
Teacher prompt template and structured response schema.

The Teacher LLM (Gemini 3 Flash Preview or Qwen3.5-35B-A3B) is asked to:
1. Infer a short user persona from visit history
2. Provide per-candidate recommendation rationales
3. Produce a ranked order of candidates

Candidates are referenced by 1-based positional index (not business_id) so that
both teacher and student models can reliably reproduce short integers instead of
opaque 22-char Yelp hashes. The mapping index→business_id is applied at eval
time using the ordered candidate list from the preprocessed sample.

Output is constrained to JSON via Gemini's response_schema (or vLLM guided
decoding) to guarantee parseable Student training data.
"""

from typing import Any, Literal

from pydantic import BaseModel, Field


# ---------- Response schema (Pydantic models used by Gemini / vLLM) ----------

# Number of candidates per sample. Matches the fixed candidate count produced
# by scripts/preprocess_yelp.py. Embedded into the Pydantic schema below as
# Literal + min_length/max_length constraints so guided-decoding backends
# (Gemini response_schema, vLLM xgrammar/outlines) can enforce
# {1..N_CANDIDATES} membership and exact length at the token level —
# eliminating the out-of-range-int failures (~65/66) we saw under loose
# `int` typing of Qwen3.5-35B outputs.
#
# `uniqueItems` is intentionally NOT used because vLLM's xgrammar backend
# explicitly rejects it (has_xgrammar_unsupported_json_features marks it
# unsupported). Cross-backend portability matters more than catching the
# residual ~1/3000 duplicate-ranking case, which validate_teacher.py
# already detects post-hoc.
N_CANDIDATES = 10
CandidateIndex = Literal[1, 2, 3, 4, 5, 6, 7, 8, 9, 10]


class Rationale(BaseModel):
    """Per-candidate recommendation reasoning."""

    candidate_index: CandidateIndex = Field(
        ...,
        description=(
            "1-based index of the candidate this rationale refers to "
            "(matches the numbering in the candidate list)"
        ),
    )
    reason: str = Field(
        ...,
        description=(
            "1-2 sentences explaining why this place fits (or does not fit) "
            "the user's inferred taste, grounded in their visit history."
        ),
    )


class TeacherResponse(BaseModel):
    """Top-level structured output the Teacher must produce."""

    persona: str = Field(
        ...,
        description=(
            "2-3 sentences summarizing the user's inferred dining/outing "
            "preferences: cuisines, ambience, price sensitivity, social mode."
        ),
    )
    rationales: list[Rationale] = Field(
        ...,
        min_length=N_CANDIDATES,
        max_length=N_CANDIDATES,
        description=(
            "One rationale per candidate, covering all candidates passed in."
        ),
    )
    ranking: list[CandidateIndex] = Field(
        ...,
        min_length=N_CANDIDATES,
        max_length=N_CANDIDATES,
        description=(
            "Candidate indices (1-based), ordered from best fit to worst fit. "
            "Must contain each candidate index exactly once."
        ),
    )


# ---------- Prompt template ----------


SYSTEM_INSTRUCTION = (
    "You are an expert local place recommender. You read a user's past visits "
    "(venues, categories, star ratings, review snippets) and recommend which "
    "of the given candidate places they are most likely to enjoy next. You "
    "explain your reasoning concisely and return a ranked list.\n\n"
    "Respond with a single JSON object that has exactly these three keys:\n"
    '  - "persona" (string): 2-3 sentences summarizing the user\'s inferred '
    "dining/outing preferences.\n"
    '  - "rationales" (array of 10 objects): one object per candidate in the '
    'order presented. Each object has exactly two keys: "candidate_index" '
    "(integer 1-10 matching the candidate number) and \"reason\" (string, 1-2 "
    "sentences grounded in the user's history).\n"
    '  - "ranking" (array of 10 integers): candidate indices 1-10 sorted from '
    "most likely to least likely next visit. Each index must appear exactly once.\n"
    "Do not include any other keys, comments, code fences, or text outside the JSON object."
)


USER_PROMPT_TEMPLATE = """\
Recommend places for one user based on the history and candidate list below.

[USER VISIT HISTORY] ({num_history} places, chronological)
{history_block}

[CANDIDATE PLACES] ({num_candidates} places)
{candidates_block}

Task:
1. Infer a concise 2-3 sentence persona capturing the user's preferences.
2. For EACH candidate (by its number), write a 1-2 sentence rationale about fit.
3. Produce a ranked list of candidate numbers (1-based) from best fit to worst fit.

Return valid JSON matching the provided schema. Do not add any other text.\
"""


# ---------- Helpers to render user/candidate blocks ----------


def _format_history_item(idx: int, item: dict[str, Any]) -> str:
    """Render one history entry as a short bullet.

    Args:
        idx (int): 1-based index for display.
        item (dict): history entry with keys
            business_id, name, categories, stars, review_snippet.

    Returns:
        str: formatted bullet line.
    """
    stars = item.get("stars")
    stars_str = f"{stars:.0f}" if isinstance(stars, (int, float)) else "?"
    snippet = (item.get("review_snippet") or "").strip().replace("\n", " ")
    if len(snippet) > 180:
        snippet = snippet[:180].rstrip() + "…"
    return (
        f"{idx}. {item.get('name', '(unknown)')} "
        f"[{item.get('categories', '')}] "
        f"user_rating={stars_str}/5 "
        f'review="{snippet}"'
    )


def _format_candidate_item(idx: int, cand: dict[str, Any]) -> str:
    """Render one candidate place as a short bullet.

    The output intentionally omits business_id — candidates are referenced
    by their 1-based index so that teacher/student models output small
    integers instead of opaque 22-char Yelp hashes.

    Args:
        idx (int): 1-based index for display.
        cand (dict): candidate entry with keys
            name, categories, avg_stars, attributes (optional).

    Returns:
        str: formatted bullet line.
    """
    avg = cand.get("avg_stars")
    avg_str = f"{avg:.1f}" if isinstance(avg, (int, float)) else "?"
    attrs = cand.get("attributes") or {}
    # Keep only a handful of useful attribute hints
    attr_keys = (
        "RestaurantsPriceRange2",
        "NoiseLevel",
        "RestaurantsAttire",
        "Ambience",
        "RestaurantsGoodForGroups",
        "GoodForKids",
    )
    shown = {k: attrs[k] for k in attr_keys if k in attrs}
    attr_str = ", ".join(f"{k}={v}" for k, v in shown.items()) if shown else "—"
    return (
        f"{idx}. {cand.get('name', '(unknown)')} "
        f"[{cand.get('categories', '')}] "
        f"avg_rating={avg_str}/5 attrs=({attr_str})"
    )


def build_user_prompt(sample: dict[str, Any]) -> str:
    """Build the user-side prompt string from a preprocessed sample.

    Args:
        sample (dict): one record from samples.jsonl with keys
            history (list[dict]), candidates (list[dict]).

    Returns:
        str: fully formatted user prompt ready to send to Gemini.
    """
    history = sample.get("history", [])
    candidates = sample.get("candidates", [])

    history_lines = [_format_history_item(i + 1, h) for i, h in enumerate(history)]
    candidate_lines = [_format_candidate_item(i + 1, c) for i, c in enumerate(candidates)]

    return USER_PROMPT_TEMPLATE.format(
        num_history=len(history),
        num_candidates=len(candidates),
        history_block="\n".join(history_lines) if history_lines else "(none)",
        candidates_block="\n".join(candidate_lines) if candidate_lines else "(none)",
    )


# ---------- Generation config helper ----------


def build_gemini_response_schema_dict() -> dict[str, Any]:
    """Build a Gemini-compatible response_schema dict for TeacherResponse.

    google-genai's ``types.Schema`` types ``enum`` as ``Optional[list[str]]``
    and rejects integer enums at validation time (observed on google-genai
    1.72.0; google-genai/python-genai). Passing the canonical Pydantic class
    ``TeacherResponse`` (whose ``CandidateIndex = Literal[1..10]`` is an int
    enum) therefore raises ``ValidationError`` before the API call. This helper
    builds an equivalent schema dict using string enums so the SDK accepts it
    and emits token-constrained ``"1".."10"`` strings; downstream readers must
    coerce those digit strings back to int (see
    ``scripts.teacher.generate_teacher.coerce_indices_to_int``).

    The vLLM xgrammar path (``TeacherResponse.model_json_schema()``) is
    unaffected — int enums are still emitted on disk for both Gemini and
    Qwen teachers, preserving the canonical Pydantic contract used by
    ``validate_teacher.py`` and the trainer.

    Returns:
        dict: response_schema dict with string enums {"1".."10"} and matching
            structural constraints (minItems / maxItems / required).
    """
    index_enum = [str(i) for i in range(1, N_CANDIDATES + 1)]
    return {
        "type": "OBJECT",
        "properties": {
            "persona": {
                "type": "STRING",
                "description": (
                    "2-3 sentences summarizing the user's inferred dining/outing "
                    "preferences: cuisines, ambience, price sensitivity, social mode."
                ),
            },
            "rationales": {
                "type": "ARRAY",
                "minItems": N_CANDIDATES,
                "maxItems": N_CANDIDATES,
                "description": (
                    "One rationale per candidate, covering all candidates passed in."
                ),
                "items": {
                    "type": "OBJECT",
                    "properties": {
                        "candidate_index": {
                            "type": "STRING",
                            "enum": index_enum,
                            "description": (
                                "1-based index of the candidate this rationale "
                                "refers to (matches the numbering in the candidate "
                                "list). Emitted as a digit string for SDK schema "
                                "compatibility; readers coerce to int."
                            ),
                        },
                        "reason": {
                            "type": "STRING",
                            "description": (
                                "1-2 sentences explaining why this place fits "
                                "(or does not fit) the user's inferred taste, "
                                "grounded in their visit history."
                            ),
                        },
                    },
                    "required": ["candidate_index", "reason"],
                },
            },
            "ranking": {
                "type": "ARRAY",
                "minItems": N_CANDIDATES,
                "maxItems": N_CANDIDATES,
                "description": (
                    "Candidate indices (1-based), ordered from best fit to worst "
                    "fit. Emitted as digit strings for SDK schema compatibility; "
                    "readers coerce to int."
                ),
                "items": {"type": "STRING", "enum": index_enum},
            },
        },
        "required": ["persona", "rationales", "ranking"],
    }


def build_generation_config(thinking_level: str = "minimal") -> dict[str, Any]:
    """Build generation_config dict for gemini_parallel.

    Args:
        thinking_level (str): one of 'minimal', 'low', 'medium', 'high'.
            Default 'minimal' for speed — this task does not need deep reasoning
            and latency is a JD goal.

    Returns:
        dict: generation_config suitable for GeminiSequentialProcessor.
    """
    return {
        "temperature": 1.0,  # Gemini 3 default; lower values can degrade quality
        "response_mime_type": "application/json",
        "response_schema": build_gemini_response_schema_dict(),
        "system_instruction": SYSTEM_INSTRUCTION,
        "thinking_config": {"thinking_level": thinking_level},
        "max_output_tokens": 2048,
    }

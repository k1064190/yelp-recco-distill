# ABOUTME: Variant C — original minimal SYSTEM_INSTRUCTION (no JSON/schema hint).
# ABOUTME: Baseline for ablating what prompt-side schema guidance adds beyond free text.

"""
Original SYSTEM_INSTRUCTION variant — no explicit schema hint;
model is told only to "return a ranked list" in natural language. This was the
instruction under which the v2/v3 training data was generated under
``response_format=json_schema`` — the schema shape came entirely from the vLLM
grammar mask, not the prompt.

All other exports (``TeacherResponse``, ``build_user_prompt``, etc.) are
re-exported from ``configs.teacher_prompt`` so downstream code that
``from configs.teacher_prompt_original import *`` keeps working.
"""

from __future__ import annotations

from configs.teacher_prompt import *  # noqa: F401, F403


SYSTEM_INSTRUCTION = (
    "You are an expert local place recommender. You read a user's past visits "
    "(venues, categories, star ratings, review snippets) and recommend which "
    "of the given candidate places they are most likely to enjoy next. You "
    "explain your reasoning concisely and return a ranked list."
)

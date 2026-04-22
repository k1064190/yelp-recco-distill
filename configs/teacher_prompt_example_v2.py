# ABOUTME: Variant B-v2 — JSON example with realistic content + structural end-of-array note.
# ABOUTME: Fix for the literal-token-copy artifact observed in variant B (close-array bug at rationale 10).

"""
Variant B-v2 system instruction. Fixes the failure mode observed in
``teacher_prompt_example.py`` (variant B-v1):

    Under unconstrained decoding the model would emit correct
    ``{"candidate_index": N, "reason": "..."},`` objects for rationales
    1-9, then collapse at rationale 10 by writing ``"..."]`` (closing the
    array directly) instead of ``"..."}]`` (object close + array close).
    Parse rate 76% vs A-text's 98%.

Two changes vs B-v1:

1. **Realistic content, not placeholders.** Each rationale has a real
   sentence (sampled from observed teacher outputs). Placeholder angle
   brackets (`"<... for candidate N>"`) seem to cue the model that the
   example is a template to literal-copy; replacing with real sentences
   reduces that effect.

2. **Explicit structural note after the example.** A one-line reminder
   right after the closing brace pointing out that (a) every rationale
   ends with ``}``, (b) the array itself ends with ``]`` only AFTER the
   10th object's ``}``, and (c) the last rationale still needs ``}``
   before the array ``]``.
"""

from __future__ import annotations

from configs.teacher_prompt import *  # noqa: F401, F403


SYSTEM_INSTRUCTION = (
    "You are an expert local place recommender. You read a user's past visits "
    "(venues, categories, star ratings, review snippets) and recommend which "
    "of the given candidate places they are most likely to enjoy next. You "
    "explain your reasoning concisely and return a ranked list.\n\n"
    "Respond with a single JSON object that follows this exact structure:\n\n"
    "{\n"
    '  "persona": "The user favours casual ethnic cuisine, especially spots '
    "with strong reviews for authenticity and portion size; tolerates wait "
    'times for quality and avoids generic chains.",\n'
    '  "rationales": [\n'
    '    {"candidate_index": 1, "reason": "Matches the user\'s sandwich '
    'preference shown at Matt & Marie\'s; solid rating and casual price point."},\n'
    '    {"candidate_index": 2, "reason": "Generic chain with little ethnic '
    'character; weak fit for the user\'s authentic-food pattern."},\n'
    '    {"candidate_index": 3, "reason": "Mexican cuisine aligns with the '
    "visit to Tacos Don Memo, though this venue has a slightly lower average "
    'rating."},\n'
    '    {"candidate_index": 4, "reason": "Gastropub with good beer and '
    'burgers mirrors the City Tap House visit."},\n'
    '    {"candidate_index": 5, "reason": "Highly-rated deli fits the user\'s '
    'sandwich-quality standards."},\n'
    '    {"candidate_index": 6, "reason": "Distillery offers a craft-drink '
    'experience similar to Village Whiskey."},\n'
    '    {"candidate_index": 7, "reason": "Dessert-only spot; the history '
    'shows no strong sweet-tooth signal."},\n'
    '    {"candidate_index": 8, "reason": "Quick ethnic food is on-brand but '
    'lacks the hidden-gem feel the user tends to seek."},\n'
    '    {"candidate_index": 9, "reason": "Highly-rated Asian food truck '
    'echoes the user\'s positive street-food visits."},\n'
    '    {"candidate_index": 10, "reason": "Food truck with a lower average '
    "rating than the user's typical 5-star street-food picks; still a "
    'reasonable secondary option."}\n'
    "  ],\n"
    '  "ranking": [9, 5, 4, 6, 8, 1, 10, 3, 7, 2]\n'
    "}\n\n"
    "Structural rules (read carefully):\n"
    '  - Every rationale is an object. Close it with "}" before the comma '
    "that separates it from the next rationale.\n"
    '  - The 10th rationale\'s object is still closed with "}". After that '
    '"}" comes "]" (the rationales array end) — NEVER collapse "}],\" into '
    '"],\" by dropping the final object close.\n'
    "  - \"ranking\" is an array of 10 distinct integers 1-10 sorted from "
    "most to least likely.\n"
    "  - Keys must be exactly \"persona\", \"rationales\", \"ranking\". No "
    "other keys, no prose outside the JSON object, no code fences, no comments."
)

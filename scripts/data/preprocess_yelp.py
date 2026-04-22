#!/usr/bin/env python
# ABOUTME: Preprocess raw Yelp Open Dataset JSON files into per-user training
# ABOUTME: samples (history + candidates + positive target) for KD pipeline.

"""
Preprocess Yelp Open Dataset into training samples for synthetic-data KD.

Produces one sample per user: the user's past visit history within a chosen
city, plus a candidate set composed of 1 held-out positive (the user's most
recent visit) and K-1 random negatives drawn from businesses the user has
NOT visited.

Sample JSONL schema:
    sample_id              str   unique identifier
    user_id                str   Yelp user_id
    city                   str   filter city (e.g. "Philadelphia")
    history                list  past visits (oldest -> most recent)
        business_id        str
        name               str
        categories         str   comma-separated
        stars              float user's rating for this visit
        review_snippet     str   first ~200 chars of the review text
        date               str   ISO date string
    candidates             list  exactly K entries, shuffled
        business_id        str
        name               str
        categories         str
        attributes         dict  selected Yelp attributes
        avg_stars          float business-level average rating
        review_count       int
    positive_business_id   str   ground-truth next visit (held out)

Example:
    $ python scripts/data/preprocess_yelp.py \\
        --business data/raw/yelp_academic_dataset_business.json \\
        --review data/raw/yelp_academic_dataset_review.json \\
        --city "Philadelphia" --state "PA" \\
        --min-history 8 --num-candidates 10 --max-users 2000 \\
        --output data/processed/philly_samples.jsonl
"""

from __future__ import annotations

import argparse
import json
import logging
import random
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterator

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("preprocess_yelp")


# Categories that should count as "place/food" for a place-recommendation task.
# A business is kept iff ANY of its comma-separated categories matches these
# (case-insensitive substring match). Intentionally broad — we trust the Teacher
# to handle cross-category taste.
FOOD_KEYWORDS = {
    "restaurants",
    "food",
    "bars",
    "cafes",
    "coffee",
    "bakeries",
    "breakfast",
    "desserts",
    "nightlife",
    "ice cream",
    "tea",
    "pubs",
    "diners",
    "delis",
    "sandwiches",
    "pizza",
    "burgers",
    "seafood",
    "steakhouses",
    "sushi",
    "barbeque",
}


def iter_jsonl(path: Path) -> Iterator[dict[str, Any]]:
    """Stream JSON-per-line records from a Yelp dump file.

    Args:
        path (Path): path to a Yelp .json file (one JSON object per line).

    Yields:
        dict: parsed record.
    """
    with path.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError as e:
                log.warning("skipping bad line %d in %s: %s", line_no, path.name, e)


def is_food_business(categories: str | None) -> bool:
    """Return True if the categories string matches any food/place keyword.

    Args:
        categories (str | None): comma-separated Yelp categories.

    Returns:
        bool: True if the business is a food/place candidate.
    """
    if not categories:
        return False
    lower = categories.lower()
    return any(kw in lower for kw in FOOD_KEYWORDS)


def load_businesses(
    path: Path, city: str, state: str
) -> dict[str, dict[str, Any]]:
    """Load Yelp businesses filtered by city + state + food categories.

    Args:
        path (Path): path to yelp_academic_dataset_business.json.
        city (str): city name filter (exact match, case-insensitive).
        state (str): 2-letter state code filter.

    Returns:
        dict[str, dict]: business_id -> trimmed business record with fields
            name, categories, attributes, stars, review_count.
    """
    log.info("loading businesses from %s ...", path.name)
    city_norm = city.strip().lower()
    state_norm = state.strip().upper()
    kept: dict[str, dict[str, Any]] = {}
    total = 0
    for rec in iter_jsonl(path):
        total += 1
        if (rec.get("city") or "").strip().lower() != city_norm:
            continue
        if (rec.get("state") or "").strip().upper() != state_norm:
            continue
        if not is_food_business(rec.get("categories")):
            continue
        kept[rec["business_id"]] = {
            "business_id": rec["business_id"],
            "name": rec.get("name", ""),
            "categories": rec.get("categories", ""),
            "attributes": rec.get("attributes") or {},
            "avg_stars": rec.get("stars"),
            "review_count": rec.get("review_count"),
        }
    log.info(
        "kept %d / %d businesses matching city=%s state=%s + food categories",
        len(kept),
        total,
        city,
        state,
    )
    return kept


def build_user_histories(
    review_path: Path,
    business_ids: set[str],
    min_history: int,
) -> dict[str, list[dict[str, Any]]]:
    """Stream reviews and collect per-user visit lists within the filter city.

    Args:
        review_path (Path): path to yelp_academic_dataset_review.json.
        business_ids (set[str]): set of business_ids that pass the city filter.
        min_history (int): drop users with fewer than this many in-city reviews.

    Returns:
        dict[str, list[dict]]: user_id -> list of review dicts sorted chronologically.
            Each review dict contains business_id, stars, date, text.
    """
    log.info(
        "streaming reviews from %s (keeping only %d in-city businesses) ...",
        review_path.name,
        len(business_ids),
    )
    per_user: dict[str, list[dict[str, Any]]] = defaultdict(list)
    kept = 0
    total = 0
    for rec in iter_jsonl(review_path):
        total += 1
        if total % 500000 == 0:
            log.info("  scanned %d reviews, kept %d", total, kept)
        bid = rec.get("business_id")
        if bid not in business_ids:
            continue
        per_user[rec["user_id"]].append(
            {
                "business_id": bid,
                "stars": rec.get("stars"),
                "date": rec.get("date", ""),
                "text": rec.get("text", ""),
            }
        )
        kept += 1
    log.info("scanned %d reviews total, kept %d in-city", total, kept)

    # Sort each user's reviews chronologically, then filter by min length.
    filtered: dict[str, list[dict[str, Any]]] = {}
    for uid, reviews in per_user.items():
        if len(reviews) < min_history:
            continue
        reviews.sort(key=lambda r: r["date"])
        filtered[uid] = reviews
    log.info(
        "%d users have >= %d in-city reviews (out of %d total in-city users)",
        len(filtered),
        min_history,
        len(per_user),
    )
    return filtered


def truncate_snippet(text: str, max_chars: int = 200) -> str:
    """Return a short, single-line snippet of a review body.

    Args:
        text (str): raw review text.
        max_chars (int): maximum length in characters.

    Returns:
        str: cleaned snippet.
    """
    if not text:
        return ""
    cleaned = " ".join(text.split())
    if len(cleaned) <= max_chars:
        return cleaned
    return cleaned[:max_chars].rstrip() + "…"


def build_samples(
    user_histories: dict[str, list[dict[str, Any]]],
    businesses: dict[str, dict[str, Any]],
    num_candidates: int,
    max_users: int | None,
    seed: int,
    city: str,
    max_history: int | None = None,
) -> Iterator[dict[str, Any]]:
    """Yield one preprocessed sample per user.

    For each user:
        history  = in-city reviews EXCEPT the last one, then (optionally)
                   truncated to the ``max_history`` most recent entries.
        positive = the last in-city review (held out as ground truth).
        candidates = [positive] + (num_candidates - 1) random negatives
                     drawn from in-city businesses the user has NEVER
                     visited (based on the FULL review history, not the
                     truncated one — so a negative is never a place the
                     user visited even long ago).
        Candidates are then shuffled so the positive position is not leaked.

    ``sample_id`` is derived from user_id alone (``f"{uid[:16]}_s0"``) so
    turning on ``max_history`` only changes the ``history`` field, not the
    sample identity. Existing teacher records keyed by sample_id remain
    matchable; what changes is only how much context the student sees
    compared to what the teacher saw at generation time. Re-generating the
    teacher after lowering ``max_history`` is the correct way to keep
    teacher and student inputs consistent.

    Args:
        user_histories (dict): user_id -> chronological review list.
        businesses (dict): business_id -> business meta dict.
        num_candidates (int): total candidate list size (positive + negatives).
        max_users (int | None): cap on number of samples produced.
        seed (int): RNG seed for reproducible negative sampling.
        city (str): city name to store on each sample.
        max_history (int | None): if set, keep only the most recent N
            history entries per user (before the held-out positive).
            ``None`` disables truncation (legacy behavior).

    Yields:
        dict: one sample record matching the documented JSONL schema.
    """
    rng = random.Random(seed)
    all_business_ids = list(businesses.keys())
    users_sorted = sorted(user_histories.keys())  # deterministic order
    produced = 0
    for uid in users_sorted:
        if max_users is not None and produced >= max_users:
            break
        reviews = user_histories[uid]
        if len(reviews) < 2:
            continue  # need at least 1 history + 1 positive

        *history_reviews, positive_review = reviews
        positive_bid = positive_review["business_id"]
        if positive_bid not in businesses:
            continue

        # Negatives are sampled from businesses the user has NEVER visited —
        # use the FULL review list here so an old visit still counts as
        # "visited", even if it will be dropped from the rendered history.
        visited_ids = {r["business_id"] for r in reviews}
        negatives_pool = [b for b in all_business_ids if b not in visited_ids]
        n_negatives = num_candidates - 1
        if len(negatives_pool) < n_negatives:
            continue
        negatives = rng.sample(negatives_pool, n_negatives)

        # Apply max_history AFTER visited_ids / negatives are settled. Reviews
        # are sorted oldest -> newest, so [-max_history:] keeps the user's
        # most recent N visits, which is the realistic deployment condition:
        # a production user also has a finite context window of recent
        # interactions, not a lifelong Yelp dump.
        if max_history is not None and max_history > 0:
            history_reviews = history_reviews[-max_history:]

        # Build candidate dicts (positive + negatives), then shuffle.
        candidate_ids = [positive_bid, *negatives]
        rng.shuffle(candidate_ids)
        candidates = [
            {
                "business_id": bid,
                "name": businesses[bid]["name"],
                "categories": businesses[bid]["categories"],
                "attributes": businesses[bid]["attributes"],
                "avg_stars": businesses[bid]["avg_stars"],
                "review_count": businesses[bid]["review_count"],
            }
            for bid in candidate_ids
        ]

        history = [
            {
                "business_id": r["business_id"],
                "name": businesses[r["business_id"]]["name"],
                "categories": businesses[r["business_id"]]["categories"],
                "stars": r["stars"],
                "review_snippet": truncate_snippet(r.get("text", "")),
                "date": r.get("date", ""),
            }
            for r in history_reviews
            if r["business_id"] in businesses  # defensive
        ]

        sample = {
            "sample_id": f"{uid[:16]}_s0",
            "user_id": uid,
            "city": city,
            "history": history,
            "candidates": candidates,
            "positive_business_id": positive_bid,
        }
        yield sample
        produced += 1
    log.info("produced %d samples", produced)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--business", type=Path, required=True, help="path to business.json")
    p.add_argument("--review", type=Path, required=True, help="path to review.json")
    p.add_argument("--city", type=str, default="Philadelphia")
    p.add_argument("--state", type=str, default="PA")
    p.add_argument(
        "--min-history",
        type=int,
        default=8,
        help="minimum in-city reviews required per user (incl. held-out positive)",
    )
    p.add_argument(
        "--max-history",
        type=int,
        default=20,
        help=(
            "keep only the most recent N in-city reviews per user when "
            "rendering the history block. Prevents prompt-length blow-up "
            "from Yelp power users with 300+ visits. Set to 0 to disable "
            "truncation (not recommended — the full 3000-sample profile on "
            "2026-04-11 showed p99 = 187 history items, max = 516, leading "
            "to >20K-token prompts)."
        ),
    )
    p.add_argument(
        "--num-candidates",
        type=int,
        default=10,
        help="candidate list size (positive + negatives)",
    )
    p.add_argument(
        "--max-users",
        type=int,
        default=None,
        help="cap on samples produced (default: no cap)",
    )
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--output", type=Path, required=True, help="output JSONL path")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    args.output.parent.mkdir(parents=True, exist_ok=True)

    businesses = load_businesses(args.business, args.city, args.state)
    if not businesses:
        log.error("no businesses matched filters; aborting")
        sys.exit(1)

    user_histories = build_user_histories(
        args.review, set(businesses), args.min_history
    )
    if not user_histories:
        log.error("no users matched min-history filter; aborting")
        sys.exit(1)

    log.info("writing samples to %s ...", args.output)
    with args.output.open("w", encoding="utf-8") as out:
        count = 0
        for sample in build_samples(
            user_histories=user_histories,
            businesses=businesses,
            num_candidates=args.num_candidates,
            max_users=args.max_users,
            seed=args.seed,
            city=args.city,
            max_history=(args.max_history if args.max_history > 0 else None),
        ):
            out.write(json.dumps(sample, ensure_ascii=False) + "\n")
            count += 1
    log.info("wrote %d samples to %s", count, args.output)


if __name__ == "__main__":
    main()

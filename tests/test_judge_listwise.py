# ABOUTME: Unit tests for scripts/judge_listwise.py and analyze_judge_listwise.py
# ABOUTME: -- schema, prompt rendering, resume logic, aggregation, retrieval join.

from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from pydantic import ValidationError

from scripts.judge.judge_listwise import (  # noqa: E402
    JUDGE_SYSTEM_LISTWISE,
    JUDGE_USER_TEMPLATE,
    RUBRIC_VERSION,
    ListwiseVerdict,
    _build_ranking_block,
    _build_rationales_block,
    aggregate_per_model,
    bootstrap_mean_ci,
    build_candidates_block,
    build_history_block,
    build_judge_prompt_listwise,
    load_done_keys,
    load_inference_cache,
    parse_model_output,
    pick_eval_samples,
)
from scripts.judge.analyze_judge_listwise import (  # noqa: E402
    _safe_pointbiserialr,
    _safe_spearmanr,
    analyze,
    per_sample_retrieval,
    render_report_md,
)


# ---------- Test fixtures ----------


def _sample_fixture() -> dict:
    """Build a minimal preprocessed sample with 2 history items + 3 candidates."""
    return {
        "sample_id": "sid-A",
        "user_id": "user-A",
        "city": "Philadelphia",
        "positive_business_id": "bid-3",
        "history": [
            {
                "business_id": "h1",
                "name": "Whiskey Bar",
                "categories": "Bars, Whiskey Bars",
                "stars": 4.0,
                "review_snippet": "love the whiskey selection",
                "date": "2020-01-01",
            },
            {
                "business_id": "h2",
                "name": "Pizza Place",
                "categories": "Pizza, Italian",
                "stars": 5.0,
                "review_snippet": "best slice in the city",
                "date": "2020-02-01",
            },
        ],
        "candidates": [
            {"business_id": "bid-1", "name": "Cand One", "categories": "Pizza", "avg_stars": 3.5, "review_count": 10, "attributes": {}},
            {"business_id": "bid-2", "name": "Cand Two", "categories": "Bars", "avg_stars": 4.5, "review_count": 50, "attributes": {}},
            {"business_id": "bid-3", "name": "Cand Three", "categories": "Italian", "avg_stars": 4.0, "review_count": 20, "attributes": {}},
        ],
    }


def _model_output_fixture(persona: str = "Foodie who likes whiskey + pizza.") -> dict:
    return {
        "persona": persona,
        "rationales": [
            {"candidate_index": 1, "reason": "fits pizza preference"},
            {"candidate_index": 2, "reason": "fits whiskey bar preference"},
            {"candidate_index": 3, "reason": "italian aligns with pizza history"},
        ],
        "ranking": [3, 2, 1],
    }


# ---------- ListwiseVerdict schema ----------


class TestListwiseVerdictSchema:
    def test_valid_payload_round_trip(self):
        v = ListwiseVerdict(
            groundedness=8,
            groundedness_evidence="G_exact=10 avg_fields=1.8; rationale #2 cites Bars category",
            personalization=9,
            personalization_evidence="P_specific=4 P_nontrivial=1 R_link=10/10",
            ranking_coherence=10,
            ranking_coherence_evidence="tones [+,+,~,-,+,...]; R_reverse=0; top3_positive=3",
        )
        d = v.model_dump()
        assert d["groundedness"] == 8
        assert d["personalization"] == 9
        assert d["ranking_coherence"] == 10
        # Re-validate to ensure round-trip stability.
        ListwiseVerdict.model_validate(d)

    @pytest.mark.parametrize("bad_value", [0, 11, -1, 100])
    def test_score_out_of_range_rejected(self, bad_value):
        base = dict(
            groundedness=5,
            groundedness_evidence="x",
            personalization=5,
            personalization_evidence="y",
            ranking_coherence=5,
            ranking_coherence_evidence="z",
        )
        for field in ("groundedness", "personalization", "ranking_coherence"):
            kwargs = {**base, field: bad_value}
            with pytest.raises(ValidationError):
                ListwiseVerdict(**kwargs)

    def test_evidence_strings_required(self):
        # Missing any of the three axis scores or their evidence must fail.
        with pytest.raises(ValidationError):
            ListwiseVerdict(groundedness=3, personalization=3, ranking_coherence=3)  # type: ignore[call-arg]

    def test_non_int_score_rejected(self):
        with pytest.raises(ValidationError):
            ListwiseVerdict(
                groundedness=3.5,  # type: ignore[arg-type]
                groundedness_evidence="x",
                personalization=3,
                personalization_evidence="y",
                ranking_coherence=3,
                ranking_coherence_evidence="z",
            )


# ---------- Prompt rendering ----------


class TestPromptRendering:
    def test_history_block_numbers_items_one_based(self):
        s = _sample_fixture()
        block = build_history_block(s)
        assert block.startswith("1. Whiskey Bar")
        assert "\n2. Pizza Place" in block

    def test_history_block_empty(self):
        assert build_history_block({"history": []}) == "(none)"

    def test_candidates_block_numbers_items_one_based(self):
        s = _sample_fixture()
        block = build_candidates_block(s)
        assert block.startswith("1. Cand One")
        assert "\n2. Cand Two" in block
        assert "\n3. Cand Three" in block

    def test_candidates_block_omits_business_id(self):
        s = _sample_fixture()
        block = build_candidates_block(s)
        # business_id is intentionally hidden in the prompt (index-only schema).
        assert "bid-1" not in block

    def test_rationales_block_sorts_by_candidate_index(self):
        rationales = [
            {"candidate_index": 3, "reason": "third"},
            {"candidate_index": 1, "reason": "first"},
            {"candidate_index": 2, "reason": "second"},
        ]
        block = _build_rationales_block(rationales)
        lines = block.splitlines()
        assert lines[0].strip().startswith("1. first")
        assert lines[1].strip().startswith("2. second")
        assert lines[2].strip().startswith("3. third")

    def test_rationales_block_empty(self):
        assert _build_rationales_block([]) == "  (none)"
        assert _build_rationales_block(None) == "  (none)"

    def test_ranking_block_renders_csv(self):
        assert _build_ranking_block([3, 1, 2]) == "3, 1, 2"

    def test_ranking_block_empty(self):
        assert _build_ranking_block([]) == "(none)"
        assert _build_ranking_block(None) == "(none)"

    def test_full_prompt_contains_key_sections(self):
        s = _sample_fixture()
        m = _model_output_fixture()
        prompt = build_judge_prompt_listwise(s, m)
        for needle in (
            "[USER VISIT HISTORY]",
            "[CANDIDATE PLACES]",
            "[MODEL OUTPUT]",
            "Persona:",
            "Rationales",
            "Ranking",
            "GROUNDEDNESS",
            "PERSONALIZATION",
            "RANKING COHERENCE",
            "Whiskey Bar",
            "Cand Three",
            "fits pizza preference",
            "3, 2, 1",
        ):
            assert needle in prompt, f"prompt missing: {needle!r}"

    def test_v3_rubric_enforces_three_axis_verification(self):
        # Guard against accidental regression. v3 must explicitly require
        # per-rationale field-citation depth (avg_fields, G_attr_N),
        # non-trivial persona inference (P_nontrivial), and ranking-vs-
        # rationale tone coherence (R_reverse on a dedicated axis).
        for needle in (
            "VERIFICATION STEP 1",
            "VERIFICATION STEP 2",
            "VERIFICATION STEP 3",
            "G_exact",
            "G_wrong",
            "G_hallu",
            "avg_fields",
            "G_attr_N",
            "P_specific",
            "P_nontrivial",
            "R_link",
            "R_reverse",
            "R_top3_positive",
            "R_bottom3_negative",
            "caps Groundedness at 2",
            "caps Personalization at 4",
        ):
            assert needle in JUDGE_USER_TEMPLATE, f"v3 rubric missing: {needle!r}"
        assert "strict" in JUDGE_SYSTEM_LISTWISE.lower()
        assert "1-10" in JUDGE_SYSTEM_LISTWISE
        assert RUBRIC_VERSION == "v3"

    def test_full_prompt_handles_empty_persona(self):
        s = _sample_fixture()
        m = _model_output_fixture(persona="   ")
        prompt = build_judge_prompt_listwise(s, m)
        assert "Persona:\n  (empty)" in prompt


# ---------- Inference cache parse + load ----------


def _write_cache(tmp_path: Path) -> Path:
    """Write a minimal inference-cache JSON suitable for unit testing."""
    cache = {
        "generated_at": "2026-04-15T00:00:00",
        "samples": [
            {
                "sample_id": "sid-A",
                "positive_business_id": "bid-3",
                "by_backend": {
                    "teacher": {
                        "output_text": json.dumps(
                            {
                                "persona": "T persona",
                                "rationales": [
                                    {"candidate_index": 1, "reason": "t1"},
                                    {"candidate_index": 2, "reason": "t2"},
                                    {"candidate_index": 3, "reason": "t3"},
                                ],
                                "ranking": [3, 2, 1],
                            }
                        ),
                        "parsed_ranking": [3, 2, 1],
                        "recovered_business_ids": ["bid-3", "bid-2", "bid-1"],
                        "json_parse_ok": True,
                        "output_tokens": 100,
                    },
                    "v2-sft": {
                        "output_text": json.dumps(
                            {
                                "persona": "S persona",
                                "rationales": [
                                    {"candidate_index": 1, "reason": "s1"},
                                    {"candidate_index": 2, "reason": "s2"},
                                    {"candidate_index": 3, "reason": "s3"},
                                ],
                                "ranking": [1, 2, 3],
                            }
                        ),
                        "parsed_ranking": [1, 2, 3],
                        "recovered_business_ids": ["bid-1", "bid-2", "bid-3"],
                        "json_parse_ok": True,
                        "output_tokens": 80,
                    },
                    "broken": {
                        "output_text": "not valid json {",
                        "parsed_ranking": None,
                        "recovered_business_ids": [],
                        "json_parse_ok": False,
                        "output_tokens": 5,
                    },
                },
            },
        ],
    }
    p = tmp_path / "all_backends_merged.json"
    p.write_text(json.dumps(cache))
    return p


class TestInferenceCache:
    def test_load_inference_cache_indexes_by_sample_id(self, tmp_path):
        cache_path = _write_cache(tmp_path)
        cache = load_inference_cache(cache_path)
        assert "sid-A" in cache
        assert "v2-sft" in cache["sid-A"]["by_backend"]

    def test_load_inference_cache_rejects_bad_shape(self, tmp_path):
        bad = tmp_path / "bad.json"
        bad.write_text(json.dumps({"foo": "bar"}))
        with pytest.raises(ValueError):
            load_inference_cache(bad)

    def test_parse_model_output_strips_json_fence(self):
        text = "```json\n" + json.dumps({"persona": "p", "rationales": [], "ranking": []}) + "\n```"
        out = parse_model_output(text)
        assert out is not None
        assert out["persona"] == "p"

    def test_parse_model_output_returns_none_on_garbage(self):
        assert parse_model_output("not json") is None
        assert parse_model_output("") is None
        assert parse_model_output(json.dumps([1, 2, 3])) is None  # not a dict

    def test_load_inference_cache_per_backend_shape(self, tmp_path):
        # Shape produced by generate_inference_samples.py --backend teacher:
        # {"backend": "teacher", "samples": [flat records]}.
        per_backend = {
            "backend": "teacher",
            "model_path_or_url": "http://localhost:8100/v1",
            "samples": [
                {
                    "sample_id": "sid-X",
                    "positive_business_id": "bid-Y",
                    "output_text": json.dumps({"persona": "p", "rationales": [], "ranking": []}),
                    "parsed_ranking": [1, 2, 3],
                    "recovered_business_ids": ["a", "b", "bid-Y"],
                    "json_parse_ok": True,
                    "output_tokens": 50,
                },
            ],
        }
        p = tmp_path / "teacher.json"
        p.write_text(json.dumps(per_backend))
        cache = load_inference_cache(p)
        assert "sid-X" in cache
        assert "teacher" in cache["sid-X"]["by_backend"]
        assert cache["sid-X"]["positive_business_id"] == "bid-Y"
        assert cache["sid-X"]["by_backend"]["teacher"]["output_tokens"] == 50

    def test_load_inference_cache_merges_multiple_per_backend_files(self, tmp_path):
        teacher_file = tmp_path / "teacher.json"
        teacher_file.write_text(json.dumps({
            "backend": "teacher",
            "samples": [{"sample_id": "s1", "positive_business_id": "b1",
                         "output_text": "{}", "output_tokens": 10,
                         "recovered_business_ids": ["b1"], "json_parse_ok": True}],
        }))
        student_file = tmp_path / "v2-sft.json"
        student_file.write_text(json.dumps({
            "backend": "v2-sft",
            "samples": [{"sample_id": "s1", "positive_business_id": "b1",
                         "output_text": "{}", "output_tokens": 20,
                         "recovered_business_ids": ["b1"], "json_parse_ok": True}],
        }))
        cache = load_inference_cache([teacher_file, student_file])
        assert set(cache["s1"]["by_backend"]) == {"teacher", "v2-sft"}
        assert cache["s1"]["by_backend"]["teacher"]["output_tokens"] == 10
        assert cache["s1"]["by_backend"]["v2-sft"]["output_tokens"] == 20


# ---------- Sample selection ----------


class TestPickEvalSamples:
    def test_spaced_selection_matches_generate_inference_samples(self):
        # Simulate 283-sample eval pool with surrogate ids.
        eval_exs = [{"sample_id": f"s{i}", "sample": None, "teacher": None}
                    for i in range(283)]
        picked = pick_eval_samples(eval_exs, 5)
        # Hardcoded expected positions from generate_inference_samples.py docstring.
        assert [p["sample_id"] for p in picked] == ["s0", "s56", "s112", "s168", "s224"]

    def test_returns_all_when_n_exceeds_pool(self):
        eval_exs = [{"sample_id": f"s{i}"} for i in range(3)]
        picked = pick_eval_samples(eval_exs, 10)
        assert picked == eval_exs

    def test_empty_inputs(self):
        assert pick_eval_samples([], 5) == []
        assert pick_eval_samples([{"sample_id": "s0"}], 0) == []


# ---------- Resume support ----------


class TestResume:
    def test_load_done_keys_skips_errors(self, tmp_path):
        raw = tmp_path / "raw.jsonl"
        raw.write_text(
            "\n".join(
                [
                    json.dumps({"sample_id": "s1", "model_tag": "m1", "error": None,
                                "groundedness": 4, "personalization": 4,
                                "groundedness_evidence": "x", "personalization_evidence": "y"}),
                    json.dumps({"sample_id": "s2", "model_tag": "m1", "error": "quota",
                                "groundedness": None, "personalization": None,
                                "groundedness_evidence": None, "personalization_evidence": None}),
                    json.dumps({"sample_id": "s1", "model_tag": "m2", "error": None,
                                "groundedness": 3, "personalization": 5,
                                "groundedness_evidence": "x", "personalization_evidence": "y"}),
                    "",  # blank line tolerated
                    "{ malformed }",  # malformed line tolerated
                ]
            )
        )
        done = load_done_keys(raw)
        assert done == {("s1", "m1"), ("s1", "m2")}

    def test_load_done_keys_missing_file(self, tmp_path):
        assert load_done_keys(tmp_path / "nope.jsonl") == set()


# ---------- Aggregation ----------


class TestAggregate:
    def test_bootstrap_mean_ci_basic(self):
        # All-equal sample -> CI collapses to the value itself.
        mean, lo, hi = bootstrap_mean_ci([3.0] * 20, n_resamples=500, seed=42)
        assert mean == pytest.approx(3.0)
        assert lo == pytest.approx(3.0)
        assert hi == pytest.approx(3.0)

    def test_bootstrap_mean_ci_empty(self):
        mean, lo, hi = bootstrap_mean_ci([])
        assert math.isnan(mean) and math.isnan(lo) and math.isnan(hi)

    def test_bootstrap_mean_ci_brackets_true_mean(self):
        # For a moderate-N sample the percentile CI must bracket the
        # observed mean (a sanity property of any bootstrap percentile CI).
        vals = [1.0, 2.0, 2.0, 3.0, 4.0, 5.0, 5.0, 5.0]
        mean, lo, hi = bootstrap_mean_ci(vals, n_resamples=2000, seed=7)
        assert lo <= mean <= hi
        assert lo < hi  # non-degenerate

    def test_aggregate_per_model_excludes_errors(self):
        verdicts = [
            {"sample_id": "s1", "model_tag": "A", "error": None,
             "groundedness": 4, "personalization": 5,
             "groundedness_evidence": "g", "personalization_evidence": "p"},
            {"sample_id": "s2", "model_tag": "A", "error": "quota",
             "groundedness": None, "personalization": None,
             "groundedness_evidence": None, "personalization_evidence": None},
            {"sample_id": "s1", "model_tag": "B", "error": None,
             "groundedness": 2, "personalization": 3,
             "groundedness_evidence": "g", "personalization_evidence": "p"},
        ]
        out = aggregate_per_model(verdicts)
        assert set(out) == {"A", "B"}
        assert out["A"]["n_total"] == 2
        assert out["A"]["n_scored"] == 1
        assert out["A"]["n_errors"] == 1
        assert out["A"]["groundedness"]["mean"] == pytest.approx(4.0)
        assert out["A"]["personalization"]["mean"] == pytest.approx(5.0)
        assert out["B"]["n_scored"] == 1
        assert out["B"]["groundedness"]["mean"] == pytest.approx(2.0)


# ---------- Per-sample retrieval reconstruction ----------


class TestPerSampleRetrieval:
    def test_perfect_top1_hit(self, tmp_path):
        cache_path = _write_cache(tmp_path)
        cache = load_inference_cache(cache_path)
        retr = per_sample_retrieval(cache["sid-A"], "teacher")
        assert retr["hit_at_1"] == 1
        assert retr["hit_at_5"] == 1
        assert retr["mrr"] == pytest.approx(1.0)
        assert retr["rank_of_pos"] == 0

    def test_third_place_hit(self, tmp_path):
        cache_path = _write_cache(tmp_path)
        cache = load_inference_cache(cache_path)
        retr = per_sample_retrieval(cache["sid-A"], "v2-sft")
        # positive bid-3 sits at index 2 in recovered_business_ids
        assert retr["hit_at_1"] == 0
        assert retr["hit_at_5"] == 1
        assert retr["mrr"] == pytest.approx(1.0 / 3.0)
        assert retr["rank_of_pos"] == 2

    def test_missing_backend_returns_nones(self, tmp_path):
        cache_path = _write_cache(tmp_path)
        cache = load_inference_cache(cache_path)
        retr = per_sample_retrieval(cache["sid-A"], "no-such-model")
        assert retr["hit_at_1"] is None
        assert retr["mrr"] is None


# ---------- Correlation guards ----------


class TestCorrelationGuards:
    def test_pointbiserial_zero_variance_returns_nan(self):
        out = _safe_pointbiserialr([3.0, 3.0, 3.0, 3.0], [1, 0, 1, 0])
        assert math.isnan(out["r"])
        assert out["n"] == 4

    def test_pointbiserial_too_few_samples(self):
        out = _safe_pointbiserialr([1.0, 2.0], [1, 0])
        assert math.isnan(out["r"])

    def test_pointbiserial_perfect_positive(self):
        # Higher score iff hit==1 -> r ~= +1.
        scores = [1.0, 2.0, 4.0, 5.0]
        binary = [0, 0, 1, 1]
        out = _safe_pointbiserialr(scores, binary)
        assert out["r"] == pytest.approx(1.0, abs=0.01) or out["r"] > 0.9

    def test_spearman_perfect_positive(self):
        out = _safe_spearmanr([1.0, 2.0, 3.0], [10.0, 20.0, 30.0])
        assert out["rho"] == pytest.approx(1.0)


# ---------- analyze() end-to-end ----------


class TestAnalyzeEndToEnd:
    def test_analyze_combines_judge_and_retrieval(self, tmp_path):
        cache_path = _write_cache(tmp_path)
        raw = tmp_path / "raw.jsonl"
        raw.write_text(
            "\n".join(
                [
                    # Teacher: top-1 hit, judge gives high marks -> agreement.
                    json.dumps({"sample_id": "sid-A", "model_tag": "teacher", "error": None,
                                "groundedness": 5, "personalization": 5,
                                "groundedness_evidence": "g", "personalization_evidence": "p"}),
                    # Student: third-place hit, judge gives mid marks.
                    json.dumps({"sample_id": "sid-A", "model_tag": "v2-sft", "error": None,
                                "groundedness": 3, "personalization": 3,
                                "groundedness_evidence": "g", "personalization_evidence": "p"}),
                ]
            )
        )
        summary = analyze(raw, cache_path)
        assert summary["n_verdicts_total"] == 2
        assert "teacher" in summary["per_model"]
        teacher = summary["per_model"]["teacher"]
        assert teacher["retrieval"] is not None
        assert teacher["retrieval"]["recall_at_1"] == pytest.approx(1.0)
        assert teacher["retrieval"]["mrr_mean"] == pytest.approx(1.0)
        # Length descriptives populated from cache output_tokens (=100 for teacher).
        assert teacher["length"]["output_tokens"]["mean"] == pytest.approx(100.0)

    def test_render_report_md_smoke(self, tmp_path):
        cache_path = _write_cache(tmp_path)
        raw = tmp_path / "raw.jsonl"
        raw.write_text(
            json.dumps({"sample_id": "sid-A", "model_tag": "teacher", "error": None,
                        "groundedness": 4, "personalization": 4,
                        "groundedness_evidence": "g", "personalization_evidence": "p"})
        )
        summary = analyze(raw, cache_path)
        md = render_report_md(summary)
        assert "Listwise LLM-as-a-Judge" in md
        assert "`teacher`" in md
        assert "Future work" in md

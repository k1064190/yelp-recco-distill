# ABOUTME: Unit tests for scripts.train.train_student_distill -- schema->regex helper,
# ABOUTME: lambda curriculum callback, CLI parsing. No GPU or teacher server required.

"""
Unit tests for the DistillationTrainer-based pipeline.

Covers pieces that do not require a running teacher server:
  - build_teacher_response_regex produces a regex that matches every
    eval-set teacher payload and rejects obvious garbage.
  - LambdaCurriculumCallback interpolates correctly and clamps at warmup_steps.
  - CLI --help returns without importing heavy GPU deps.
  - The server-forced `loss_top_k=1` override fires when the user passes a
    mismatched value.
"""

from __future__ import annotations

import json
import re
import subprocess
import sys
from pathlib import Path
from unittest.mock import MagicMock

try:
    import pytest
except ImportError:  # pragma: no cover -- allows direct execution in the matching environment
    pytest = None

PROJECT_ROOT = Path(__file__).resolve().parents[1]


# ---------- Regex helper ----------


def test_regex_matches_pydantic_payload():
    """The regex must accept an exemplar TeacherResponse JSON instance."""
    from configs.teacher_prompt import Rationale, TeacherResponse
    from scripts.train.train_student_distill import build_teacher_response_regex

    regex = build_teacher_response_regex()
    pattern = re.compile(regex, flags=re.DOTALL)

    payload = TeacherResponse(
        persona="Loves ramen and craft beer on weekends.",
        rationales=[
            Rationale(candidate_index=i, reason=f"fits well because reason {i}")
            for i in range(1, 11)
        ],
        ranking=[1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
    )
    # model_json_schema defines field order; model_dump_json preserves it.
    text = payload.model_dump_json()
    assert pattern.fullmatch(text) is not None, "regex rejected a canonical payload"


def test_regex_rejects_invalid_candidate_index():
    """Regex must reject candidate_index outside 1..10."""
    from scripts.train.train_student_distill import build_teacher_response_regex

    regex = build_teacher_response_regex()
    pattern = re.compile(regex, flags=re.DOTALL)

    # candidate_index=11 is illegal (CandidateIndex = Literal[1..10]).
    bad = {
        "persona": "x",
        "rationales": [
            {"candidate_index": 11, "reason": "oops"}
            for _ in range(10)
        ],
        "ranking": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
    }
    assert pattern.fullmatch(json.dumps(bad)) is None


def test_regex_rejects_wrong_ranking_length():
    """Regex must enforce exactly 10 ranking entries."""
    from scripts.train.train_student_distill import build_teacher_response_regex

    regex = build_teacher_response_regex()
    pattern = re.compile(regex, flags=re.DOTALL)

    bad = {
        "persona": "x",
        "rationales": [
            {"candidate_index": i, "reason": "ok"} for i in range(1, 11)
        ],
        "ranking": [1, 2, 3, 4, 5],  # wrong length
    }
    assert pattern.fullmatch(json.dumps(bad)) is None


# ---------- Lambda curriculum ----------


def test_lambda_curriculum_interpolates():
    """Callback must linearly ramp lmbda from start to end across warmup steps."""
    from scripts.train.train_student_distill import _make_lambda_curriculum_callback_class

    LambdaCB = _make_lambda_curriculum_callback_class()
    trainer = MagicMock()
    trainer.lmbda = 0.0

    cb = LambdaCB(trainer=trainer, start=0.0, end=1.0, warmup_steps=10)

    # Mid-curriculum (step 5 -> frac 0.5 -> lambda 0.5)
    state = MagicMock()
    state.global_step = 5
    cb.on_step_begin(args=None, state=state, control=None)
    assert abs(trainer.lmbda - 0.5) < 1e-6

    # Beyond warmup (step 20 -> clamped to end)
    state.global_step = 20
    cb.on_step_begin(args=None, state=state, control=None)
    assert abs(trainer.lmbda - 1.0) < 1e-6

    # Step 0 -> start
    state.global_step = 0
    cb.on_step_begin(args=None, state=state, control=None)
    assert abs(trainer.lmbda - 0.0) < 1e-6


def test_lambda_curriculum_zero_warmup_uses_end():
    """warmup_steps=0 must set lambda directly to end (no interpolation)."""
    from scripts.train.train_student_distill import _make_lambda_curriculum_callback_class

    LambdaCB = _make_lambda_curriculum_callback_class()
    trainer = MagicMock()
    trainer.lmbda = 0.0

    cb = LambdaCB(trainer=trainer, start=0.0, end=0.7, warmup_steps=0)
    state = MagicMock()
    state.global_step = 0
    cb.on_step_begin(args=None, state=state, control=None)
    assert abs(trainer.lmbda - 0.7) < 1e-6


# ---------- CLI smoke ----------


def test_help_exits_zero_and_lists_key_flags():
    """`python train_student_distill.py --help` must work without GPU init."""
    script = PROJECT_ROOT / "scripts" / "train_student_distill.py"
    result = subprocess.run(
        [sys.executable, str(script), "--help"],
        capture_output=True, text=True, timeout=30,
    )
    assert result.returncode == 0, f"--help failed: {result.stderr}"
    # Spot-check that the plan's headline flags are exposed.
    for flag in (
        "--teacher-url", "--lmbda", "--lmbda-start", "--lmbda-warmup-steps",
        "--beta", "--teacher-top-k", "--no-guided-json",
        "--eval-callback-steps", "--eval-callback-threshold",
    ):
        assert flag in result.stdout, f"missing flag in --help: {flag}"


# ---------- Config contract ----------


def test_distillation_config_rejects_top_k_gt_1_with_server_beta_gt_0():
    """Sanity: the server+beta>0 => top_k=1 constraint actually fires.

    This is the rule the CLI guards against by pinning --teacher-top-k to 1
    when the user passes beta>0 with a server teacher. If trl ever relaxes
    the constraint we want the test to flag it so we can widen the CLI.
    """
    if pytest is not None:
        pytest.importorskip("trl.experimental.distillation")
    from trl.experimental.distillation import DistillationConfig

    raised = False
    try:
        DistillationConfig(
            output_dir="/tmp/distill-test",
            use_teacher_server=True,
            teacher_model_server_url="http://localhost:8200",
            beta=1.0,
            loss_top_k=100,
        )
    except ValueError as exc:
        assert "loss_top_k must be 1" in str(exc), str(exc)
        raised = True
    assert raised, "DistillationConfig should have raised ValueError"

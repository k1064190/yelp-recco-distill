# ABOUTME: Unit tests for scripts.serve.http_teacher_adapter.HTTPTeacherAdapter.
# ABOUTME: Mocks the vLLM /v1/completions endpoint to avoid network dependency.

"""
Tests for HTTPTeacherAdapter (the GKD Stage-2 2-node teacher proxy).

Verifies:
  - Batch payload: single HTTP call for a B-row micro-batch, pad stripped per
    row via attention_mask.
  - Position mapping: HF logits[t] is populated from prompt_logprobs[t + 1].
  - Dense tensor shape: [B, T, V] on the input device, top-K slots filled and
    the remaining V-K slots at fill_logit.
  - Retry behaviour: 5xx transient errors are retried, 4xx is not.
  - Sort invariance: response choices reordered to match prompt order even
    when the server returns them out of sequence.
  - log_softmax stability under fill_logit: softmax mass collapses onto top-K.

All tests stub requests.Session.post — no real HTTP traffic.
"""

from __future__ import annotations

import math
from typing import Any
from unittest.mock import MagicMock

import pytest
import torch

from scripts.serve.http_teacher_adapter import HTTPTeacherAdapter, TeacherHTTPError


# ---------- helpers ----------


def _stub_response(choices: list[dict[str, Any]], status: int = 200) -> MagicMock:
    """Build a MagicMock that quacks like a ``requests.Response``."""
    resp = MagicMock()
    resp.status_code = status
    resp.json.return_value = {"choices": choices}
    resp.text = "ok" if status < 400 else "error"
    return resp


def _plp_entry(top_k: dict[int, float]) -> dict[str, dict[str, Any]]:
    """Build one ``prompt_logprobs`` entry in vLLM's wire format.

    Args:
        top_k (dict[int, float]): token_id -> logprob.
    """
    return {
        str(tok): {"decoded_token": f"<{tok}>", "logprob": lp, "rank": i + 1}
        for i, (tok, lp) in enumerate(top_k.items())
    }


def _make_adapter(**overrides: Any) -> HTTPTeacherAdapter:
    """Adapter under test with a small vocab so tensor shapes stay printable."""
    kwargs = dict(
        base_url="http://dummy:8100/v1",
        model_name="qwen35-teacher",
        vocab_size=20,
        top_k=3,
        fill_logit=-50.0,
        max_retries=3,
        timeout=1.0,
    )
    kwargs.update(overrides)
    return HTTPTeacherAdapter(**kwargs)


# ---------- forward-path tests ----------


def test_forward_shape_and_fill(monkeypatch):
    """Shape is [B, T, V]; non-top-K slots sit at fill_logit."""
    adapter = _make_adapter()
    # One prompt, 3 tokens. Only position 1 has a non-None prompt_logprobs
    # entry (set tok 5 -> -1.0). Position 2 is left null.
    choices = [{
        "index": 0,
        "prompt_logprobs": [
            None,
            _plp_entry({5: -1.0, 7: -2.0, 9: -3.0}),
            None,
        ],
    }]
    mock_post = MagicMock(return_value=_stub_response(choices))
    monkeypatch.setattr(adapter.session, "post", mock_post)

    input_ids = torch.tensor([[100, 200, 300]])
    attention_mask = torch.tensor([[1, 1, 1]])
    out = adapter(input_ids=input_ids, attention_mask=attention_mask)

    assert out.logits.shape == (1, 3, 20)
    # Row 0, position 0 — filled from prompt_logprobs[1]. Token 5 gets -1.0.
    assert out.logits[0, 0, 5].item() == pytest.approx(-1.0)
    assert out.logits[0, 0, 7].item() == pytest.approx(-2.0)
    # Token not in top-K stays at fill_logit.
    assert out.logits[0, 0, 0].item() == pytest.approx(-50.0)
    # Positions that were not populated also stay at fill_logit.
    assert out.logits[0, 1, 5].item() == pytest.approx(-50.0)
    assert out.logits[0, 2, 0].item() == pytest.approx(-50.0)


def test_forward_position_mapping_is_shifted_by_one(monkeypatch):
    """prompt_logprobs[t+1] -> logits[t]: the canonical HF/vLLM shift.

    Token id ``12`` appears with logprob -0.5 at prompt_logprobs[1]. That must
    land at logits[0, 0, 12], NOT logits[0, 1, 12].
    """
    adapter = _make_adapter()
    choices = [{
        "index": 0,
        "prompt_logprobs": [
            None,
            _plp_entry({12: -0.5, 13: -1.0, 14: -1.5}),
            _plp_entry({12: -2.0, 13: -2.5, 14: -3.0}),
        ],
    }]
    monkeypatch.setattr(
        adapter.session, "post",
        MagicMock(return_value=_stub_response(choices)),
    )

    out = adapter(
        input_ids=torch.tensor([[1, 2, 3]]),
        attention_mask=torch.tensor([[1, 1, 1]]),
    )
    assert out.logits[0, 0, 12].item() == pytest.approx(-0.5)
    assert out.logits[0, 1, 12].item() == pytest.approx(-2.0)


def test_batch_payload_strips_padding(monkeypatch):
    """Padded rows are sent to vLLM as variable-length integer lists."""
    adapter = _make_adapter()
    # Two rows: row 0 has 2 real tokens, row 1 has 3. Row 0 padded to T=3.
    input_ids = torch.tensor([[10, 11, 0], [20, 21, 22]])
    attention_mask = torch.tensor([[1, 1, 0], [1, 1, 1]])

    choices = [
        {"index": 0, "prompt_logprobs": [None, _plp_entry({10: -0.1, 11: -0.2, 12: -0.3})]},
        {"index": 1, "prompt_logprobs": [None,
                                         _plp_entry({10: -1.0, 11: -1.1, 12: -1.2}),
                                         _plp_entry({10: -2.0, 11: -2.1, 12: -2.2})]},
    ]
    mock_post = MagicMock(return_value=_stub_response(choices))
    monkeypatch.setattr(adapter.session, "post", mock_post)

    out = adapter(input_ids=input_ids, attention_mask=attention_mask)

    sent_prompt = mock_post.call_args.kwargs["json"]["prompt"]
    assert sent_prompt == [[10, 11], [20, 21, 22]], (
        "expected pad-stripped, variable-length List[List[int]] payload"
    )
    assert mock_post.call_args.kwargs["json"]["prompt_logprobs"] == 3
    assert mock_post.call_args.kwargs["json"]["max_tokens"] == 1
    # Row 0 padded position (t=1 corresponds to prompt_logprobs[2] which doesn't
    # exist for the shorter row) stays at fill_logit.
    assert out.logits[0, 1, 10].item() == pytest.approx(-50.0)
    # Row 1 position 1 is populated from prompt_logprobs[2].
    assert out.logits[1, 1, 10].item() == pytest.approx(-2.0)


def test_single_http_call_per_forward(monkeypatch):
    """One forward() → one POST, regardless of batch size."""
    adapter = _make_adapter()
    choices = [
        {"index": i, "prompt_logprobs": [None, _plp_entry({1: -1.0, 2: -2.0, 3: -3.0})]}
        for i in range(4)
    ]
    mock_post = MagicMock(return_value=_stub_response(choices))
    monkeypatch.setattr(adapter.session, "post", mock_post)

    adapter(
        input_ids=torch.tensor([[1, 2]] * 4),
        attention_mask=torch.ones(4, 2, dtype=torch.long),
    )
    assert mock_post.call_count == 1


def test_response_reordered_by_index(monkeypatch):
    """Server returns choices in arbitrary order; adapter sorts by ``index``."""
    adapter = _make_adapter()
    choices = [
        {"index": 1, "prompt_logprobs": [None, _plp_entry({9: -9.9, 10: -10.0, 11: -11.0})]},
        {"index": 0, "prompt_logprobs": [None, _plp_entry({1: -0.1, 2: -0.2, 3: -0.3})]},
    ]
    monkeypatch.setattr(
        adapter.session, "post",
        MagicMock(return_value=_stub_response(choices)),
    )

    out = adapter(
        input_ids=torch.tensor([[1, 2], [3, 4]]),
        attention_mask=torch.ones(2, 2, dtype=torch.long),
    )
    # Row 0 must get the choice originally labelled index=0.
    assert out.logits[0, 0, 1].item() == pytest.approx(-0.1)
    # Row 1 must get the choice originally labelled index=1.
    assert out.logits[1, 0, 9].item() == pytest.approx(-9.9)


# ---------- retry / error-path tests ----------


def test_retry_on_5xx_then_success(monkeypatch):
    """Transient 5xx retried with backoff, then succeeds."""
    adapter = _make_adapter(max_retries=3)
    success = _stub_response(
        [{"index": 0, "prompt_logprobs": [None, _plp_entry({1: -1.0, 2: -2.0, 3: -3.0})]}],
    )
    side_effects = [_stub_response([], status=503), _stub_response([], status=500), success]
    mock_post = MagicMock(side_effect=side_effects)
    monkeypatch.setattr(adapter.session, "post", mock_post)
    monkeypatch.setattr("scripts.serve.http_teacher_adapter.time.sleep", lambda _s: None)

    out = adapter(
        input_ids=torch.tensor([[1, 2]]),
        attention_mask=torch.ones(1, 2, dtype=torch.long),
    )
    assert mock_post.call_count == 3
    assert out.logits.shape == (1, 2, 20)


def test_4xx_not_retried(monkeypatch):
    """4xx is a client error and raises immediately (no retry)."""
    adapter = _make_adapter(max_retries=5)
    mock_post = MagicMock(return_value=_stub_response([], status=400))
    monkeypatch.setattr(adapter.session, "post", mock_post)

    with pytest.raises(TeacherHTTPError, match="4xx"):
        adapter(
            input_ids=torch.tensor([[1, 2]]),
            attention_mask=torch.ones(1, 2, dtype=torch.long),
        )
    assert mock_post.call_count == 1


def test_exhausted_retries_raises(monkeypatch):
    """Persistent 5xx exhausts retries and raises TeacherHTTPError."""
    adapter = _make_adapter(max_retries=2)
    mock_post = MagicMock(return_value=_stub_response([], status=503))
    monkeypatch.setattr(adapter.session, "post", mock_post)
    monkeypatch.setattr("scripts.serve.http_teacher_adapter.time.sleep", lambda _s: None)

    with pytest.raises(TeacherHTTPError, match="exhausted"):
        adapter(
            input_ids=torch.tensor([[1, 2]]),
            attention_mask=torch.ones(1, 2, dtype=torch.long),
        )
    assert mock_post.call_count == 2


def test_choice_count_mismatch_raises(monkeypatch):
    """If the server returns the wrong number of choices, fail loudly."""
    adapter = _make_adapter()
    bad = _stub_response([{"index": 0, "prompt_logprobs": [None]}])
    monkeypatch.setattr(adapter.session, "post", MagicMock(return_value=bad))

    with pytest.raises(TeacherHTTPError, match="choices"):
        adapter(
            input_ids=torch.tensor([[1, 2], [3, 4]]),
            attention_mask=torch.ones(2, 2, dtype=torch.long),
        )


# ---------- loss-stability sanity ----------


def test_log_softmax_recovers_topk_probability_mass(monkeypatch):
    """Under fill_logit=-50, softmax over reconstructed logits ≈ exp(logprob)
    for top-K positions and ~0 elsewhere — the premise of the approximation.
    """
    adapter = _make_adapter(vocab_size=32, top_k=3, fill_logit=-50.0)
    # Three tokens with log-probabilities that exponentiate to ~[0.5, 0.3, 0.2].
    logp = {5: math.log(0.5), 6: math.log(0.3), 7: math.log(0.2)}
    choices = [{"index": 0, "prompt_logprobs": [None, _plp_entry(logp)]}]
    monkeypatch.setattr(
        adapter.session, "post",
        MagicMock(return_value=_stub_response(choices)),
    )

    out = adapter(
        input_ids=torch.tensor([[1, 2]]),
        attention_mask=torch.ones(1, 2, dtype=torch.long),
    )
    probs = out.logits[0, 0].softmax(dim=-1)
    assert probs[5].item() == pytest.approx(0.5, abs=1e-6)
    assert probs[6].item() == pytest.approx(0.3, abs=1e-6)
    assert probs[7].item() == pytest.approx(0.2, abs=1e-6)
    # Mass on filler slots is negligible.
    filler_mass = probs.sum().item() - probs[[5, 6, 7]].sum().item()
    assert filler_mass < 1e-15


# ---------- miscellaneous ----------


def test_generate_raises_to_forbid_seq_kd():
    """seq_kd=True would call teacher.generate; we make that impossible."""
    adapter = _make_adapter()
    with pytest.raises(NotImplementedError, match="seq_kd=False"):
        adapter.generate()


def test_pickle_excludes_session():
    """Session is not pickled (it holds a lock); buffers survive."""
    import pickle
    adapter = _make_adapter()
    _ = adapter.session  # force lazy init
    blob = pickle.dumps(adapter)
    restored = pickle.loads(blob)
    assert restored._session is None
    assert restored.vocab_size == adapter.vocab_size
    # Lazy-rebuild works.
    assert restored.session is not None

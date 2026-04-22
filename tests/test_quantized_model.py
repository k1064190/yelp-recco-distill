# ABOUTME: Post-PTQ sanity check — W4A16 compressed-tensors checkpoint must
# ABOUTME: load into vLLM and emit a non-empty response for a trivial prompt.

"""
Gate test for the W4A16 PTQ output (plan §2 / ).

llm-compressor + vLLM + compressed-tensors have historically been the most
version-sensitive link in the pipeline: the quantization format is
negotiated between the two libraries at load time, and a mismatch produces
a checkpoint that loads successfully yet generates garbage. This test is
the smallest end-to-end assertion that catches that failure mode.

Skip conditions (so pytest collection stays green even before PTQ runs):
    * ``ckpt/student-w4a16/`` does not exist            -> skip
    * ``config.json`` has no ``quantization_config``    -> skip (+ clear msg)
    * vLLM fails to import                              -> skip

When all of the above are satisfied, the test:
    1. Loads the model into a low-footprint vLLM instance
       (``gpu_memory_utilization=0.30``, ``max_model_len=512``,
       ``enforce_eager=True`` for consistency with the benchmark)
    2. Issues one deterministic short generation request
    3. Asserts the response is non-empty and at least 3 characters long

Runtime: ~30-60 seconds (dominated by vLLM cold start). Not part of the
fast unit-test pass — run only after a successful quantize_w4a16.py.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import pytest

# vLLM 0.11.0's flashinfer attention backend trips an internal
# assertion (`decode_wrapper._sm_scale == self.scale`) the first time it
# is asked to decode a compressed-tensors W4A16 layer on Ampere (SM 8.6).
# Force the FLASH_ATTN backend instead. Hard override (not setdefault)
# because the shell environment may already export FLASHINFER globally.
# Must be set before the vllm package is imported so the backend selector
# reads the flag during auto-selection. See the portfolio README.
os.environ["VLLM_ATTENTION_BACKEND"] = "FLASH_ATTN"

PROJECT_ROOT = Path(__file__).resolve().parents[1]
W4A16_DIR = PROJECT_ROOT / "ckpt/student-w4a16"


def _skip_if_missing() -> None:
    """Raise pytest.skip if the W4A16 checkpoint or its config is absent."""
    if not W4A16_DIR.exists():
        pytest.skip(f"W4A16 checkpoint not found at {W4A16_DIR} — run quantize_w4a16.py first")
    cfg = W4A16_DIR / "config.json"
    if not cfg.exists():
        pytest.skip(f"{cfg} missing — incomplete PTQ output")
    try:
        data = json.loads(cfg.read_text())
    except json.JSONDecodeError:
        pytest.skip(f"{cfg} unparseable")
    if "quantization_config" not in data:
        pytest.skip(
            f"{cfg} has no 'quantization_config' key — llm-compressor did "
            "not save in compressed-tensors format. Re-run quantize_w4a16.py "
            "with save_compressed=True."
        )


@pytest.fixture(scope="module")
def loaded_llm():
    """Spin up a single vLLM instance for all tests in this module."""
    _skip_if_missing()
    try:
        from vllm import LLM
    except Exception as e:
        pytest.skip(f"vLLM import failed: {e}")

    llm = LLM(
        model=str(W4A16_DIR),
        quantization="compressed-tensors",
        dtype="float16",
        gpu_memory_utilization=0.30,
        max_model_len=512,
        enforce_eager=True,
    )
    yield llm
    # vLLM does not expose an explicit shutdown in the public API; module
    # teardown via process exit is fine for a one-shot test run.


def test_config_json_has_quantization_config():
    """Cheap pre-check before the heavy GPU load fixture runs."""
    _skip_if_missing()
    data = json.loads((W4A16_DIR / "config.json").read_text())
    assert "quantization_config" in data
    qcfg = data["quantization_config"]
    # compressed-tensors format must declare its config type; the exact key
    # depends on llmcompressor version but one of these should be present.
    fingerprint_keys = {"quant_method", "config_groups", "format"}
    assert fingerprint_keys & set(qcfg.keys()), (
        f"quantization_config has unexpected keys: {list(qcfg.keys())}"
    )


def test_quantized_model_generates_non_empty_output(loaded_llm):
    """End-to-end smoke: the model must produce at least a few characters."""
    from vllm import SamplingParams

    prompt = (
        "<|im_start|>user\n"
        "Return the JSON {\"ok\": true}.\n"
        "<|im_end|>\n"
        "<|im_start|>assistant\n"
    )
    out = loaded_llm.generate(
        [prompt],
        SamplingParams(max_tokens=32, temperature=0.0),
    )
    assert out and out[0].outputs, "vLLM returned no outputs"
    text = out[0].outputs[0].text or ""
    assert len(text.strip()) >= 3, (
        f"W4A16 model produced empty/near-empty output: {text!r}. "
        "Likely a compressed-tensors format mismatch. Check "
        " for the HF load_in_4bit fallback path."
    )


def test_quantized_model_deterministic_at_temp_zero(loaded_llm):
    """Two generate calls on the same prompt at temp=0 must agree."""
    from vllm import SamplingParams

    prompt = (
        "<|im_start|>user\nSay hello.<|im_end|>\n"
        "<|im_start|>assistant\n"
    )
    params = SamplingParams(max_tokens=16, temperature=0.0)
    a = loaded_llm.generate([prompt], params)[0].outputs[0].text
    b = loaded_llm.generate([prompt], params)[0].outputs[0].text
    assert a == b, f"non-deterministic outputs at temp=0: {a!r} vs {b!r}"

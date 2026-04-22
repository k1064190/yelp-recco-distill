# ABOUTME: Guided-JSON student rollout for legacy TRL GKDTrainer.
# ABOUTME: Subclass that injects an xgrammar LogitsProcessor into model.generate().

"""
Bridge between the legacy `trl.experimental.gkd.GKDTrainer` and xgrammar so
student on-policy rollouts emit schema-valid JSON instead of unconstrained
text.

Why this is needed
------------------
Legacy ``GKDTrainer`` calls ``model.generate()`` directly inside
``generate_on_policy_outputs`` with no hook to inject a ``LogitsProcessor``.
Without a grammar mask, a small student deviating slightly from the SFT
manifold produces broken JSON → parse rate collapses to 0 %, KL signal
becomes uninformative, and downstream eval cannot recover (the collapse pattern).

This module subclasses ``GKDTrainer`` and overrides
``generate_on_policy_outputs`` to pass an xgrammar ``LogitsProcessor``. The
teacher-side scoring is unchanged: the ``HTTPTeacherAdapter`` still sees the
student-generated token sequence and returns top-K logprobs via the running
vLLM serve. Only the student's sampling step is masked.

Instantiate one ``xgr.CompiledGrammar`` once at trainer init; a fresh
``GrammarMatcher`` is created inside each rollout because matchers are
stateful.
"""

from __future__ import annotations

from typing import Any

import torch

try:
    import xgrammar as xgr
    from xgrammar.contrib.hf import LogitsProcessor as XgrammarLogitsProcessor
except ImportError as e:  # pragma: no cover
    raise ImportError(
        "Guided GKD requires xgrammar. Install it in the matching environment (see ENV_VERSION.md) "
        "(xgrammar is not available in the matching environment)."
    ) from e

from transformers import LogitsProcessorList
from trl.experimental.gkd import GKDTrainer


def build_json_schema_grammar(
    tokenizer: Any, schema: dict, vocab_size: int | None = None
) -> xgr.CompiledGrammar:
    """Compile a JSON-schema grammar against the student's tokenizer.

    Args:
        tokenizer: HF tokenizer (fast).
        schema (dict): JSON Schema dict (e.g. TeacherResponse.model_json_schema()).
        vocab_size (int | None): full vocab size (may exceed tokenizer.vocab_size
            due to padding for kernel alignment). If None, uses ``len(tokenizer)``.

    Returns:
        xgr.CompiledGrammar ready to be fed to LogitsProcessor instances.
    """
    if vocab_size is None:
        vocab_size = len(tokenizer)
    tok_info = xgr.TokenizerInfo.from_huggingface(tokenizer, vocab_size=vocab_size)
    compiler = xgr.GrammarCompiler(tok_info)
    return compiler.compile_json_schema(schema)


class GuidedGKDTrainer(GKDTrainer):
    """GKDTrainer with xgrammar JSON-schema mask on student rollouts.

    Usage:
        compiled = build_json_schema_grammar(tokenizer, TeacherResponse.model_json_schema())
        trainer = GuidedGKDTrainer(
            compiled_grammar=compiled,
            model=student, teacher_model=teacher, args=config, ...
        )
    """

    def __init__(self, *args, compiled_grammar: xgr.CompiledGrammar | None = None, **kwargs):
        """Store the compiled grammar; forward the rest to GKDTrainer."""
        super().__init__(*args, **kwargs)
        self._compiled_grammar = compiled_grammar

    def generate_on_policy_outputs(
        self, model, inputs, generation_config, pad_token_id=None
    ):
        """Override: inject a fresh xgrammar LogitsProcessor per call.

        Shadows the base staticmethod as an instance method — Python's
        ``self.method()`` dispatch finds the subclass version first, so the
        TRL trainer's internal call site picks this up transparently.
        """
        logits_processor = None
        if self._compiled_grammar is not None:
            # Matcher is stateful; instantiate per-call.
            xgr_proc = XgrammarLogitsProcessor(self._compiled_grammar)
            logits_processor = LogitsProcessorList([xgr_proc])

        generated_outputs = model.generate(
            input_ids=inputs["prompts"],
            attention_mask=inputs.get("prompt_attention_mask", None),
            generation_config=generation_config,
            return_dict_in_generate=True,
            logits_processor=logits_processor,
        )
        generated_tokens = generated_outputs.sequences
        new_attention_mask = torch.ones_like(generated_tokens)
        new_labels = generated_tokens.clone()
        if pad_token_id is not None:
            new_labels[new_labels == pad_token_id] = -100
            new_attention_mask[generated_tokens == pad_token_id] = 0
        return generated_tokens, new_attention_mask, new_labels

# ABOUTME: HTTPTeacherAdapter — nn.Module wrapper around a vLLM OpenAI-compatible
# ABOUTME: /v1/completions endpoint, exposing .forward() with HF-style dense logits.

"""
HTTPTeacherAdapter — drop-in teacher for TRL GKDTrainer in the 2-node topology.

Why this exists
---------------
Stage 2 of the post-deadline plan moved the Qwen3.5-35B-A3B teacher off the
student's node and onto a dedicated 4090 node running vLLM as an OpenAI-compatible
server (see scripts/serve/serve_teacher_vllm.sh). GKDTrainer, however, calls
``self.teacher_model(input_ids, attention_mask)`` and expects a dense
``[B, T, V]`` logits tensor to feed into ``generalized_jsd_loss``. This adapter
hides the HTTP round-trip behind that interface.

How it works
------------
1. The caller passes a padded ``input_ids: [B, T]`` with ``attention_mask: [B, T]``.
2. We strip pad positions per row via the attention mask, then POST all B
   prompts in a **single** request as ``prompt: List[List[int]]`` to vLLM's
   ``/v1/completions``. This triggers vLLM's continuous batching scheduler so
   the whole micro-batch shares one prefill pass.
3. We request ``prompt_logprobs=K`` (default 50) so vLLM returns the top-K
   log-probabilities at every non-initial prompt position.
4. We rebuild a dense ``[B, T, V]`` tensor on the student's device, initialised
   to a low "fill" logit (default -50.0) and overwriting only the top-K slots
   at each valid position with the returned log-probabilities. Because softmax
   is shift-invariant, log-probabilities can be used directly as logits — the
   softmax recovers the original top-K probability mass within ``exp(fill)``
   numerical slack of zero on the remaining vocabulary slots.

Position semantics
------------------
HF: ``model(input_ids).logits[:, t, :]`` predicts ``input_ids[t+1]`` given
``input_ids[:t+1]``.

vLLM ``prompt_logprobs[t]`` for ``t >= 1`` predicts ``prompt[t]`` given
``prompt[:t]``.

Therefore ``logits[:, t, :]`` is populated from ``prompt_logprobs[t + 1]``.
The last valid-row position ``t = L - 1`` has no corresponding prompt_logprobs
entry (the trainer slices up to ``-1`` anyway, so this position is never read).

seq_kd note
-----------
TRL's GKDTrainer only calls ``teacher.generate()`` when ``seq_kd=True``. We do
not implement ``.generate()`` — train_student_gkd.py leaves ``seq_kd=False``.
If someone enables seq_kd with this adapter, ``generate()`` will raise.

Example
-------
    >>> adapter = HTTPTeacherAdapter(
    ...     base_url="http://10.1.1.48:8100/v1",
    ...     model_name="qwen35-teacher",
    ...     vocab_size=248077,
    ...     top_k=50,
    ... )
    >>> out = adapter(input_ids=torch.tensor([[9707, 1879]]),
    ...               attention_mask=torch.tensor([[1, 1]]))
    >>> out.logits.shape
    torch.Size([1, 2, 248077])
"""

from __future__ import annotations

import logging
import time
from typing import Any

import requests
import torch
import torch.nn as nn
from transformers.modeling_outputs import CausalLMOutputWithPast

log = logging.getLogger(__name__)


class TeacherHTTPError(RuntimeError):
    """Raised when the vLLM teacher endpoint returns a non-retryable error."""


class HTTPTeacherAdapter(nn.Module):
    """Adapter exposing a vLLM ``/v1/completions`` endpoint as an HF-style LM.

    Args:
        base_url (str): vLLM OpenAI base URL, e.g. ``http://10.1.1.48:8100/v1``.
        model_name (str): vLLM ``--served-model-name``. For our setup this is
            ``qwen35-teacher``.
        vocab_size (int): size of the student's output vocabulary. The returned
            logits tensor has last-dim ``vocab_size``. Teacher and student share
            the Qwen3.5 tokenizer so a single value is correct for both.
        top_k (int): number of top log-probabilities to request per position
            (default 50). Must be ``<= --max-logprobs`` on the server.
        fill_logit (float): value used for the ``V - K`` vocabulary slots with
            no returned log-probability (default -50.0). Low enough that
            ``exp(fill_logit)`` is ~1e-22, so it contributes negligibly to
            softmax normalisation at the positions top-K logprobs are ~-2..-10.
        max_retries (int): retry attempts on 5xx / timeout / connection errors.
            Exponential backoff between attempts (default 3).
        timeout (float): per-request timeout in seconds (default 60.0). Prefill
            on a 4×4090 for an 8-prompt × 4096-token micro-batch should complete
            well inside a few seconds; 60 s is a generous ceiling.
        logits_dtype (torch.dtype): dtype of the returned logits tensor
            (default torch.float32). GKDTrainer casts to fp32 internally for
            ``F.log_softmax``; we match to avoid a second cast.
    """

    def __init__(
        self,
        base_url: str,
        model_name: str,
        vocab_size: int,
        top_k: int = 50,
        fill_logit: float = -50.0,
        max_retries: int = 3,
        timeout: float = 60.0,
        logits_dtype: torch.dtype = torch.float32,
    ) -> None:
        super().__init__()
        self.base_url = base_url.rstrip("/")
        self.model_name = model_name
        self.vocab_size = int(vocab_size)
        self.top_k = int(top_k)
        self.fill_logit = float(fill_logit)
        self.max_retries = int(max_retries)
        self.timeout = float(timeout)
        self.logits_dtype = logits_dtype

        # Dummy buffer so .to(device) + accelerator.prepare_model have a concrete
        # device handle to follow. We never read its value — we only consult
        # self._device_anchor.device to place the output tensor.
        self.register_buffer("_device_anchor", torch.zeros(1), persistent=False)

        # HTTP session is lazy + excluded from pickling so accelerate checkpoint
        # hooks cannot choke on it.
        self._session: requests.Session | None = None

        # Placeholder to satisfy any caller that inspects .config. GKDTrainer
        # itself never reads the teacher's config.
        self.config = _AdapterConfig(vocab_size=self.vocab_size)

    # ---- infrastructure ---------------------------------------------------

    def __getstate__(self) -> dict[str, Any]:
        """Exclude the requests.Session from pickling (it contains a lock)."""
        state = self.__dict__.copy()
        state["_session"] = None
        return state

    @property
    def session(self) -> requests.Session:
        if self._session is None:
            s = requests.Session()
            # Both adapters tuned for the expected 1 outstanding request at a
            # time — GKDTrainer calls teacher.forward once per training step.
            adapter = requests.adapters.HTTPAdapter(
                pool_connections=2, pool_maxsize=4, max_retries=0,
            )
            s.mount("http://", adapter)
            s.mount("https://", adapter)
            self._session = s
        return self._session

    # ---- intentionally disabled API ---------------------------------------

    def generate(self, *args: Any, **kwargs: Any) -> None:
        """Not implemented. Use ``seq_kd=False`` in GKDConfig."""
        raise NotImplementedError(
            "HTTPTeacherAdapter does not support .generate(). Set seq_kd=False "
            "on GKDConfig — only the student generates on-policy sequences."
        )

    def gradient_checkpointing_enable(self, *args: Any, **kwargs: Any) -> None:
        """GKDTrainer turns this on for teacher too; HTTP teacher ignores it."""
        return None

    def gradient_checkpointing_disable(self, *args: Any, **kwargs: Any) -> None:
        return None

    # ---- core -------------------------------------------------------------

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        **_ignored: Any,
    ) -> CausalLMOutputWithPast:
        """Produce teacher logits over ``input_ids`` via one HTTP call.

        Args:
            input_ids (torch.Tensor): int64 tensor of shape ``[B, T]`` with
                pad positions already inserted by the collator.
            attention_mask (torch.Tensor | None): ``[B, T]`` binary mask, 1 for
                real tokens, 0 for padding. Required in practice — without it
                we would send pad tokens to the teacher and waste prefill.
            **_ignored: absorbs ``position_ids``, ``use_cache``, etc. from HF
                call conventions.

        Returns:
            CausalLMOutputWithPast: ``.logits`` is ``[B, T, V]`` on the input
            device with top-K slots filled from the vLLM response and the
            remaining ``V - K`` slots at ``fill_logit``.
        """
        if input_ids.dim() != 2:
            raise ValueError(f"expected input_ids [B, T], got {tuple(input_ids.shape)}")
        B, T = input_ids.shape
        device = input_ids.device

        # Strip pad per row so vLLM scheduling packs only real tokens.
        valid_lens, prompts = self._unpad(input_ids, attention_mask)

        # One HTTP round-trip for the whole micro-batch.
        choices = self._request_prompt_logprobs(prompts)

        # Rebuild dense logits tensor.
        logits = torch.full(
            (B, T, self.vocab_size),
            self.fill_logit,
            dtype=self.logits_dtype,
            device=device,
        )

        for b in range(B):
            entry = choices[b]
            prompt_logprobs = entry["prompt_logprobs"]
            L = valid_lens[b]
            if prompt_logprobs is None or len(prompt_logprobs) != L:
                raise TeacherHTTPError(
                    f"row {b}: expected prompt_logprobs length {L}, got "
                    f"{None if prompt_logprobs is None else len(prompt_logprobs)}"
                )

            # HF logits[t] is populated from prompt_logprobs[t + 1].
            # prompt_logprobs[0] is always None (no prior context).
            for t_target in range(L - 1):
                plp = prompt_logprobs[t_target + 1]
                if plp is None:
                    continue
                for tok_id_str, info in plp.items():
                    tok_id = int(tok_id_str)
                    if 0 <= tok_id < self.vocab_size:
                        logits[b, t_target, tok_id] = float(info["logprob"])
                    # Silently drop any id outside the student's vocabulary;
                    # teacher-tokenizer ids beyond student vocab are unreachable
                    # by the student anyway.

        return CausalLMOutputWithPast(logits=logits)

    # ---- helpers ----------------------------------------------------------

    def _unpad(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor | None,
    ) -> tuple[list[int], list[list[int]]]:
        """Strip pad tokens per row using ``attention_mask``.

        Args:
            input_ids (torch.Tensor): ``[B, T]`` int tensor.
            attention_mask (torch.Tensor | None): ``[B, T]`` binary mask or
                ``None`` (treat all positions as valid).

        Returns:
            tuple: (valid_lens, prompts)
                - valid_lens (list[int]): length ``B``, real token count per row.
                - prompts (list[list[int]]): length ``B``, per-row token-id
                  lists truncated to their valid length, safe to JSON-serialise.
        """
        B, T = input_ids.shape
        if attention_mask is None:
            valid_lens = [T] * B
        else:
            valid_lens = attention_mask.sum(dim=1).to(torch.int64).tolist()

        # .tolist() once and slice in Python is cheaper than B calls for short T.
        rows = input_ids.to("cpu").tolist()
        prompts = [rows[b][: valid_lens[b]] for b in range(B)]
        return valid_lens, prompts

    def _request_prompt_logprobs(
        self,
        prompts: list[list[int]],
    ) -> list[dict[str, Any]]:
        """POST to ``/v1/completions`` with batched prompt_logprobs, with retry.

        Args:
            prompts (list[list[int]]): per-row token-id lists. vLLM returns one
                ``choice`` per entry, ordered by the ``index`` field.

        Returns:
            list[dict]: choices sorted by ``index``, each with a
            ``prompt_logprobs`` field (length equal to that row's token count,
            first entry ``None``).

        Raises:
            TeacherHTTPError: on non-retryable 4xx, schema mismatch, or
            persistent 5xx / timeout after ``max_retries``.
        """
        url = f"{self.base_url}/completions"
        payload = {
            "model": self.model_name,
            "prompt": prompts,
            "max_tokens": 1,
            "temperature": 0.0,
            "prompt_logprobs": self.top_k,
            # We discard the generated token; logprobs=0 keeps payload minimal.
            "logprobs": 0,
        }

        last_err: Exception | None = None
        for attempt in range(1, self.max_retries + 1):
            try:
                resp = self.session.post(url, json=payload, timeout=self.timeout)
            except (requests.ConnectionError, requests.Timeout) as exc:
                last_err = exc
                log.warning("teacher HTTP attempt %d/%d failed: %s",
                            attempt, self.max_retries, exc)
                self._backoff(attempt)
                continue

            if resp.status_code >= 500:
                last_err = TeacherHTTPError(
                    f"teacher 5xx: status={resp.status_code} body={resp.text[:200]}"
                )
                log.warning("teacher HTTP attempt %d/%d 5xx: %s",
                            attempt, self.max_retries, resp.status_code)
                self._backoff(attempt)
                continue

            if resp.status_code >= 400:
                raise TeacherHTTPError(
                    f"teacher 4xx (not retried): status={resp.status_code} "
                    f"body={resp.text[:500]}"
                )

            data = resp.json()
            choices = data.get("choices")
            if not choices or len(choices) != len(prompts):
                raise TeacherHTTPError(
                    f"teacher response has {len(choices) if choices else 0} "
                    f"choices for {len(prompts)} prompts"
                )
            # Ensure the returned choices are in the same order as prompts.
            choices = sorted(choices, key=lambda c: int(c.get("index", 0)))
            return choices

        raise TeacherHTTPError(
            f"teacher HTTP exhausted {self.max_retries} retries; last error: {last_err}"
        )

    @staticmethod
    def _backoff(attempt: int) -> None:
        """Exponential backoff: 0.5 s, 1 s, 2 s, ..."""
        time.sleep(0.5 * (2 ** (attempt - 1)))


class _AdapterConfig:
    """Minimal stand-in for a HF PretrainedConfig.

    GKDTrainer itself does not read teacher.config, but accelerate's prepare
    occasionally inspects ``.is_encoder_decoder`` and similar. We expose the
    fields that have been observed to matter in practice; others raise
    AttributeError, which mirrors HF behaviour for unknown config keys.
    """

    def __init__(self, vocab_size: int) -> None:
        self.vocab_size = int(vocab_size)
        self.is_encoder_decoder = False
        self.model_type = "http_teacher_adapter"
        self.use_cache = False

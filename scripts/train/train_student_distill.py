#!/usr/bin/env python
# ABOUTME: Guided-JSON on-policy distillation for Qwen3.5 student via TRL
# ABOUTME: DistillationTrainer with external vLLM teacher server and λ curriculum.

"""
Guided JSON GKD via DistillationTrainer (TRL 1.1.0 experimental).

Student on-policy generation runs through a colocated vLLM engine with a
regex-constrained output (derived from TeacherResponse schema) so every
student sample is syntactically valid JSON. Teacher logprobs are fetched
from an external ``trl vllm-serve`` instance over HTTP via ``VLLMClient``.

This replaces three hand-rolled pieces from the previous GKDTrainer pipeline:

    1. ``HTTPTeacherAdapter``      → ``DistillationConfig.use_teacher_server``
    2. fill_logit=-50 sparse hack  → ``loss_top_k`` / ``loss_add_tail``
    3. ``GuidedGKDTrainer`` patch  → ``vllm_structured_outputs_regex``

Key design deviations from the original plan (`fuzzy-juggling-dewdrop.md`),
imposed by the DistillationTrainer API contract:

    * ``loss_top_k == 1`` is enforced when ``use_teacher_server=True`` and
      ``beta > 0``. Reverse-KL with the sampled completion token is the only
      supported server path (distillation_config.py:431). We keep ``beta=1.0``
      (reverse KL, deep-research recommendation) and collapse the plan's
      ``loss_top_k=100`` to 1.
    * The teacher must be served with ``trl vllm-serve`` (custom endpoints
      ``/generate/``, ``/get_sequence_logprobs/``, ``/health/``). The
      OpenAI-compatible ``vllm.entrypoints.openai.api_server`` used by
      ``scripts/serve/serve_teacher_vllm.sh`` is incompatible with ``VLLMClient``.
    * Teacher URL uses the server root (no ``/v1`` suffix).

Hyperparameter guide:
    lmbda (λ): on-policy fraction (0=off-policy, 1=on-policy). Default 0.5
        with curriculum warmup 0→target over `--lmbda-warmup-steps` steps.
    beta (β): 0 forward KL, 1 reverse KL (server-backed default). 0.5 JSD
        (local teacher only).
    loss_top_k: 1 when use_teacher_server + beta>0 (forced).

Requires the matching environment (see ENV_VERSION.md) (trl 1.1.0, outlines-core 0.2.14).

Example smoke test (teacher serve on GPU 1, student on GPU 2):
    $ DEVICES=1 bash scripts/serve/launch_teacher_trl_serve.sh
    $ CUDA_VISIBLE_DEVICES=2 \\
        python \\
        scripts/train/train_student_distill.py \\
          --student-model ckpt/student-v3-sft-merged \\
          --teacher-url http://localhost:8200 \\
          --lmbda 0.5 --lmbda-start 0.0 --lmbda-warmup-steps 5 \\
          --epochs 1 --max-train-samples 10 \\
          --eval-callback-steps 5 --no-wandb

Full run:
    $ CUDA_VISIBLE_DEVICES=2 \\
        python \\
        scripts/train/train_student_distill.py \\
          --student-model ckpt/student-v3-sft-merged \\
          --teacher-url http://localhost:8200 \\
          --lmbda 0.5 --lmbda-start 0.0 --lmbda-warmup-steps 200 \\
          --beta 1.0 --learning-rate 5e-5 --epochs 1 \\
          --eval-callback-steps 100 --eval-callback-threshold 0.6 \\
          --output ckpt/student-v3-distill \\
          --merged-output ckpt/student-v3-distill-merged \\
          --wandb-run-name v3-distill-guided-lmbda0.5-reverseKL
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from pathlib import Path

# Silence TRL experimental warning (must precede any trl import)
os.environ.setdefault("TRL_EXPERIMENTAL_SILENCE", "1")

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# Reuse data + callback infrastructure from the GKD script to avoid
# duplicating 300+ lines of identical plumbing. The helpers imported here
# have no TRL dependency at module load.
from scripts.train.train_student_gkd import (  # noqa: E402
    _make_parse_rate_callback_class,
    build_gkd_dataset,
    load_and_filter,
    split_examples,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("train_distill")


# ---------- Schema → regex (vLLM structured output) ----------


def build_teacher_response_regex() -> str:
    """Build a regex matching the TeacherResponse JSON schema.

    DistillationConfig.vllm_structured_outputs_regex expects a regex string.
    We generate it from the canonical Pydantic model so the constraint stays
    in sync with the rest of the pipeline (teacher prompt, validator, etc.).

    The chat template for Qwen3.5 ends the assistant prompt with ``\\n\\n``
    after a hidden empty ``<think>`` block. Without allowing leading and
    trailing whitespace, vLLM's xgrammar backend rejects the very first
    token the student samples (typically a newline or space) -- observed as
    "grammar rejected tokens [0]" + on-policy completion length = 1.

    Returns:
        str: regex string that matches any valid TeacherResponse JSON with
            optional surrounding whitespace.
    """
    from outlines_core.json_schema import build_regex_from_schema

    from configs.teacher_prompt import TeacherResponse

    schema_dict = TeacherResponse.model_json_schema()
    body = build_regex_from_schema(json.dumps(schema_dict))
    return r"\s*" + body + r"\s*"


# ---------- λ curriculum callback ----------


def _make_lambda_curriculum_callback_class():
    """Factory for LambdaCurriculumCallback (deferred transformers import).

    Returns:
        type: LambdaCurriculumCallback class inheriting from TrainerCallback.
    """
    from transformers import TrainerCallback

    class LambdaCurriculumCallback(TrainerCallback):
        """Linearly ramp the trainer's on-policy fraction ``lmbda`` over
        ``warmup_steps`` training steps, from ``start`` to ``end``.

        Rationale: a student that has never seen its own outputs starts with
        broken JSON prefixes. Forcing λ=0.5 from step 0 feeds those broken
        prefixes to the teacher, which returns garbage logprobs (the collapse pattern).
        Starting at λ=0 (pure off-policy) and ramping gives the student time
        to internalize teacher-style completions before it has to generate
        its own.

        Args:
            trainer: the DistillationTrainer instance whose ``self.lmbda`` is
                mutated in place (the inner loop reads this attribute each
                step, not args.lmbda).
            start (float): initial λ value at step 0.
            end (float): target λ value at step ``warmup_steps``.
            warmup_steps (int): step count over which λ linearly interpolates.
                0 means "set to end immediately" (no curriculum).
        """

        def __init__(self, trainer, start: float, end: float, warmup_steps: int):
            super().__init__()
            self._trainer = trainer
            self._start = float(start)
            self._end = float(end)
            self._warmup = int(warmup_steps)

        def on_step_begin(self, args, state, control, **kwargs):
            if self._warmup <= 0:
                self._trainer.lmbda = self._end
                return
            frac = min(1.0, state.global_step / self._warmup)
            self._trainer.lmbda = self._start + frac * (self._end - self._start)

    return LambdaCurriculumCallback


# ---------- CLI ----------


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    # Data
    p.add_argument(
        "--samples", type=Path,
        default=PROJECT_ROOT / "data/processed/philly_samples.jsonl",
        help="processed samples JSONL",
    )
    p.add_argument(
        "--teacher-data", type=Path,
        default=PROJECT_ROOT / "data/teacher/philly_teacher_qwen35_http_identity.jsonl",
        help=(
            "teacher output JSONL used as off-policy targets (default: "
            "the HTTP-identity dataset used by v3-sft)."
        ),
    )

    # Student
    p.add_argument(
        "--student-model", type=str,
        default=str(PROJECT_ROOT / "ckpt/student-v3-sft-merged"),
        help="HF model id or path for the student (trainee)",
    )
    p.add_argument(
        "--attn-impl", type=str, default="sdpa",
        choices=["sdpa", "flash_attention_2", "eager"],
        help=(
            "student attention impl for the HF training forward pass. "
            "Colocated vLLM uses its own attention backend and is unaffected "
            "by this flag."
        ),
    )

    # Teacher (external vLLM server via trl vllm-serve)
    p.add_argument(
        "--teacher-url", type=str,
        default=os.environ.get("TEACHER_BASE_URL", "http://localhost:8200"),
        help=(
            "base URL of the teacher vLLM server. Must be served with "
            "`trl vllm-serve` (custom endpoints). NO /v1 suffix."
        ),
    )

    # Distillation hyperparameters
    p.add_argument(
        "--lmbda", type=float, default=0.5,
        help="target on-policy fraction (0=off-policy, 1=on-policy). Default 0.5.",
    )
    p.add_argument(
        "--lmbda-start", type=float, default=0.0,
        help="curriculum starting λ (default 0.0, pure off-policy warmup).",
    )
    p.add_argument(
        "--lmbda-warmup-steps", type=int, default=200,
        help="steps to linearly ramp λ from --lmbda-start to --lmbda (default 200).",
    )
    p.add_argument(
        "--beta", type=float, default=1.0,
        help=(
            "divergence interpolation. 0=forward KL, 0.5=JSD, 1=reverse KL "
            "(default). When use_teacher_server=True and beta>0, "
            "DistillationConfig forces loss_top_k=1."
        ),
    )
    p.add_argument(
        "--temperature", type=float, default=1.0,
        help="softmax temperature for loss and on-policy sampling (default 1.0).",
    )
    p.add_argument(
        "--teacher-top-k", type=int, default=1,
        help=(
            "loss_top_k passed to DistillationConfig. With beta>0 + teacher "
            "server, this is pinned to 1 (trl server-side contract)."
        ),
    )

    # Training
    p.add_argument(
        "--gradient-checkpointing", action="store_true", default=True,
        help="enable gradient checkpointing (default True; saves VRAM at cost of compute).",
    )
    p.add_argument(
        "--no-gradient-checkpointing", dest="gradient_checkpointing",
        action="store_false",
        help="disable gradient checkpointing (useful for diagnosing NaN in backward).",
    )
    p.add_argument("--epochs", type=int, default=1)
    p.add_argument("--learning-rate", type=float, default=5e-5)
    p.add_argument("--warmup-steps", type=int, default=20)
    p.add_argument("--weight-decay", type=float, default=0.01)
    p.add_argument("--max-grad-norm", type=float, default=1.0)
    p.add_argument("--batch-size", type=int, default=1)
    p.add_argument("--grad-accum", type=int, default=2)
    p.add_argument(
        "--max-length", type=int, default=4096,
        help="max total (prompt + completion) tokens for training.",
    )
    p.add_argument(
        "--max-completion-length", type=int, default=1024,
        help="max new tokens to generate on-policy per step.",
    )
    p.add_argument(
        "--eval-ratio", type=float, default=0.9,
        help="train fraction for deterministic sample_id split.",
    )
    p.add_argument(
        "--max-train-samples", type=int, default=None,
        help="cap training set (smoke tests).",
    )

    # vLLM student generation
    p.add_argument(
        "--vllm-mode", type=str, default="server",
        choices=["colocate", "server"],
        help=(
            "'server' (default) uses an external `trl vllm-serve` student "
            "process (see scripts/serve/launch_student_trl_serve.sh). 'colocate' "
            "loads the student vLLM in the same process as the HF trainer; "
            "on vLLM 0.19 + Blackwell this produces all-NaN logits."
        ),
    )
    p.add_argument(
        "--vllm-server-base-url", type=str,
        default=os.environ.get("STUDENT_SERVER_URL", "http://localhost:8300"),
        help=(
            "student vLLM server URL when --vllm-mode=server. Must be a "
            "`trl vllm-serve` endpoint (TRL custom routes)."
        ),
    )
    p.add_argument(
        "--vllm-gpu-mem-util", type=float, default=0.35,
        help=(
            "GPU memory fraction reserved for the colocated vLLM engine. "
            "Only used when --vllm-mode=colocate."
        ),
    )
    p.add_argument(
        "--vllm-max-model-len", type=int, default=4096,
        help="max sequence length for the colocated vLLM engine.",
    )
    p.add_argument(
        "--no-guided-json", action="store_true",
        help=(
            "disable vllm_structured_outputs_regex (debug only — triggers "
            "the structured-output collapse pathology)."
        ),
    )
    p.add_argument(
        "--structured-backend", type=str, default="outlines",
        choices=["auto", "xgrammar", "outlines", "lm-format-enforcer", "guidance"],
        help=(
            "vLLM structured-outputs backend. Default 'outlines' because "
            "'auto' picks xgrammar on vLLM 0.19, which fails to apply the "
            "grammar bitmask in colocate mode under TRL 1.1.0 (TRL was "
            "tested against vLLM 0.11-0.17). Set to 'auto' to restore "
            "default behaviour."
        ),
    )
    p.add_argument(
        "--vllm-enforce-eager", action="store_true", default=True,
        help=(
            "force enforce_eager=True on the colocated student vLLM engine. "
            "Default True: CUDA graph capture produces all-NaN logits on "
            "first forward for Qwen3.5-0.8B on Blackwell (SM 12.0). "
            "Eager execution skips graph capture."
        ),
    )
    p.add_argument(
        "--no-vllm-enforce-eager", dest="vllm_enforce_eager",
        action="store_false",
        help="allow CUDA graph capture on the colocated student vLLM engine.",
    )
    p.add_argument(
        "--vllm-drop-external-launcher", action="store_true",
        help=(
            "drop `distributed_executor_backend=\"external_launcher\"` from "
            "the colocated LLM constructor. TRL 1.1.0 hardcodes this, but "
            "standalone vLLM 0.19 works fine without it. Dropping lets vLLM "
            "pick its default backend (mp) which may avoid the all-NaN "
            "logits observed on the colocated student forward pass."
        ),
    )
    p.add_argument(
        "--student-vlm-shell", action="store_true",
        help=(
            "student vLLM serve loads a VLM-rehydrated ckpt "
            "(Qwen3_5ForConditionalGeneration; e.g. "
            "ckpt/student-v3-sft-merged_vllm_vlm) whose parameter names are "
            "namespaced under `language_model.*` / `visual.*`. Training-process "
            "student is text-only (Qwen3_5ForCausalLM; `model.*` / "
            "`lm_head.*`). Enable this flag to rekey training params "
            "(`model.X` -> `language_model.model.X`, `lm_head.X` -> "
            "`language_model.lm_head.X`) during sync_weights so the serve-side "
            "load_weights() hits the right slots. vLLM 0.19 "
            "text-only Qwen3.5 path is skeletal, so we keep VLM shell on the "
            "serve side and bridge the name gap here."
        ),
    )

    # Output
    p.add_argument(
        "--output", type=Path,
        default=PROJECT_ROOT / "ckpt/student-v3-distill",
        help="output directory for checkpoints.",
    )
    p.add_argument(
        "--merged-output", type=Path,
        default=PROJECT_ROOT / "ckpt/student-v3-distill-merged",
        help="where to save the final merged full model.",
    )

    # Logging
    p.add_argument("--wandb-project", type=str, default="llm-distillation-yelp-distill")
    p.add_argument("--wandb-run-name", type=str, default=None)
    p.add_argument("--no-wandb", action="store_true", help="disable wandb logging")
    p.add_argument("--logging-steps", type=int, default=5)

    # Eval callback (prevention)
    p.add_argument(
        "--eval-callback-steps", type=int, default=0,
        help=(
            "run a parse-rate sanity check every N steps. 0 disables. "
            "Recommended: 100 for long runs, 5 for smoke tests."
        ),
    )
    p.add_argument(
        "--eval-callback-samples", type=int, default=5,
        help="eval samples generated per callback check.",
    )
    p.add_argument(
        "--eval-callback-threshold", type=float, default=0.6,
        help=(
            "minimum parse rate to continue training. Lower than the GKD "
            "callback default (0.8) because guided decoding is constrained "
            "already; the callback here checks *unconstrained* generation "
            "to detect representation drift away from JSON."
        ),
    )

    return p.parse_args()


def _patch_vllm_llm_init(backend: str, enforce_eager: bool, drop_external_launcher: bool = False) -> None:
    """Wrap vllm.LLM.__init__ to inject fields TRL's VLLMGeneration doesn't expose.

    Two injections:

    1. ``structured_outputs_config``: vLLM's default "auto" picks xgrammar,
       which on vLLM 0.19 + TRL 1.1.0 mis-applies the grammar bitmask in
       colocate + external_launcher mode. Forcing ``outlines`` uses a
       different codepath that applies the mask correctly.

    2. ``enforce_eager``: TRL's LLM() call lets vLLM capture CUDA graphs by
       default. Under our probe we observed all-NaN logits on the first
       student forward pass -- a known class of failure for small models
       when CUDA graph capture mode mismatches runtime batch shape. Forcing
       eager execution (as our teacher launcher already does) skips graph
       capture and produces finite logits.

    Args:
        backend (str): structured-output backend name.
        enforce_eager (bool): if True, disable CUDA graph capture.
    """
    import vllm
    from vllm.config import StructuredOutputsConfig

    if getattr(vllm.LLM, "_distill_patched", False):
        return
    original_init = vllm.LLM.__init__

    def patched_init(self, *init_args, **init_kwargs):
        if "structured_outputs_config" not in init_kwargs:
            init_kwargs["structured_outputs_config"] = StructuredOutputsConfig(
                backend=backend,
            )
        if enforce_eager and "enforce_eager" not in init_kwargs:
            init_kwargs["enforce_eager"] = True
        if drop_external_launcher and init_kwargs.get("distributed_executor_backend") == "external_launcher":
            log.info(
                "patched vllm.LLM: dropping distributed_executor_backend=external_launcher"
            )
            init_kwargs.pop("distributed_executor_backend", None)
        return original_init(self, *init_args, **init_kwargs)

    vllm.LLM.__init__ = patched_init
    vllm.LLM._distill_patched = True
    log.info(
        "patched vllm.LLM: structured_outputs backend=%s, enforce_eager=%s",
        backend, enforce_eager,
    )


def _patch_vllm_generation_for_vlm_shell() -> None:
    """Rekey training-process params before sync_weights pushes them to a VLM-shell student vLLM serve.

    vLLM 0.19's text-only Qwen3.5 path is skeletal, so we
    launch student inference with a VLM-rehydrated ckpt
    (`Qwen3_5ForConditionalGeneration`) whose parameters live under
    `language_model.*` / `visual.*`. Training keeps text-only
    `Qwen3_5ForCausalLM` whose keys are `model.*` / `lm_head.*`.

    `_fix_param_name_to_vllm` is the single hook every sync path (non-FSDP,
    FSDP1, FSDP2, PEFT) routes through before calling `update_named_param`
    (see trl/generation/vllm_generation.py:369,394,420,507). Prepend
    `language_model.` there so the serve's load_weights() hits the right
    slots. Visual-side params on the serve stay frozen because training
    process does not hold them.
    """
    from trl.generation.vllm_generation import VLLMGeneration

    if getattr(VLLMGeneration, "_vlm_shell_patched", False):
        return
    original = VLLMGeneration._fix_param_name_to_vllm

    def patched(self, name, extra_prefixes=None):
        name = original(self, name, extra_prefixes)
        if not name.startswith("language_model.") and (
            name.startswith("model.") or name.startswith("lm_head.")
        ):
            name = "language_model." + name
        return name

    VLLMGeneration._fix_param_name_to_vllm = patched
    VLLMGeneration._vlm_shell_patched = True
    log.info(
        "patched VLLMGeneration._fix_param_name_to_vllm: prepend "
        "'language_model.' for VLM-shell serve (bridge)"
    )


def main() -> int:
    args = parse_args()

    # Sanity: pin loss_top_k to 1 when the server-side contract requires it.
    if args.beta > 0 and args.teacher_top_k != 1:
        log.warning(
            "beta=%.2f>0 with use_teacher_server=True forces loss_top_k=1; "
            "overriding --teacher-top-k=%d.",
            args.beta, args.teacher_top_k,
        )
        args.teacher_top_k = 1

    # Heavy imports deferred so --help stays fast.
    import torch  # noqa: F401
    from datasets import Dataset
    from transformers import AutoTokenizer
    from trl.experimental.distillation import DistillationConfig, DistillationTrainer

    # Force structured-outputs backend + eager mode before any vLLM LLM()
    # instantiation inside DistillationTrainer. enforce_eager avoids a
    # CUDA-graph-capture pathology that produces all-NaN logits on first
    # forward for small student models on Blackwell. Backend override is
    # no-op if --no-guided-json or args.structured_backend == "auto".
    backend = None if args.no_guided_json or args.structured_backend == "auto" else args.structured_backend
    if backend is not None or args.vllm_enforce_eager or args.vllm_drop_external_launcher:
        _patch_vllm_llm_init(
            backend=backend or "auto",
            enforce_eager=args.vllm_enforce_eager,
            drop_external_launcher=args.vllm_drop_external_launcher,
        )

    if args.student_vlm_shell:
        _patch_vllm_generation_for_vlm_shell()

    # ---- 1. Load + split --------------------------------------------------
    examples, _stats = load_and_filter(args.samples, args.teacher_data)
    if not examples:
        log.error("no training examples after filter; aborting")
        return 2

    train_exs, eval_exs = split_examples(examples, ratio=args.eval_ratio)
    if args.max_train_samples is not None:
        train_exs = train_exs[: args.max_train_samples]
    log.info("split: train=%d, eval=%d", len(train_exs), len(eval_exs))

    # ---- 2. Build messages-format dataset ---------------------------------
    train_data = build_gkd_dataset(train_exs)
    eval_data = build_gkd_dataset(eval_exs)
    train_ds = Dataset.from_list(train_data)
    eval_ds = Dataset.from_list(eval_data) if eval_data else None

    # ---- 3. Tokenizer -----------------------------------------------------
    log.info("loading tokenizer from %s", args.student_model)
    tokenizer = AutoTokenizer.from_pretrained(
        args.student_model, use_fast=True, trust_remote_code=True,
    )
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    # ---- 4. Regex constraint ---------------------------------------------
    if args.no_guided_json:
        log.warning("guided JSON DISABLED (debug mode — expect structured-output collapse)")
        regex = None
    else:
        regex = build_teacher_response_regex()
        log.info("built TeacherResponse regex (%d chars)", len(regex))

    # ---- 5. Config --------------------------------------------------------
    run_name = args.wandb_run_name or (
        f"distill-lmbda{args.lmbda}-beta{args.beta}-{args.epochs}ep"
    )
    if args.no_wandb:
        log.info("wandb: DISABLED")
        report_to = "none"
    else:
        os.environ.setdefault("WANDB_PROJECT", args.wandb_project)
        report_to = "wandb"
        log.info("wandb: ENABLED project=%s run_name=%s", args.wandb_project, run_name)

    config = DistillationConfig(
        output_dir=str(args.output),
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        learning_rate=args.learning_rate,
        lr_scheduler_type="cosine",
        warmup_steps=args.warmup_steps,
        weight_decay=args.weight_decay,
        max_grad_norm=args.max_grad_norm,
        bf16=True,
        fp16=False,
        gradient_checkpointing=args.gradient_checkpointing,
        gradient_checkpointing_kwargs={"use_reentrant": False} if args.gradient_checkpointing else None,
        max_length=args.max_length,
        max_completion_length=args.max_completion_length,
        logging_steps=args.logging_steps,
        save_strategy="epoch",
        save_total_limit=2,
        eval_strategy="epoch" if eval_ds is not None else "no",
        report_to=report_to,
        run_name=run_name,
        dataloader_num_workers=0,
        disable_dropout=True,

        # External teacher via trl vllm-serve
        use_teacher_server=True,
        teacher_model_server_url=args.teacher_url,

        # Distillation core
        lmbda=args.lmbda,  # curriculum callback overrides this each step
        beta=args.beta,
        temperature=args.temperature,
        loss_top_k=args.teacher_top_k,
        loss_add_tail=True,
        reverse_kl_top_1_mode="sampled",  # only supported mode with server

        # vLLM student generation -- disabled when λ target + start are both 0
        # (pure off-policy) to save the colocate engine's GPU fraction. No need
        # to load a student vLLM instance if we never generate on-policy.
        use_vllm=(args.lmbda > 0 or args.lmbda_start > 0),
        vllm_mode=args.vllm_mode,
        vllm_server_base_url=(
            args.vllm_server_base_url if args.vllm_mode == "server" else None
        ),
        vllm_gpu_memory_utilization=args.vllm_gpu_mem_util,
        vllm_max_model_length=args.vllm_max_model_len,
        vllm_structured_outputs_regex=regex,
    )

    # ---- 6. Trainer -------------------------------------------------------
    log.info(
        "building DistillationTrainer: teacher=%s, lambda(target)=%.2f, "
        "beta=%.2f, top_k=%d, guided=%s",
        args.teacher_url, args.lmbda, args.beta, args.teacher_top_k,
        regex is not None,
    )
    trainer = DistillationTrainer(
        model=args.student_model,
        args=config,
        train_dataset=train_ds,
        eval_dataset=eval_ds,
        processing_class=tokenizer,
    )

    # ---- 7. Callbacks -----------------------------------------------------
    LambdaCB = _make_lambda_curriculum_callback_class()
    lambda_cb = LambdaCB(
        trainer=trainer,
        start=args.lmbda_start,
        end=args.lmbda,
        warmup_steps=args.lmbda_warmup_steps,
    )
    trainer.add_callback(lambda_cb)
    log.info(
        "lambda curriculum: %.2f -> %.2f over %d steps",
        args.lmbda_start, args.lmbda, args.lmbda_warmup_steps,
    )

    if args.eval_callback_steps > 0:
        ParseRateCB = _make_parse_rate_callback_class()
        parse_cb = ParseRateCB(
            eval_examples=eval_exs,
            tokenizer=tokenizer,
            check_steps=args.eval_callback_steps,
            n_samples=args.eval_callback_samples,
            threshold=args.eval_callback_threshold,
            log_wandb_sample=not args.no_wandb,
        )
        trainer.add_callback(parse_cb)
        log.info(
            "parse-rate callback: every %d steps, %d samples, threshold=%.0f%%, "
            "wandb_sample=%s",
            args.eval_callback_steps, args.eval_callback_samples,
            args.eval_callback_threshold * 100,
            not args.no_wandb,
        )

    # ---- 8. Train ---------------------------------------------------------
    log.info("starting distillation: %d epochs", args.epochs)
    trainer.train()

    # ---- 9. Save ---------------------------------------------------------
    # Restore use_cache=True for downstream inference (matches the
    # train_student_gkd.py pattern -- training sets it False for grad-ckpt,
    # but saved config must carry True so vLLM eval doesn't run O(N^2)).
    unwrapped = trainer.accelerator.unwrap_model(trainer.model)
    unwrapped.config.use_cache = True
    args.merged_output.mkdir(parents=True, exist_ok=True)
    trainer.save_model(str(args.merged_output))
    tokenizer.save_pretrained(str(args.merged_output))
    log.info("saved full-FT model to %s", args.merged_output)
    return 0


if __name__ == "__main__":
    sys.exit(main())

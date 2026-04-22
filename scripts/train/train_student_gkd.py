#!/usr/bin/env python
# ABOUTME: On-policy GKD (Generalized Knowledge Distillation) training for Qwen3.5
# ABOUTME: student with Qwen3.5-35B-A3B teacher using token-level JSD loss.

"""
GKD on-policy distillation: Qwen3.5-35B-A3B teacher → Qwen3.5-0.8B student.

Implements the GKD algorithm from "On-Policy Distillation of Language Models:
Learning from Self-Generated Mistakes" (Agarwal et al., ICLR 2024). At each
training step, with probability λ the student generates its own sequence
(on-policy), otherwise the teacher's pre-generated sequence is used (off-policy).
In both cases, the loss is the Generalized Jensen-Shannon Divergence between the
teacher's and student's token-level logit distributions on the completion tokens.

Key difference from off-policy SFT (train_student.py):
    - Off-policy SFT: student sees only teacher-generated text → train-test
      distribution mismatch (exposure bias). Loss is NLL on teacher tokens.
    - On-policy GKD: student also trains on its own generated text, guided by
      teacher logits → learns from its own mistakes. Loss is JSD on the full
      vocabulary distribution, not just the teacher's chosen token.

Pipeline:
    1. Load Qwen3.5 teacher data (off-policy targets from philly_teacher_qwen35.jsonl)
    2. Build messages-format dataset for TRL DataCollatorForChatML
    3. Load teacher (Qwen3.5-35B-A3B) in bitsandbytes NF4 on dedicated GPU
    4. Load student (Qwen3.5-0.8B or SFT checkpoint) with LoRA
    5. GKD training via TRL experimental GKDTrainer
    6. Save LoRA adapter, optionally merge

Requires the matching environment (see ENV_VERSION.md)ironment (see scripts/env/setup_gkd_env.sh).
    transformers >= 5.x (for qwen3_5 Gated DeltaNet architecture)
    trl >= 1.0 (for experimental GKDTrainer)
    peft, bitsandbytes, datasets, accelerate

GPU layout (4× RTX 4090, 25 GB each):
    GPU 0: Student Qwen3.5-0.8B + LoRA training  (~4 GB weights + optimizer)
    GPU 1: Teacher Qwen3.5-35B-A3B NF4 4-bit     (~18 GB, inference only)

Example (from SFT checkpoint warm start):
    $ ENV=python/bin
    $ CUDA_VISIBLE_DEVICES=0,1 $ENV/python scripts/train/train_student_gkd.py \\
        --samples data/processed/philly_samples.jsonl \\
        --teacher-data data/teacher/philly_teacher_qwen35.jsonl \\
        --teacher-model Qwen/Qwen3.5-35B-A3B \\
        --student-model ckpt/student-qwen35-0.8b-merged \\
        --output ckpt/student-gkd-v0 \\
        --epochs 3 --lmbda 0.5 --beta 0.5

Example (from base model, no prior SFT):
    $ CUDA_VISIBLE_DEVICES=0,1 $ENV/python scripts/train/train_student_gkd.py \\
        --teacher-model Qwen/Qwen3.5-35B-A3B \\
        --student-model Qwen/Qwen3.5-0.8B \\
        --output ckpt/student-gkd-v0 \\
        --epochs 3

Hyperparameter guide:
    lmbda (λ): fraction of on-policy steps. 0.0 = pure off-policy (reverts to
        sequence-level KD), 1.0 = pure on-policy. Paper default 0.5.
    beta (β): JSD interpolation. 0.0 = forward KL (mode-covering, recommended
        by paper), 0.5 = symmetric JSD, 1.0 = reverse KL (mode-seeking).
    temperature: softmax temperature for both logit distributions and student
        sampling. Higher = softer distributions = more exploration. Default 0.9.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from pathlib import Path
from typing import Any, Iterator

# Silence TRL experimental warning
os.environ.setdefault("TRL_EXPERIMENTAL_SILENCE", "1")

# Project root so we can import configs.* and scripts.*
PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from configs.teacher_prompt import SYSTEM_INSTRUCTION, build_user_prompt  # noqa: E402
from scripts.teacher.validate_teacher import load_samples_by_id, validate_record  # noqa: E402

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("train_gkd")


# ---------- Data preparation ----------


def iter_jsonl(path: Path) -> Iterator[dict[str, Any]]:
    """Stream JSONL records from a file, skipping blanks and parse errors.

    Args:
        path (Path): path to a JSONL file.

    Yields:
        dict: one parsed record per non-empty line.
    """
    with path.open("r", encoding="utf-8") as f:
        for raw in f:
            line = raw.strip()
            if not line:
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError:
                continue


def teacher_output_to_assistant_text(teacher_output: dict[str, Any]) -> str:
    """Serialize teacher_output dict to the canonical assistant-turn string.

    Args:
        teacher_output (dict): dict with keys persona, rationales, ranking.

    Returns:
        str: JSON-serialized string used as the SFT/GKD completion target.
    """
    payload = {
        "persona": teacher_output["persona"],
        "rationales": teacher_output["rationales"],
        "ranking": teacher_output["ranking"],
    }
    return json.dumps(payload, ensure_ascii=False, indent=2)


def load_and_filter(
    samples_path: Path,
    teacher_path: Path,
) -> tuple[list[dict[str, Any]], dict[str, int]]:
    """Join samples with teacher outputs, dropping invalid or failed records.

    Args:
        samples_path (Path): processed samples JSONL.
        teacher_path (Path): teacher output JSONL.

    Returns:
        tuple: (examples, stats) where examples is a list of joined records
            {"sample_id", "sample", "teacher"}.
    """
    samples_by_id = load_samples_by_id(samples_path)
    log.info("loaded %d processed samples from %s", len(samples_by_id), samples_path)

    stats: dict[str, int] = {
        "total_teacher": 0,
        "dropped_preexisting_error": 0,
        "dropped_no_sample": 0,
        "dropped_invalid": 0,
        "kept": 0,
    }
    examples: list[dict[str, Any]] = []

    for rec in iter_jsonl(teacher_path):
        stats["total_teacher"] += 1
        if rec.get("error") is not None:
            stats["dropped_preexisting_error"] += 1
            continue
        sid = rec.get("sample_id")
        sample = samples_by_id.get(sid) if sid else None
        if sample is None:
            stats["dropped_no_sample"] += 1
            continue
        err = validate_record(rec, sample)
        if err is not None:
            stats["dropped_invalid"] += 1
            continue
        examples.append({"sample_id": sid, "sample": sample, "teacher": rec})
        stats["kept"] += 1

    log.info("join+filter stats: %s", stats)
    return examples, stats


def _split_bucket(sample_id: str, ratio: float) -> str:
    """Deterministically assign a sample_id to 'train' or 'eval'.

    Args:
        sample_id (str): unique id of the joined record.
        ratio (float): fraction in [0, 1] to assign to the train bucket.

    Returns:
        str: "train" or "eval".
    """
    import hashlib
    h = hashlib.sha1(sample_id.encode("utf-8")).digest()
    v = int.from_bytes(h[:4], "big") / 2**32
    return "train" if v < ratio else "eval"


def split_examples(
    examples: list[dict[str, Any]],
    ratio: float = 0.9,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Split joined examples deterministically into train and eval lists.

    Args:
        examples (list[dict]): output of load_and_filter.
        ratio (float): train fraction (default 0.9).

    Returns:
        tuple: (train_examples, eval_examples)
    """
    train: list[dict[str, Any]] = []
    ev: list[dict[str, Any]] = []
    for ex in examples:
        if _split_bucket(ex["sample_id"], ratio) == "train":
            train.append(ex)
        else:
            ev.append(ex)
    return train, ev


def build_messages_example(
    sample_rec: dict[str, Any],
    teacher_rec: dict[str, Any],
) -> dict[str, list[dict[str, str]]]:
    """Build one GKD training example in ChatML messages format.

    GKDTrainer's DataCollatorForChatML expects a "messages" field containing
    the full conversation. The collator automatically splits messages into
    prompt (all but last) and completion (last message), applying loss masking
    on the prompt tokens.

    Args:
        sample_rec (dict): processed sample record (history, candidates, ...).
        teacher_rec (dict): teacher output record (teacher_output).

    Returns:
        dict: {"messages": [system_msg, user_msg, assistant_msg]}
    """
    user_text = build_user_prompt(sample_rec)
    assistant_text = teacher_output_to_assistant_text(teacher_rec["teacher_output"])
    return {
        "messages": [
            {"role": "system", "content": SYSTEM_INSTRUCTION},
            {"role": "user", "content": user_text},
            {"role": "assistant", "content": assistant_text},
        ],
    }


def build_gkd_dataset(
    examples: list[dict[str, Any]],
) -> list[dict[str, list[dict[str, str]]]]:
    """Convert joined examples to GKD messages format.

    Args:
        examples (list[dict]): joined records from load_and_filter.

    Returns:
        list[dict]: each entry has a "messages" field for DataCollatorForChatML.
    """
    return [
        build_messages_example(ex["sample"], ex["teacher"])
        for ex in examples
    ]


# ---------- Eval callback (lesson: never run blind for 24h) ----------


def _make_parse_rate_callback_class():
    """Factory that returns GKDParseRateCallback with TrainerCallback base.

    Deferred so transformers is not imported at module level (keeps --help fast).

    Returns:
        type: GKDParseRateCallback class inheriting from TrainerCallback.
    """
    from transformers import TrainerCallback

    class GKDParseRateCallback(TrainerCallback):
        """TrainerCallback that generates samples and checks JSON parse rate.

        Every ``check_steps`` training steps, picks ``n_samples`` random eval
        prompts, generates completions with the current student, and attempts
        JSON parse on each. If the parse rate drops below ``threshold``,
        training is stopped early.

        This prevents a repeat of the incident where a 24h GKD run produced 0%
        parseable outputs and the collapse was only discovered at eval time.

        Args:
            eval_examples (list[dict]): joined records from load_and_filter
                (the eval split). Each has "sample" and "teacher" sub-dicts.
            tokenizer: HF tokenizer for the student model.
            check_steps (int): check every N global steps.
            n_samples (int): how many samples to generate per check.
            threshold (float): minimum parse rate to continue training.
        """

        def __init__(
            self,
            eval_examples: list[dict[str, Any]],
            tokenizer: Any,
            check_steps: int = 100,
            n_samples: int = 5,
            threshold: float = 0.8,
            log_wandb_sample: bool = False,
        ):
            super().__init__()
            import random
            self._eval_examples = eval_examples
            self._tokenizer = tokenizer
            self._check_steps = check_steps
            self._n_samples = n_samples
            self._threshold = threshold
            self._log_wandb_sample = log_wandb_sample
            self._rng = random.Random(42)

        def on_step_end(self, args, state, control, model=None, **kwargs):
            """Called at the end of each training step.

            Args:
                args: TrainingArguments (unused beyond logging).
                state: TrainerState with global_step.
                control: TrainerControl — set should_training_stop=True to halt.
                model: the student model.
            """
            if self._check_steps <= 0:
                return
            if state.global_step % self._check_steps != 0 or state.global_step == 0:
                return
            if model is None:
                return

            import torch

            n = min(self._n_samples, len(self._eval_examples))
            picks = self._rng.sample(self._eval_examples, n)

            # Build prompts (system + user, no assistant)
            prompts = []
            for ex in picks:
                user_text = build_user_prompt(ex["sample"])
                messages = [
                    {"role": "system", "content": SYSTEM_INSTRUCTION},
                    {"role": "user", "content": user_text},
                ]
                prompt_text = self._tokenizer.apply_chat_template(
                    messages, tokenize=False, add_generation_prompt=True,
                )
                prompts.append(prompt_text)

            # Temporarily enable use_cache for generation (training sets it
            # False for gradient_checkpointing compatibility)
            was_training = model.training
            orig_use_cache = model.config.use_cache
            model.eval()
            model.config.use_cache = True

            parsed = 0
            first_text: str | None = None
            first_parsed_ok = False
            try:
                with torch.no_grad():
                    for idx, prompt_text in enumerate(prompts):
                        inputs = self._tokenizer(
                            prompt_text, return_tensors="pt",
                        ).to(model.device)
                        out = model.generate(
                            **inputs, max_new_tokens=1024,
                            do_sample=False, use_cache=True,
                        )
                        # Decode only the generated tokens (skip prompt)
                        gen_ids = out[0, inputs["input_ids"].shape[1]:]
                        text = self._tokenizer.decode(gen_ids, skip_special_tokens=True)
                        ok = False
                        try:
                            json.loads(text)
                            parsed += 1
                            ok = True
                        except (json.JSONDecodeError, ValueError):
                            pass
                        if idx == 0:
                            first_text = text
                            first_parsed_ok = ok
            finally:
                # Restore training state
                model.config.use_cache = orig_use_cache
                if was_training:
                    model.train()

            rate = parsed / n if n > 0 else 0.0
            log.info(
                "eval_callback step=%d: parse_rate=%.1f%% (%d/%d)",
                state.global_step, rate * 100, parsed, n,
            )

            # Log the first generation to wandb for human eyeball inspection
            # of the JSON structure. Cheap signal complementing the scalar
            # parse_rate: collapse patterns (length=1, infinite loops,
            # repetitive text) show up visibly in the text panel before the
            # parse_rate threshold fires.
            if self._log_wandb_sample and first_text is not None:
                try:
                    import wandb  # local import keeps --help fast
                    if wandb.run is not None:
                        import html as _html
                        # Don't pass step=; HF Trainer's wandb integration
                        # already advances an opaque internal step counter
                        # far beyond global_step (seen one run where
                        # global_step=20 but wandb counter was 344 → "step
                        # must be monotonically increasing" warning + data
                        # dropped). Letting wandb auto-append keeps the
                        # sample generation line up with the scalar
                        # metrics recorded by the same forward pass.
                        wandb.log(
                            {
                                "val/generation": wandb.Html(
                                    f"<pre>{_html.escape(first_text[:4000])}</pre>"
                                ),
                                "val/parsed": int(first_parsed_ok),
                                "val/gen_length_chars": len(first_text),
                                "val/parse_rate": rate,
                                "val/global_step": state.global_step,
                            },
                        )
                except Exception as e:  # pragma: no cover
                    log.warning("wandb validation log failed: %s", e)

            if rate < self._threshold:
                log.warning(
                    "eval_callback: parse_rate %.1f%% < threshold %.1f%% — "
                    "stopping training to prevent mode collapse.",
                    rate * 100, self._threshold * 100,
                )
                control.should_training_stop = True

    return GKDParseRateCallback


# ---------- Model loading ----------


def load_teacher_model(
    model_name: str,
    teacher_device: str,
    load_in_4bit: bool = True,
    teacher_gpus: str | None = None,
    per_gpu_max_memory_gb: int = 20,
):
    """Load the teacher model for GKD inference (no gradient).

    Loads the teacher in bitsandbytes NF4 4-bit quantization by default to
    fit the 35B MoE model. The teacher only needs forward passes (no
    optimizer states or gradients).

    GPU placement modes:
      - Single GPU (default): ``teacher_device`` like "cuda:1" places the
        whole model on one device. Works when the model fits (e.g.
        Qwen3.5-35B-A3B NF4 ≈ 20 GB on a 24 GB card).
      - Multi-GPU shard: pass ``teacher_gpus="0,1,2"`` to spread the model
        across listed GPUs via ``device_map="auto"`` with a per-device
        memory cap. The student gets a separate GPU (see --student-gpu).
        Recommended for the 4× 24 GB layout where the student also needs
        a GPU for training.

    Args:
        model_name (str): HF model id or local path for the teacher.
        teacher_device (str): single-GPU placement, e.g. "cuda:1". Ignored
            when ``teacher_gpus`` is set.
        load_in_4bit (bool): use bitsandbytes NF4 quantization (default True).
        teacher_gpus (str | None): comma-separated GPU indices for sharding,
            e.g. "0,1,2". When set, switches to device_map="auto" with
            ``max_memory`` restricted to the listed GPUs.
        per_gpu_max_memory_gb (int): cap per teacher GPU when sharding. GPUs
            not in the list are zeroed in max_memory so HF's auto placement
            cannot spill onto them.

    Returns:
        PreTrainedModel: loaded teacher model in eval mode.
    """
    import torch
    from transformers import AutoModelForImageTextToText, BitsAndBytesConfig

    log.info(
        "loading teacher: %s (device=%s, gpus=%s, 4bit=%s)",
        model_name, teacher_device, teacher_gpus, load_in_4bit,
    )

    kwargs: dict[str, Any] = {
        "torch_dtype": torch.bfloat16,
        "trust_remote_code": True,
    }

    if load_in_4bit:
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
        )
        kwargs["quantization_config"] = bnb_config

    if teacher_gpus:
        # Multi-GPU shard mode. Two compounding constraints:
        #   1. HF's device-map planner sizes layouts at bf16 — Qwen3.5-35B-A3B
        #      is ~70 GB bf16, but only ~20 GB once bnb shrinks to NF4. So we
        #      budget by the post-quantization size we want and let
        #      ``llm_int8_enable_fp32_cpu_offload`` absorb the handful of
        #      non-quantizable modules (embeddings, norms, vision encoder)
        #      onto CPU during load.
        #   2. "auto" fills each GPU sequentially, which overflows a single
        #      device during load. "balanced" distributes evenly, which is
        #      what we want when the model is genuinely tight.
        teacher_set = {int(s.strip()) for s in teacher_gpus.split(",") if s.strip()}
        max_memory: dict[int | str, str] = {}
        for i in range(torch.cuda.device_count()):
            max_memory[i] = f"{per_gpu_max_memory_gb}GiB" if i in teacher_set else "0GiB"
        # Let embeddings + vision components land on CPU if they don't fit:
        max_memory["cpu"] = "64GiB"
        kwargs["device_map"] = "balanced"
        kwargs["max_memory"] = max_memory
        if load_in_4bit:
            # Required by bnb_4bit when any module (even bf16) might land on CPU.
            kwargs["quantization_config"].llm_int8_enable_fp32_cpu_offload = True
    else:
        kwargs["device_map"] = {"": teacher_device}

    teacher = AutoModelForImageTextToText.from_pretrained(model_name, **kwargs)
    teacher.eval()
    log.info("teacher loaded: %s parameters", f"{sum(p.numel() for p in teacher.parameters()):,}")
    return teacher


def load_student_model(
    model_name: str,
    attn_impl: str = "sdpa",
    student_gpu: str | None = None,
):
    """Load the student model for GKD training.

    Qwen3.5-0.8B is a VLM (Qwen3_5ForConditionalGeneration) but works for
    text-only tasks. The model is loaded in bfloat16. When LoRA is used,
    GKDTrainer wraps it with PEFT; when --no-lora is set, the full model
    trains directly.

    Args:
        model_name (str): HF model id or local checkpoint path.
        attn_impl (str): attention implementation ("sdpa", "flash_attention_2",
            "eager"). Default "sdpa" for broadest compatibility; flash_attention_2
            may not support Gated DeltaNet linear_attention layers.
        student_gpu (str | None): explicit GPU index ("3") or device string
            ("cuda:3") for the student. Required when teacher shards across
            multiple GPUs so the student doesn't compete for memory. If unset,
            the model lands on cuda:0 by default (HF behavior).

    Returns:
        PreTrainedModel: loaded student model ready for training.
    """
    import torch
    from transformers import AutoConfig, AutoModelForCausalLM, AutoModelForImageTextToText

    log.info("loading student: %s (attn_impl=%s, gpu=%s)", model_name, attn_impl, student_gpu)

    # Dispatch by config.model_type:
    #   - "qwen3_5"       → multimodal wrapper (Qwen3_5ForConditionalGeneration)
    #                       shipped at HF hub (Qwen/Qwen3.5-0.8B base).
    #   - "qwen3_5_text"  → text-only submodel (Qwen3_5ForCausalLM) that
    #                       trainer.save_model persists after SFT (the inner
    #                       decoder without vision stack). AutoModelForImageTextToText
    #                       cannot load this class, so fall back to
    #                       AutoModelForCausalLM. Both load paths produce a
    #                       .forward(input_ids, …) → .logits interface that
    #                       GKDTrainer and the HF Trainer base expect.
    cfg = AutoConfig.from_pretrained(model_name, trust_remote_code=True)
    model_type = getattr(cfg, "model_type", "")
    is_text_only = model_type.endswith("_text")
    model_cls = AutoModelForCausalLM if is_text_only else AutoModelForImageTextToText
    log.info(
        "student config.model_type=%s → loading via %s",
        model_type, model_cls.__name__,
    )

    model = model_cls.from_pretrained(
        model_name,
        torch_dtype=torch.bfloat16,
        attn_implementation=attn_impl,
        trust_remote_code=True,
    )
    if student_gpu is not None:
        device = student_gpu if student_gpu.startswith("cuda") else f"cuda:{student_gpu}"
        model.to(device)
        log.info("student moved to %s", device)
    model.config.use_cache = False  # required with gradient checkpointing
    log.info("student loaded: %s parameters", f"{sum(p.numel() for p in model.parameters()):,}")
    return model


def find_lora_target_modules(model) -> list[str]:
    """Auto-detect LoRA-eligible linear modules in a Qwen3.5 model.

    Scans the model's named modules for nn.Linear layers in the text decoder,
    excluding embeddings and lm_head. Returns unique suffixes suitable for
    PEFT LoRA target_modules.

    This handles the Gated DeltaNet architecture where both linear_attention
    and full_attention layers contain projection matrices.

    Args:
        model: loaded HF model.

    Returns:
        list[str]: deduplicated module name suffixes (e.g. ["q_proj", "v_proj", ...]).
    """
    import torch.nn as nn

    targets = set()
    for name, module in model.named_modules():
        if not isinstance(module, nn.Linear):
            continue
        # Skip embedding and head layers
        if any(skip in name for skip in ["embed", "lm_head", "visual", "merger"]):
            continue
        # Extract the suffix (e.g., "q_proj" from "model.layers.0.self_attn.q_proj")
        suffix = name.split(".")[-1]
        if suffix.endswith("_proj") or suffix in ("gate_proj", "up_proj", "down_proj"):
            targets.add(suffix)

    result = sorted(targets)
    log.info("auto-detected LoRA targets: %s", result)
    return result


# ---------- CLI + orchestration ----------


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
        default=PROJECT_ROOT / "data/teacher/philly_teacher_qwen35.jsonl",
        help="teacher output JSONL (off-policy targets)",
    )

    # Models
    p.add_argument(
        "--teacher-model", type=str,
        default="Qwen/Qwen3.5-35B-A3B",
        help="HF model id or path for the teacher (logit source)",
    )
    p.add_argument(
        "--student-model", type=str,
        default="Qwen/Qwen3.5-0.8B",
        help="HF model id or path for the student (trainee)",
    )
    p.add_argument(
        "--teacher-device", type=str, default="cuda:1",
        help="single-GPU device for teacher (used only when --teacher-gpus is unset)",
    )
    p.add_argument(
        "--teacher-gpus", type=str, default=None,
        help=(
            "comma-separated GPU indices to shard teacher across via "
            "device_map='auto' + max_memory restriction. Example: '0,1,2' "
            "puts the 35B-A3B teacher (NF4 ~20 GB total) across GPUs 0–2 "
            "and leaves GPU 3 for the student. When set, --teacher-device "
            "is ignored."
        ),
    )
    p.add_argument(
        "--student-gpu", type=str, default=None,
        help=(
            "explicit GPU index for the student (e.g. '3'). Required when "
            "--teacher-gpus shards across multiple GPUs so that the student "
            "lands on a non-shared GPU. If unset, defaults to cuda:0."
        ),
    )
    p.add_argument(
        "--teacher-full-precision", action="store_true",
        help="load teacher in bf16 instead of NF4 4-bit (needs more VRAM)",
    )
    p.add_argument(
        "--teacher-url", type=str, default=os.environ.get("TEACHER_BASE_URL"),
        help=(
            "vLLM OpenAI base URL (e.g. 'http://10.1.1.48:8100/v1'). When set, "
            "the teacher is queried via HTTPTeacherAdapter instead of being "
            "loaded in-process. Skips --teacher-model / --teacher-gpus / "
            "--teacher-device / --teacher-full-precision. Required for the "
            "2-node Stage-3 topology. Falls back to env var TEACHER_BASE_URL."
        ),
    )
    p.add_argument(
        "--teacher-served-name", type=str, default="qwen35-teacher",
        help=(
            "--served-model-name used on the vLLM teacher (default: "
            "'qwen35-teacher', matching scripts/serve/serve_teacher_vllm.sh). "
            "Only used with --teacher-url."
        ),
    )
    p.add_argument(
        "--teacher-top-k", type=int, default=50,
        help=(
            "top-K log-probabilities to request per position via "
            "prompt_logprobs (default 50). Server --max-logprobs must be "
            "at least this large. Only used with --teacher-url."
        ),
    )
    p.add_argument(
        "--teacher-fill-logit", type=float, default=-50.0,
        help=(
            "logit value for the V-K unfilled vocabulary slots in the HTTP "
            "teacher's returned logits tensor (default -50.0). Low enough "
            "that exp(fill) ≈ 0 so top-K softmax mass is preserved."
        ),
    )
    p.add_argument(
        "--teacher-logits-dtype", type=str, default="fp32",
        choices=["fp32", "bf16"],
        help=(
            "dtype for the [B,T,V] tensor the HTTP adapter allocates and "
            "returns. With Qwen3.5 (V=248320) the full-precision tensor is "
            "~5 GB per rank; bf16 halves it to ~2.5 GB at the cost of a few "
            "bits of precision before TRL's internal log_softmax. Default "
            "fp32 for numerical safety; use bf16 when peak memory is tight."
        ),
    )
    p.add_argument(
        "--no-lora", action="store_true",
        help=(
            "disable LoRA and train the full student model. Recommended for "
            "Qwen3.5-0.8B (~15 GB peak on single 24 GB GPU) for consistency "
            "with the v2-sft baseline (also full-FT). When set, peft is "
            "never imported and the merge step is skipped (the trained "
            "model is the merged checkpoint)."
        ),
    )

    # Output
    p.add_argument(
        "--output", type=Path,
        default=PROJECT_ROOT / "ckpt/student-gkd-v0",
        help="output directory for LoRA adapter checkpoints",
    )
    p.add_argument(
        "--merged-output", type=Path,
        default=PROJECT_ROOT / "ckpt/student-gkd-merged",
        help="where to save the LoRA-merged full model",
    )

    # GKD hyperparameters
    p.add_argument(
        "--lmbda", type=float, default=0.5,
        help="on-policy fraction: 0.0 = pure off-policy, 1.0 = pure on-policy (default: 0.5)",
    )
    p.add_argument(
        "--beta", type=float, default=0.5,
        help="JSD interpolation: 0.0 = forward KL, 0.5 = JSD, 1.0 = reverse KL (default: 0.5)",
    )
    p.add_argument(
        "--temperature", type=float, default=0.9,
        help="softmax temperature for logit distributions and student sampling (default: 0.9)",
    )
    p.add_argument(
        "--max-new-tokens", type=int, default=1024,
        help="max tokens for on-policy student generation (default: 1024)",
    )
    p.add_argument(
        "--seq-kd", action="store_true",
        help="use sequence-level KD (teacher generates, student learns NLL) instead of token-level GKD",
    )

    # Training hyperparameters
    p.add_argument("--epochs", type=int, default=3)
    p.add_argument("--lora-r", type=int, default=16)
    p.add_argument("--lora-alpha", type=int, default=32)
    p.add_argument("--lora-dropout", type=float, default=0.05)
    p.add_argument(
        "--lora-targets", type=str, nargs="+", default=None,
        help="LoRA target modules (default: auto-detect from model architecture)",
    )
    p.add_argument("--learning-rate", type=float, default=2e-4)
    p.add_argument("--warmup-steps", type=int, default=20)
    p.add_argument("--max-length", type=int, default=4096)
    p.add_argument("--batch-size", type=int, default=1)
    p.add_argument("--grad-accum", type=int, default=8)
    p.add_argument("--eval-ratio", type=float, default=0.9, help="train fraction")
    p.add_argument("--max-train-samples", type=int, default=None, help="cap training set (smoke tests)")
    p.add_argument(
        "--attn-impl", type=str, default="sdpa",
        choices=["sdpa", "flash_attention_2", "eager"],
        help="student attention impl (default: sdpa for Gated DeltaNet compat)",
    )

    # Merge + logging
    p.add_argument("--skip-merge", action="store_true", help="skip LoRA merge step")
    p.add_argument("--wandb-project", type=str, default="llm-distillation-yelp-gkd")
    p.add_argument("--wandb-run-name", type=str, default=None)
    p.add_argument("--no-wandb", action="store_true", help="disable wandb logging")

    # Eval callback — lesson: never run blind for 24h
    p.add_argument(
        "--eval-callback-steps", type=int, default=0,
        help=(
            "run a parse-rate sanity check every N training steps. 0 = "
            "disabled (default). Recommended: 100. The callback generates "
            "a few samples, attempts JSON parse, and logs the parse rate. "
            "If the rate falls below --eval-callback-threshold, training "
            "stops early to prevent wasting GPU hours on a collapsed model."
        ),
    )
    p.add_argument(
        "--eval-callback-samples", type=int, default=5,
        help="number of eval samples to generate per callback check (default 5)",
    )
    p.add_argument(
        "--eval-callback-threshold", type=float, default=0.8,
        help=(
            "minimum parse rate to continue training (default 0.8). If the "
            "callback detects parse rate below this, training stops."
        ),
    )
    p.add_argument(
        "--guided-student-rollout", action="store_true",
        help=(
            "Inject an xgrammar LogitsProcessor (TeacherResponse JSON schema) "
            "into the on-policy student generate() call. Requires xgrammar "
            "(present in the matching environment (see ENV_VERSION.md); absent in the matching environment). Use for "
            "guided-JSON GKD; see scripts/train/guided_gkd.py."
        ),
    )

    return p.parse_args()


def main() -> int:
    args = parse_args()

    # HTTP teacher does not implement .generate(); seq_kd would call it.
    if args.teacher_url and args.seq_kd:
        log.error(
            "--teacher-url is incompatible with --seq-kd: sequence-level KD "
            "requires teacher.generate(), which the HTTP adapter does not "
            "implement. Either drop --seq-kd or use an in-process teacher."
        )
        return 2

    # Heavy imports deferred so --help stays fast.
    # peft is only imported when LoRA is requested — full-FT envs (e.g.
    # the matching environment) need not install peft.
    import torch
    from datasets import Dataset
    from transformers import AutoTokenizer
    from trl.experimental.gkd import GKDConfig, GKDTrainer

    if args.no_lora:
        LoraConfig = None  # type: ignore[assignment]
    else:
        from peft import LoraConfig  # noqa: F401

    # ---- 1. Load + filter + split --------------------------------------------------
    examples, _stats = load_and_filter(args.samples, args.teacher_data)
    if not examples:
        log.error("no training examples after filter; aborting")
        return 2

    train_exs, eval_exs = split_examples(examples, ratio=args.eval_ratio)
    if args.max_train_samples is not None:
        train_exs = train_exs[: args.max_train_samples]
    log.info("split: train=%d, eval=%d", len(train_exs), len(eval_exs))

    # ---- 2. Build GKD dataset in messages format -----------------------------------
    train_data = build_gkd_dataset(train_exs)
    eval_data = build_gkd_dataset(eval_exs)
    log.info("built messages-format dataset: train=%d, eval=%d", len(train_data), len(eval_data))

    train_ds = Dataset.from_list(train_data)
    eval_ds = Dataset.from_list(eval_data) if eval_data else None

    # ---- 3. Load tokenizer ---------------------------------------------------------
    log.info("loading tokenizer from %s", args.student_model)
    tokenizer = AutoTokenizer.from_pretrained(
        args.student_model, use_fast=True, trust_remote_code=True,
    )
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    # ---- 4. Load student model first (HTTP adapter needs its vocab dim) ------------
    student_model = load_student_model(
        args.student_model,
        attn_impl=args.attn_impl,
        student_gpu=args.student_gpu,
    )

    # ---- 5. Load teacher model -----------------------------------------------------
    # Two paths:
    #   - 2-node HTTP: --teacher-url set → HTTPTeacherAdapter queries vLLM on
    #     the other node. No teacher weights materialise on this GPU.
    #   - in-process: load teacher weights on this machine (NF4 or bf16),
    #     either on a single GPU or sharded across several.
    if args.teacher_url:
        from scripts.serve.http_teacher_adapter import HTTPTeacherAdapter

        # The HTTP teacher must emit logits whose last dimension matches the
        # student's lm_head out_features (HF typically pads vocab up to a
        # multiple of 64 for kernel alignment — tokenizer length alone is too
        # small). JSD stacks the two logits tensors, so any mismatch throws.
        student_out = student_model.get_output_embeddings()
        vocab_size = int(student_out.weight.shape[0])
        log.info(
            "teacher: HTTP adapter → %s (model=%s, top_k=%d, vocab=%d)",
            args.teacher_url, args.teacher_served_name,
            args.teacher_top_k, vocab_size,
        )
        teacher_logits_dtype = (
            torch.bfloat16 if args.teacher_logits_dtype == "bf16" else torch.float32
        )
        teacher_model = HTTPTeacherAdapter(
            base_url=args.teacher_url,
            model_name=args.teacher_served_name,
            vocab_size=vocab_size,
            top_k=args.teacher_top_k,
            fill_logit=args.teacher_fill_logit,
            logits_dtype=teacher_logits_dtype,
        )
    else:
        teacher_model = load_teacher_model(
            args.teacher_model,
            teacher_device=args.teacher_device,
            load_in_4bit=not args.teacher_full_precision,
            teacher_gpus=args.teacher_gpus,
        )

    # ---- 6. LoRA config (skipped under --no-lora) ----------------------------------
    if args.no_lora:
        log.info("--no-lora: full FT (peft_config=None, no adapter/merge)")
        lora_config = None
    else:
        target_modules = args.lora_targets or find_lora_target_modules(student_model)
        if not target_modules:
            log.error("no LoRA target modules found; check model architecture")
            return 2

        lora_config = LoraConfig(
            r=args.lora_r,
            lora_alpha=args.lora_alpha,
            lora_dropout=args.lora_dropout,
            bias="none",
            task_type="CAUSAL_LM",
            target_modules=target_modules,
        )

    # ---- 7. GKD config -------------------------------------------------------------
    if args.no_lora:
        run_name = args.wandb_run_name or (
            f"gkd-fullft-lmbda{args.lmbda}-beta{args.beta}-{args.epochs}ep"
        )
    else:
        run_name = args.wandb_run_name or (
            f"gkd-lmbda{args.lmbda}-beta{args.beta}-r{args.lora_r}-{args.epochs}ep"
        )
    if args.no_wandb:
        log.info("wandb: DISABLED")
    else:
        os.environ.setdefault("WANDB_PROJECT", args.wandb_project)
        log.info("wandb: ENABLED project=%s run_name=%s", args.wandb_project, run_name)

    gkd_config = GKDConfig(
        output_dir=str(args.output),
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        learning_rate=args.learning_rate,
        lr_scheduler_type="cosine",
        warmup_steps=args.warmup_steps,
        weight_decay=0.01,
        max_grad_norm=1.0,
        bf16=True,
        fp16=False,
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        max_length=args.max_length,
        logging_steps=5,
        save_strategy="epoch",
        save_total_limit=2,
        eval_strategy="epoch" if eval_ds is not None else "no",
        report_to="none" if args.no_wandb else "wandb",
        run_name=run_name,
        dataloader_num_workers=0,
        # GKD-specific parameters
        temperature=args.temperature,
        lmbda=args.lmbda,
        beta=args.beta,
        max_new_tokens=args.max_new_tokens,
        disable_dropout=True,
        seq_kd=args.seq_kd,
    )

    # ---- 8. GKD Trainer ------------------------------------------------------------
    callbacks = []
    if args.eval_callback_steps > 0:
        ParseRateCB = _make_parse_rate_callback_class()
        # log_wandb_sample: surface parse rate + first generation HTML to wandb
        # when the run is wandb-backed. Callback is the only place we measure
        # parse rate during training; stderr prints get eaten by the tqdm
        # progress bar, so wandb is the only reliable visibility channel.
        parse_cb = ParseRateCB(
            eval_examples=eval_exs,
            tokenizer=tokenizer,
            check_steps=args.eval_callback_steps,
            n_samples=args.eval_callback_samples,
            threshold=args.eval_callback_threshold,
            log_wandb_sample=not args.no_wandb,
        )
        callbacks.append(parse_cb)
        log.info(
            "eval_callback enabled: every %d steps, %d samples, threshold=%.0f%%",
            args.eval_callback_steps, args.eval_callback_samples,
            args.eval_callback_threshold * 100,
        )

    # Guided student rollout: build xgrammar JSON-schema mask and swap trainer
    # class for GuidedGKDTrainer (option B in the 2026-04-21 plan — closes the
    # unconstrained-rollout parse collapse while keeping legacy dense JSD
    # on the teacher side).
    if args.guided_student_rollout:
        from configs.teacher_prompt import TeacherResponse
        from scripts.train.guided_gkd import GuidedGKDTrainer, build_json_schema_grammar

        vocab_size = int(student_model.get_output_embeddings().weight.shape[0])
        compiled_grammar = build_json_schema_grammar(
            tokenizer, TeacherResponse.model_json_schema(), vocab_size=vocab_size,
        )
        log.info(
            "guided student rollout: xgrammar compiled for TeacherResponse schema "
            "(vocab=%d)", vocab_size,
        )
        trainer_cls = GuidedGKDTrainer
        trainer_extra = {"compiled_grammar": compiled_grammar}
    else:
        trainer_cls = GKDTrainer
        trainer_extra = {}

    trainer = trainer_cls(
        model=student_model,
        teacher_model=teacher_model,
        args=gkd_config,
        train_dataset=train_ds,
        eval_dataset=eval_ds,
        processing_class=tokenizer,
        peft_config=lora_config,
        **trainer_extra,
    )

    for cb in callbacks:
        trainer.add_callback(cb)

    # Override TRL's over-conservative cache policy for the on-policy generation
    # phase. gkd_trainer.py sets generation_kwargs["use_cache"] = False whenever
    # gradient_checkpointing is True, arguing that the two are incompatible —
    # but .generate() runs under unwrap_model_for_generation() + no_grad, so
    # the KV cache cannot be touched by the backward pass. Without the cache,
    # generating max_new_tokens tokens is O(N²) in sequence length: observed
    # 343 s/step for max_new_tokens=1024 on Qwen3.5-0.8B. Enabling the cache
    # cuts generation to linear in N (~25 s/step for 1024 tokens), without
    # affecting gradient correctness. Override both the GenerationConfig object
    # and the generation_kwargs dict (the former is passed to .generate(),
    # the latter to unwrap_model_for_generation).
    trainer.generation_config.use_cache = True
    trainer.generation_kwargs["use_cache"] = True
    log.info("override: trainer.generation_config.use_cache = True (speedup on-policy gen 7-8×)")

    log.info(
        "starting GKD training: %d epochs, λ=%.2f (on-policy frac), β=%.2f (JSD), T=%.1f",
        args.epochs, args.lmbda, args.beta, args.temperature,
    )
    trainer.train()

    # ---- 9. Save final model -------------------------------------------------------
    # LoRA path: save adapter → reload base + merge → save merged full model.
    # Full-FT path: trainer already holds the trained full model; save it
    # directly to merged_output (PTQ scripts expect that path).
    if args.no_lora:
        # Restore use_cache before saving — training sets it to False for
        # gradient_checkpointing compatibility, but inference (eval_metrics,
        # vLLM) needs True for KV-cache-enabled generation. Without this
        # reset the saved config.json carries use_cache=false and downstream
        # generation becomes O(N²).
        unwrapped = trainer.accelerator.unwrap_model(trainer.model)
        unwrapped.config.use_cache = True
        args.merged_output.mkdir(parents=True, exist_ok=True)
        trainer.save_model(str(args.merged_output))
        tokenizer.save_pretrained(str(args.merged_output))
        log.info("saved full-FT model to %s (no LoRA merge needed)", args.merged_output)
        return 0

    final_ckpt = args.output / "final"
    trainer.save_model(str(final_ckpt))
    log.info("saved final LoRA adapter to %s", final_ckpt)

    # ---- 10. Merge LoRA into base (optional) ---------------------------------------
    if args.skip_merge:
        log.info("--skip-merge set; not producing merged model")
        return 0

    from peft import PeftModel

    log.info("merging LoRA adapter into base for %s", args.merged_output)
    base_for_merge = load_student_model(args.student_model, attn_impl=args.attn_impl)
    peft_model = PeftModel.from_pretrained(base_for_merge, str(final_ckpt))
    merged = peft_model.merge_and_unload()
    args.merged_output.mkdir(parents=True, exist_ok=True)
    merged.save_pretrained(str(args.merged_output))
    tokenizer.save_pretrained(str(args.merged_output))
    log.info("merged model saved to %s", args.merged_output)
    return 0


if __name__ == "__main__":
    sys.exit(main())

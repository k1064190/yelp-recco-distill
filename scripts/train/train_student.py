#!/usr/bin/env python
# ABOUTME: LoRA SFT for Qwen3-4B student on teacher outputs (persona + rationales
# ABOUTME: + index-based ranking) using TRL v1.0 SFTTrainer + prompt-completion format.

"""
Train the Qwen3-4B student on Gemini-generated teacher outputs via LoRA SFT.

Pipeline:
    1. Load processed samples and teacher outputs, join them on sample_id,
       drop any records whose teacher output fails validate_record (silent
       SFT-data-poison defense, see scripts/teacher/validate_teacher.py).
    2. Deterministic 90/10 train/eval split (hash of sample_id, so the same
       sample always lands in the same split across runs — important for
       calibration-data reuse in scripts/quantize/quantize_w4a16.py).
    3. Build one training example per record in TRL v1.0 "conversational
       prompt-completion" format:
           {"prompt":     [system, user],
            "completion": [assistant]}
       where the assistant content is the JSON-serialized teacher_output.
       TRL's SFTTrainer applies the model's chat template automatically and,
       by default, computes loss only on the completion tokens
       (``SFTConfig.completion_only_loss=True``) — no hand-rolled collator.
    4. Load Qwen3-4B in bf16 with attn_implementation="flash_attention_2"
       (flash-attn 2.8.3 is functional on the current torch 2.8 / cu128
       stack — see ENV_VERSION.md §4.1 and ). The --attn-impl
       flag still accepts "sdpa" as a fallback for environments where
       flash-attn cannot be installed.
    5. Wrap with PEFT LoRA (r=16, alpha=32, all attention + MLP projections).
    6. Train 3 epochs, cosine schedule, effective batch 8, gradient
       checkpointing. Save best LoRA adapter per epoch, then merge adapter
       into base weights for downstream W4A16 PTQ.

Example (smoke test, 1 epoch):
    $ python scripts/train/train_student.py \\
        --samples data/processed/philly_samples.jsonl \\
        --teacher data/teacher/philly_teacher.jsonl \\
        --base Qwen/Qwen3-4B \\
        --output ckpt/student-v0 \\
        --epochs 1 \\
        --max-train-samples 30

Full run:
    $ python scripts/train/train_student.py \\
        --samples data/processed/philly_samples.jsonl \\
        --teacher data/teacher/philly_teacher.jsonl \\
        --base Qwen/Qwen3-4B \\
        --output ckpt/student-v0 \\
        --epochs 3

Logging:
    Weights & Biases logging is ON by default (report_to="wandb"). Before the
    first full run on a new machine, authenticate once with either:
        $ wandb login                              # interactive
        $ export WANDB_API_KEY=<your key>         # non-interactive
    The run writes to project ``llm-distillation-yelp`` unless --wandb-project
    is passed. Use --no-wandb to opt out (useful for smoke tests where setting
    up the sync is overhead). At startup the script prints an explicit banner
    line showing the wandb state so runs cannot silently skip tracking.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import os
import sys
from pathlib import Path
from typing import Any, Iterator

# Project root so we can import scripts.* and configs.*
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
log = logging.getLogger("train_student")


# ---------- Data preparation (pure, testable) ----------


def iter_jsonl(path: Path) -> Iterator[dict[str, Any]]:
    """Stream JSONL records from a file.

    Args:
        path (Path): path to a JSONL file.

    Yields:
        dict: one parsed record per non-empty line. Blank and unparseable
            lines are skipped silently.
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
    """Serialize a teacher_output dict to the assistant-turn completion string.

    We use ``json.dumps`` with indent=2 so that the student learns to emit
    human-readable JSON (easier to inspect at benchmark time) and with
    ``ensure_ascii=False`` so non-ASCII characters (e.g. accents in Yelp
    business names) survive round-trip without escape noise.

    Args:
        teacher_output (dict): dict with keys persona (str), rationales
            (list of {candidate_index: int, reason: str}), ranking
            (list of int, 1-based candidate indices).

    Returns:
        str: the canonical serialized form that becomes the SFT target.
    """
    payload = {
        "persona": teacher_output["persona"],
        "rationales": teacher_output["rationales"],
        "ranking": teacher_output["ranking"],
    }
    return json.dumps(payload, ensure_ascii=False, indent=2)


def build_training_example(
    sample_rec: dict[str, Any],
    teacher_rec: dict[str, Any],
) -> dict[str, list[dict[str, str]]]:
    """Build one TRL v1.0 conversational prompt-completion example.

    The example uses TRL's native prompt/completion conversational format,
    which causes SFTTrainer to mask loss over the prompt tokens automatically
    (``completion_only_loss=True`` is the SFTConfig default). This avoids any
    hand-rolled DataCollatorForCompletionOnlyLM wiring and keeps the
    prompt/completion boundary unambiguous.

    Args:
        sample_rec (dict): a processed sample record (keys: history,
            candidates, ...), used to rebuild the exact user prompt the
            teacher saw at generation time.
        teacher_rec (dict): a teacher output record (keys: teacher_output).
            Must already have passed ``scripts.teacher.validate_teacher.validate_record``.

    Returns:
        dict: {
            "prompt":     [{"role": "system", "content": ...},
                           {"role": "user",   "content": ...}],
            "completion": [{"role": "assistant", "content": ...}],
        }
    """
    user_text = build_user_prompt(sample_rec)
    assistant_text = teacher_output_to_assistant_text(teacher_rec["teacher_output"])
    return {
        "prompt": [
            {"role": "system", "content": SYSTEM_INSTRUCTION},
            {"role": "user", "content": user_text},
        ],
        "completion": [
            {"role": "assistant", "content": assistant_text},
        ],
    }


def _split_bucket(sample_id: str, ratio: float) -> str:
    """Deterministically assign a sample_id to 'train' or 'eval'.

    Hash-based so that 90/10 always maps the same sample_ids to the same
    split across runs, which lets quantize_w4a16.py reuse the train split
    for calibration without reintroducing eval leakage.

    Args:
        sample_id (str): unique id of the joined record.
        ratio (float): fraction in [0, 1] to assign to the train bucket.

    Returns:
        str: "train" or "eval".
    """
    h = hashlib.sha1(sample_id.encode("utf-8")).digest()
    # Use the first 4 bytes as a uint32 -> [0, 1) float
    v = int.from_bytes(h[:4], "big") / 2**32
    return "train" if v < ratio else "eval"


def load_and_filter(
    samples_path: Path,
    teacher_path: Path,
) -> tuple[list[dict[str, Any]], dict[str, int]]:
    """Join samples with teacher outputs, dropping invalid or failed records.

    Any teacher record whose ``error`` is non-None is dropped (pre-existing
    quota failures, etc). The remaining records are validated inline against
    their source sample with ``scripts.teacher.validate_teacher.validate_record``;
    records that fail validation (ranking mismatch, empty persona, ...) are
    also dropped and counted per-reason. This is the same contract as running
    ``scripts/teacher/validate_teacher.py`` standalone, but without mutating the
    teacher file on disk — important because ``generate_teacher.py`` may
    still be appending to that file concurrently.

    Args:
        samples_path (Path): processed samples JSONL.
        teacher_path (Path): teacher output JSONL.

    Returns:
        tuple: (examples, stats) where examples is a list of joined records
            {"sample_id", "sample", "teacher"} ready to feed into
            ``build_training_example``, and stats is a dict of drop reasons.
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


def split_examples(
    examples: list[dict[str, Any]],
    ratio: float = 0.9,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Split joined examples deterministically into train and eval lists.

    Args:
        examples (list[dict]): output of ``load_and_filter``.
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


# ---------- Training orchestration ----------


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--samples", type=Path, default=PROJECT_ROOT / "data/processed/philly_samples.jsonl")
    p.add_argument("--teacher", type=Path, default=PROJECT_ROOT / "data/teacher/philly_teacher.jsonl")
    p.add_argument("--base", type=str, default="Qwen/Qwen3-4B", help="HF base model id")
    p.add_argument("--output", type=Path, default=PROJECT_ROOT / "ckpt/student-v0")
    p.add_argument(
        "--merged-output",
        type=Path,
        default=PROJECT_ROOT / "ckpt/student-merged",
        help="where to save the LoRA-merged full model (for vLLM serving + PTQ)",
    )
    p.add_argument("--epochs", type=int, default=3)
    p.add_argument(
        "--no-lora",
        action="store_true",
        help=(
            "disable LoRA and train the full model. Use for small students "
            "(e.g. Qwen3.5-0.8B) where full FT fits on a single 24 GB GPU "
            "(~15 GB peak). The resulting checkpoint is already a full model, "
            "so the merge_and_unload step is skipped."
        ),
    )
    p.add_argument("--lora-r", type=int, default=16)
    p.add_argument("--lora-alpha", type=int, default=32)
    p.add_argument("--lora-dropout", type=float, default=0.05)
    p.add_argument("--learning-rate", type=float, default=2e-4)
    p.add_argument("--warmup-steps", type=int, default=20)
    p.add_argument(
        "--max-length",
        type=int,
        default=8192,
        help=(
            "hard sequence length cap. Samples longer than max_length are "
            "dropped at load time (not truncated) to avoid silent "
            "assistant-turn truncation that would corrupt SFT. For the "
            "current 2931-record Qwen3.5 teacher dataset, the rendered "
            "(chat template + JSON indent=2) distribution is p95=3493, "
            "p99=3614, p99.9=3870, max=4002 (see "
            "data/results/token_distribution.json), so max_length=4096 "
            "drops 0/2931 samples. Raising above 4096 only wastes memory."
        ),
    )
    p.add_argument("--batch-size", type=int, default=1)
    p.add_argument("--grad-accum", type=int, default=8)
    p.add_argument("--eval-ratio", type=float, default=0.9, help="train fraction (not eval)")
    p.add_argument(
        "--max-train-samples",
        type=int,
        default=None,
        help="cap training set (for smoke tests)",
    )
    p.add_argument(
        "--attn-impl",
        type=str,
        default="flash_attention_2",
        choices=["sdpa", "flash_attention_2", "eager"],
        help=(
            "Qwen3 attention implementation. Default 'flash_attention_2' uses "
            "O(N) attention memory via fused kernels (enabled by flash_attn "
            "2.8.3 on the torch 2.8 stack — see ENV_VERSION.md). Fall back to "
            "'sdpa' only in environments where flash-attn is not installable "
            "(e.g. torch 2.9+ without pre-built wheels); sdpa uses O(N^2) "
            "attention memory so long sequences may OOM."
        ),
    )
    p.add_argument(
        "--optim",
        type=str,
        default="adamw_torch",
        help=(
            "Optimizer passed through to HF TrainingArguments. Default "
            "'adamw_torch' (fp32 m+v states, 8 bytes/param) was used for "
            "Qwen3.5-0.8B v2-sft. For Qwen3.5-9B full-FT on a single 96 GB "
            "Pro 6000, switch to 'paged_adamw_8bit': keeps AdamW dynamics but "
            "stores m+v in 8-bit (2 bytes/param) with CPU paging on spikes, "
            "dropping optimizer state from ~72 GB to ~18 GB. Requires "
            "bitsandbytes (present in the matching environment)."
        ),
    )
    p.add_argument(
        "--skip-merge",
        action="store_true",
        help="skip LoRA->base merge step (useful for smoke tests on tiny data)",
    )
    p.add_argument(
        "--wandb-project",
        type=str,
        default="llm-distillation-yelp",
    )
    p.add_argument(
        "--wandb-run-name",
        type=str,
        default=None,
        help="default: student-v0-lora-r{r}-{epochs}ep",
    )
    p.add_argument(
        "--no-wandb",
        action="store_true",
        help=(
            "disable wandb logging (report_to='none'). Intended for smoke "
            "tests; full training runs should leave wandb ON so loss, "
            "grad_norm, learning_rate, and eval metrics are tracked."
        ),
    )
    return p.parse_args()


def main() -> int:
    args = parse_args()

    # Heavy ML imports deferred so --help and unit tests don't pay the cost.
    # peft is only imported when LoRA is requested — full-FT envs (e.g. the matching environment)
    # need not install peft.
    import torch
    from datasets import Dataset
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from trl import SFTConfig, SFTTrainer

    if args.no_lora:
        LoraConfig = None  # type: ignore[assignment]
    else:
        from peft import LoraConfig  # noqa: F401

    # ---- 1. Load + filter + split ---------------------------------------------------
    examples, _stats = load_and_filter(args.samples, args.teacher)
    if not examples:
        log.error("no training examples after filter; aborting")
        return 2

    train_exs, eval_exs = split_examples(examples, ratio=args.eval_ratio)
    if args.max_train_samples is not None:
        train_exs = train_exs[: args.max_train_samples]
    log.info("split: train=%d, eval=%d", len(train_exs), len(eval_exs))

    if len(train_exs) < 20:
        log.warning(
            "very small training set (%d examples) — expect noisy gradients",
            len(train_exs),
        )

    # ---- 2. Load tokenizer (needed before length filter) ---------------------------
    log.info("loading tokenizer for %s", args.base)
    tokenizer = AutoTokenizer.from_pretrained(args.base, use_fast=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    def _measure_tokens(example: dict) -> int:
        """Tokenize a prompt-completion example and return total length.

        Args:
            example (dict): output of ``build_training_example``.

        Returns:
            int: total token count for the rendered chat template.
        """
        full_text = tokenizer.apply_chat_template(
            example["prompt"] + example["completion"],
            tokenize=False,
            add_generation_prompt=False,
        )
        return len(tokenizer(full_text, add_special_tokens=False)["input_ids"])

    def _filter_by_length(
        source: list[dict], label: str
    ) -> list[dict]:
        """Drop pre-built examples whose total tokenized length exceeds max_length.

        Over-length samples are dropped (not truncated) because SFTTrainer's
        default truncation strategy could silently cut the assistant-turn
        completion, leaving the model to train on empty targets.

        Args:
            source (list[dict]): a list of prompt-completion dicts.
            label (str): log label ("train" or "eval").

        Returns:
            list[dict]: only the examples that fit within args.max_length.
        """
        kept: list[dict] = []
        dropped = 0
        for ex in source:
            if _measure_tokens(ex) <= args.max_length:
                kept.append(ex)
            else:
                dropped += 1
        if dropped:
            log.warning(
                "%s: dropped %d/%d examples for exceeding max_length=%d",
                label, dropped, len(source), args.max_length,
            )
        return kept

    train_data = [build_training_example(e["sample"], e["teacher"]) for e in train_exs]
    eval_data = [build_training_example(e["sample"], e["teacher"]) for e in eval_exs]

    train_data = _filter_by_length(train_data, "train")
    eval_data = _filter_by_length(eval_data, "eval") if eval_data else []
    log.info("after length filter: train=%d, eval=%d", len(train_data), len(eval_data))
    if not train_data:
        log.error("length filter dropped all training examples; aborting")
        return 2

    train_ds = Dataset.from_list(train_data)
    eval_ds = Dataset.from_list(eval_data) if eval_data else None

    # ---- 3. Load base model ---------------------------------------------------------
    log.info("loading base model %s (dtype=bfloat16, attn_impl=%s)", args.base, args.attn_impl)

    model = AutoModelForCausalLM.from_pretrained(
        args.base,
        dtype=torch.bfloat16,
        attn_implementation=args.attn_impl,
    )
    model.config.use_cache = False  # required with gradient checkpointing

    # ---- 4. LoRA config (skipped under --no-lora) -----------------------------------
    if args.no_lora:
        log.info("--no-lora: training full model (peft_config=None, no adapter/merge)")
        lora_config = None
    else:
        lora_config = LoraConfig(
            r=args.lora_r,
            lora_alpha=args.lora_alpha,
            lora_dropout=args.lora_dropout,
            bias="none",
            task_type="CAUSAL_LM",
            target_modules=[
                "q_proj",
                "k_proj",
                "v_proj",
                "o_proj",
                "gate_proj",
                "up_proj",
                "down_proj",
            ],
        )

    # ---- 5. SFT config --------------------------------------------------------------
    if args.no_lora:
        run_name = args.wandb_run_name or f"student-fullft-{args.epochs}ep"
    else:
        run_name = args.wandb_run_name or f"student-v0-lora-r{args.lora_r}-{args.epochs}ep"
    if args.no_wandb:
        log.info("wandb: DISABLED (--no-wandb); report_to='none'")
    else:
        os.environ.setdefault("WANDB_PROJECT", args.wandb_project)
        wandb_key_present = bool(os.environ.get("WANDB_API_KEY"))
        log.info(
            "wandb: ENABLED project=%s run_name=%s (WANDB_API_KEY %s)",
            args.wandb_project,
            run_name,
            "set" if wandb_key_present else "NOT set — run `wandb login` once if this is the first run on this box",
        )

    sft_config = SFTConfig(
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
        optim=args.optim,
        logging_steps=5,
        save_strategy="epoch",
        save_total_limit=2,
        eval_strategy="epoch" if eval_ds is not None else "no",
        report_to="none" if args.no_wandb else "wandb",
        run_name=run_name,
        dataloader_num_workers=0,  # tokenizer fork issues on some boxes
        remove_unused_columns=False,
        completion_only_loss=True,  # mask loss on prompt tokens (default, explicit for clarity)
    )

    # ---- 6. Train --------------------------------------------------------------------
    trainer = SFTTrainer(
        model=model,
        args=sft_config,
        train_dataset=train_ds,
        eval_dataset=eval_ds,
        processing_class=tokenizer,
        peft_config=lora_config,
    )

    log.info("starting training: %d epochs x %d train examples", args.epochs, len(train_data))
    trainer.train()

    # ---- 7. Save final model --------------------------------------------------------
    # LoRA path: save adapter → reload base + merge → save merged full model.
    # Full-FT path: trainer already holds the full trained model; save it
    # directly to merged_output (PTQ scripts expect that path).
    if args.no_lora:
        args.merged_output.mkdir(parents=True, exist_ok=True)
        trainer.save_model(str(args.merged_output))
        tokenizer.save_pretrained(str(args.merged_output))
        log.info("saved full-FT model to %s (no LoRA merge needed)", args.merged_output)
        return 0

    final_ckpt = args.output / "final"
    trainer.save_model(str(final_ckpt))
    log.info("saved final LoRA adapter to %s", final_ckpt)

    # ---- 8. Merge LoRA into base and save full bf16 model for downstream PTQ --------
    if args.skip_merge:
        log.info("--skip-merge set; not producing merged model")
        return 0

    from peft import PeftModel

    log.info("merging LoRA adapter into base for %s", args.merged_output)
    # Reload base in bf16 without attention overrides — merge is a no-grad op.
    base_for_merge = AutoModelForCausalLM.from_pretrained(
        args.base,
        dtype=torch.bfloat16,
        attn_implementation=args.attn_impl,
    )
    peft_model = PeftModel.from_pretrained(base_for_merge, str(final_ckpt))
    merged = peft_model.merge_and_unload()
    args.merged_output.mkdir(parents=True, exist_ok=True)
    merged.save_pretrained(str(args.merged_output))
    tokenizer.save_pretrained(str(args.merged_output))
    log.info("merged model saved to %s", args.merged_output)
    return 0


if __name__ == "__main__":
    sys.exit(main())

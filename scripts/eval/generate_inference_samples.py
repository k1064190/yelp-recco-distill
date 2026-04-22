#!/usr/bin/env python
# ABOUTME: Generate 5 inference examples (prompt + output) per model variant and
# ABOUTME: save as readable JSON. Supports teacher (HTTP), FP16, NF4, W4A16, GGUF backends.

"""Produce side-by-side inference samples for every model variant we trained.

Writes one JSON per backend into ``data/inference_samples/<backend>.json`` with
structure::

    {
      "backend": "teacher|v2-sft|v2-sft-w4a16|v2-sft-nf4|v2-sft-gguf-q4km",
      "model_path_or_url": "...",
      "dtype": "bf16|fp16|nf4|gguf-q4km",
      "generated_at": "2026-04-14T...",
      "samples": [
        {
          "sample_id": "...",
          "positive_business_id": "...",
          "prompt_preview": "first 400 chars",
          "prompt_length_tokens": int,
          "output_text": "...",
          "parsed_ranking": [int, ...] | null,
          "json_parse_ok": bool,
          "latency_sec": float
        },
        ...
      ]
    }

Picks the same 5 eval examples across all backends (deterministic: positions
0, 56, 113, 169, 226 of the 283-sample eval split).

Backends:
  - ``teacher``: queries the vLLM OpenAI endpoint (default http://localhost:8100/v1).
    Run in any env that has ``openai`` installed.
  - ``fp16``: HF transformers, bf16 weights (``ckpt/student-v2-sft-merged``).
    Run in ``the matching environment``.
  - ``nf4``: HF transformers + bnb (``ckpt/student-v2-sft-nf4``). Run in ``the matching environment``.
  - ``w4a16``: HF transformers + compressed-tensors (``ckpt/student-v2-sft-w4a16``).
    Run in ``the matching environment``.
  - ``gguf``: llama-cpp-python (``ckpt/student-v2-sft-gguf-q4km/student-q4-k-m.gguf``).
    Run in ``the matching environment`` (only env with the CUDA-built llama-cpp-python wheel).
"""

from __future__ import annotations

import argparse
import gc
import json
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from configs.teacher_prompt import SYSTEM_INSTRUCTION, build_user_prompt  # noqa: E402
from scripts.train.train_student import load_and_filter, split_examples  # noqa: E402
from scripts.eval.eval_metrics import parse_json_ranking, extract_student_ranking  # noqa: E402

DEFAULT_OUT_DIR = PROJECT_ROOT / "data" / "inference_samples"
DEFAULT_TEACHER_URL = "http://localhost:8100/v1"
DEFAULT_TEACHER_MODEL = "qwen35-teacher"
DEFAULT_STUDENT_CKPT = {
    "fp16":  PROJECT_ROOT / "ckpt/student-v2-sft-merged",
    "nf4":   PROJECT_ROOT / "ckpt/student-v2-sft-nf4",
    "w4a16": PROJECT_ROOT / "ckpt/student-v2-sft-w4a16",
    "gguf":  PROJECT_ROOT / "ckpt/student-v2-sft-gguf-q4km/student-q4-k-m.gguf",
}


# ---------- Sample selection ----------


def pick_samples(n: int = 5) -> list[dict[str, Any]]:
    """Return ``n`` deterministically-spaced eval samples.

    Args:
        n (int): number of samples to return.

    Returns:
        list[dict]: each dict has keys ``sample`` and ``teacher`` (per
            ``load_and_filter`` output).
    """
    examples, _ = load_and_filter(
        PROJECT_ROOT / "data/processed/philly_samples.jsonl",
        PROJECT_ROOT / "data/teacher/philly_teacher_qwen35.jsonl",
    )
    _, eval_exs = split_examples(examples, ratio=0.9)
    if len(eval_exs) < n:
        return eval_exs
    step = len(eval_exs) // n
    return [eval_exs[i * step] for i in range(n)]


def build_chat(sample: dict[str, Any]) -> list[dict[str, str]]:
    """Build the OpenAI-style chat messages for a sample."""
    return [
        {"role": "system", "content": SYSTEM_INSTRUCTION},
        {"role": "user", "content": build_user_prompt(sample)},
    ]


def summarize_output(text: str, candidates: list[dict[str, Any]]) -> dict[str, Any]:
    """Try to parse the model output as the training JSON schema.

    Args:
        text (str): raw model output.
        candidates (list[dict]): ordered candidate records from the sample,
            each with a ``business_id`` key. Passed straight through to
            ``extract_student_ranking`` which indexes into it.

    Returns:
        dict: with keys ``parsed_ranking`` (list[int] or None),
            ``recovered_business_ids`` (list[str] or None), ``json_parse_ok`` (bool).
    """
    parsed = parse_json_ranking(text)
    if parsed is None or "ranking" not in parsed:
        return {"parsed_ranking": None, "recovered_business_ids": None, "json_parse_ok": False}
    ranking_idx = parsed.get("ranking")
    try:
        ranking = extract_student_ranking(text, candidates)
    except Exception:
        ranking = None
    return {
        "parsed_ranking": ranking_idx,
        "recovered_business_ids": ranking,
        "json_parse_ok": ranking is not None,
    }


# ---------- Backends ----------


def run_teacher(samples: list[dict[str, Any]], base_url: str, model_name: str,
                max_new_tokens: int = 1024, temperature: float = 0.0,
                warmup: int = 1) -> list[dict[str, Any]]:
    """Query the vLLM OpenAI endpoint once per sample.

    Uses vLLM's ``guided_json`` (structured output) with the same
    ``TeacherResponse`` schema the teacher was originally sampled under,
    so the output here is comparable to the training data (not a free-form
    "Thinking Process:" preamble).

    Does ``warmup`` throwaway requests first (prime the server's prefix cache
    and any lazy CUDA init) so the measured latencies are steady-state.
    Reports end-to-end latency, prompt/output token counts (from the usage
    block), and the derived ms/output-token rate.
    """
    from openai import OpenAI
    from configs.teacher_prompt import TeacherResponse
    schema = TeacherResponse.model_json_schema()
    client = OpenAI(base_url=base_url, api_key="dummy")

    response_format = {
        "type": "json_schema",
        "json_schema": {"name": "teacher_response", "schema": schema},
    }

    # Warmup — use the first sample, discard timing.
    if warmup > 0 and samples:
        warmup_messages = build_chat(samples[0]["sample"])
        for _ in range(warmup):
            client.chat.completions.create(
                model=model_name, messages=warmup_messages,
                max_tokens=16, temperature=temperature,
                response_format=response_format,
            )

    out = []
    for ex in samples:
        sample = ex["sample"]
        messages = build_chat(sample)
        t0 = time.perf_counter()
        resp = client.chat.completions.create(
            model=model_name,
            messages=messages,
            max_tokens=max_new_tokens,
            temperature=temperature,
            response_format=response_format,
        )
        dt = time.perf_counter() - t0
        text = resp.choices[0].message.content or ""
        usage = getattr(resp, "usage", None)
        prompt_tokens = getattr(usage, "prompt_tokens", None) if usage else None
        output_tokens = getattr(usage, "completion_tokens", None) if usage else None
        ms_per_tok = round(1000.0 * dt / output_tokens, 3) if output_tokens else None
        candidates = sample["candidates"]
        out.append({
            "sample_id": sample["sample_id"],
            "positive_business_id": sample["positive_business_id"],
            "prompt_preview": messages[-1]["content"][:400],
            "prompt_length_chars": len(messages[-1]["content"]),
            "prompt_tokens": prompt_tokens,
            "output_tokens": output_tokens,
            "output_text": text,
            **summarize_output(text, candidates),
            "latency_sec": round(dt, 3),
            "ms_per_output_token": ms_per_tok,
        })
    return out


def run_hf_student(samples: list[dict[str, Any]], ckpt: Path, quant: str,
                   max_new_tokens: int = 1024, device: str = "cuda:0",
                   warmup: int = 1) -> list[dict[str, Any]]:
    """Load a HF checkpoint (fp16/nf4/w4a16) and generate with greedy decoding.

    Warmup runs generate a short sequence (throwaway) before timed passes so
    that CUDA graphs / lazy allocations are settled. ``torch.cuda.synchronize``
    brackets every timed region so the wall clock reflects GPU work, not
    kernel queueing.
    """
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    kwargs: dict[str, Any] = {"trust_remote_code": True}
    if quant == "nf4":
        from transformers import BitsAndBytesConfig
        kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True,
        )
        kwargs["device_map"] = {"": device}
    elif quant == "w4a16":
        kwargs["dtype"] = torch.float16
        kwargs["device_map"] = {"": device}
    else:  # fp16 means the bf16-merged student (our training dtype)
        kwargs["dtype"] = torch.bfloat16
        kwargs["device_map"] = {"": device}

    tokenizer = AutoTokenizer.from_pretrained(str(ckpt), use_fast=True, trust_remote_code=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(str(ckpt), attn_implementation="sdpa", **kwargs)
    model.eval()

    # Warmup — generate a short sequence against the first sample's prompt.
    if warmup > 0 and samples:
        warmup_prompt = tokenizer.apply_chat_template(
            build_chat(samples[0]["sample"]),
            tokenize=False, add_generation_prompt=True,
        )
        warmup_inputs = tokenizer(warmup_prompt, return_tensors="pt", add_special_tokens=False)
        warmup_inputs = {k: v.to(device) for k, v in warmup_inputs.items()}
        with torch.no_grad():
            for _ in range(warmup):
                model.generate(**warmup_inputs, max_new_tokens=8, do_sample=False,
                               pad_token_id=tokenizer.pad_token_id)
        torch.cuda.synchronize(device) if device.startswith("cuda") else None

    out = []
    for ex in samples:
        sample = ex["sample"]
        messages = build_chat(sample)
        prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = tokenizer(prompt, return_tensors="pt", add_special_tokens=False)
        inputs = {k: v.to(device) for k, v in inputs.items()}
        prompt_tokens = int(inputs["input_ids"].shape[1])
        if device.startswith("cuda"):
            torch.cuda.synchronize(device)
        t0 = time.perf_counter()
        with torch.no_grad():
            gen = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                pad_token_id=tokenizer.pad_token_id,
            )
        if device.startswith("cuda"):
            torch.cuda.synchronize(device)
        dt = time.perf_counter() - t0
        new_tokens = gen[0, prompt_tokens:]
        output_tokens = int(new_tokens.shape[0])
        text = tokenizer.decode(new_tokens, skip_special_tokens=True)
        ms_per_tok = round(1000.0 * dt / output_tokens, 3) if output_tokens else None
        candidates = sample["candidates"]
        out.append({
            "sample_id": sample["sample_id"],
            "positive_business_id": sample["positive_business_id"],
            "prompt_preview": messages[-1]["content"][:400],
            "prompt_tokens": prompt_tokens,
            "output_tokens": output_tokens,
            "output_text": text,
            **summarize_output(text, candidates),
            "latency_sec": round(dt, 3),
            "ms_per_output_token": ms_per_tok,
        })

    # free GPU memory so the next backend (if any) has headroom.
    del model
    gc.collect()
    torch.cuda.empty_cache()
    return out


def run_vllm(samples: list[dict[str, Any]], model_path_or_id: str, variant: str,
             tp: int = 1, max_new_tokens: int = 1024, warmup: int = 1,
             gpu_memory_utilization: float = 0.85,
             max_model_len: int = 4096) -> list[dict[str, Any]]:
    """Load a model inside vLLM offline (``vllm.LLM``) and generate per-prompt.

    One prompt per ``llm.generate`` call (batch=1) so the measured latency is
    directly comparable to the HF path we already have. vLLM's own CUDA graph
    + prefix cache + kernel fusion do the heavy lifting on top of batch=1.

    Uses the Qwen3.5 chat template via ``llm.get_tokenizer()`` for prompt
    rendering. For the teacher we also attach a ``guided_decoding`` JSON
    schema (via ``SamplingParams.guided_decoding``) so the teacher emits
    training-comparable JSON instead of a chain-of-thought preamble.

    Args:
        samples (list[dict]): joined eval records.
        model_path_or_id (str): checkpoint path or HF id.
        variant (str): one of ``{"teacher", "fp16", "w4a16", "nf4"}``. Controls
            dtype / quantization.
        tp (int): tensor-parallel size (teacher uses 4 here).
        max_new_tokens (int): sampling cap.
        warmup (int): throwaway warmup passes before the timed region.
        gpu_memory_utilization (float): passed to vLLM.
        max_model_len (int): vLLM KV-cache budget knob.

    Returns:
        list[dict]: one entry per sample with the standard fields.
    """
    import torch
    from vllm import LLM, ModelRegistry, SamplingParams
    # vLLM renamed GuidedDecodingParams -> StructuredOutputsParams in 0.19+.
    try:
        from vllm.sampling_params import StructuredOutputsParams as _StructParams  # 0.19+
    except ImportError:
        from vllm.sampling_params import GuidedDecodingParams as _StructParams  # older

    # vLLM 0.19.1rc1 upstream quirks that matter for our renamed text-only ckpt:
    # 1. `Qwen3_5ForCausalLM` is not in the ModelRegistry (only the VLM variant
    #    `Qwen3_5ForConditionalGeneration` is). Register it so the loader picks
    #    the text-only class for our checkpoint.
    # 2. That `Qwen3_5ForCausalLM` class does NOT inherit from `IsHybrid`,
    #    which is what tells vLLM's KV-cache planner to tolerate per-layer
    #    page-size disagreement between linear_attention (Gated DeltaNet) and
    #    full_attention layers. Without it the engine raises
    #    "The page size of the layer is not divisible by the maximum page
    #    size." Subclass it to add the interface, register the subclass.
    if "Qwen3_5ForCausalLM" not in ModelRegistry.get_supported_archs():
        from vllm.model_executor.models.qwen3_5 import Qwen3_5ForCausalLM as _QCausal
        from vllm.model_executor.models.interfaces import IsHybrid as _IsHybrid

        class Qwen3_5ForCausalLMHybrid(_QCausal, _IsHybrid):  # type: ignore[misc]
            pass

        ModelRegistry.register_model("Qwen3_5ForCausalLM", Qwen3_5ForCausalLMHybrid)

    llm_kwargs: dict[str, Any] = {
        "model": str(model_path_or_id),
        "tensor_parallel_size": tp,
        "gpu_memory_utilization": gpu_memory_utilization,
        "max_model_len": max_model_len,
        "trust_remote_code": True,
        "enforce_eager": False,  # allow CUDA graphs — main vLLM speedup
    }
    if variant == "w4a16":
        llm_kwargs["quantization"] = "compressed-tensors"
        llm_kwargs["dtype"] = "float16"  # W4A16 fp16 compute path
    elif variant == "nf4":
        # bnb in vLLM can either load a bf16 ckpt and quantize on-the-fly
        # (load_in_4bit via quantization="bitsandbytes"), or read a
        # pre-packed ckpt. The VLM-merged bf16 ckpt is the bf16 case.
        llm_kwargs["quantization"] = "bitsandbytes"
        llm_kwargs["dtype"] = "bfloat16"
    else:
        # fp16 (student bf16) or teacher
        llm_kwargs["dtype"] = "bfloat16"

    llm = LLM(**llm_kwargs)
    tokenizer = llm.get_tokenizer()

    # Structured output for teacher (match training-time vLLM guided_json).
    structured = None
    if variant == "teacher":
        from configs.teacher_prompt import TeacherResponse
        structured = _StructParams(json=TeacherResponse.model_json_schema())

    sp_kwargs = {"temperature": 0.0, "max_tokens": max_new_tokens}
    if structured is not None:
        sp_kwargs["structured_outputs"] = structured
    sp = SamplingParams(**sp_kwargs)

    # Warmup — use the first sample with a tight token budget.
    if warmup > 0 and samples:
        warmup_prompt = tokenizer.apply_chat_template(
            build_chat(samples[0]["sample"]),
            tokenize=False, add_generation_prompt=True,
        )
        warmup_sp_kwargs = {"temperature": 0.0, "max_tokens": 16}
        if structured is not None:
            warmup_sp_kwargs["structured_outputs"] = structured
        warmup_sp = SamplingParams(**warmup_sp_kwargs)
        for _ in range(warmup):
            llm.generate([warmup_prompt], warmup_sp, use_tqdm=False)

    out = []
    for ex in samples:
        sample = ex["sample"]
        prompt = tokenizer.apply_chat_template(
            build_chat(sample), tokenize=False, add_generation_prompt=True,
        )
        t0 = time.perf_counter()
        results = llm.generate([prompt], sp, use_tqdm=False)
        dt = time.perf_counter() - t0
        r = results[0]
        text = r.outputs[0].text
        output_tokens = len(r.outputs[0].token_ids)
        prompt_tokens = len(r.prompt_token_ids)
        ms_per_tok = round(1000.0 * dt / output_tokens, 3) if output_tokens else None
        candidates = sample["candidates"]
        out.append({
            "sample_id": sample["sample_id"],
            "positive_business_id": sample["positive_business_id"],
            "prompt_preview": build_chat(sample)[-1]["content"][:400],
            "prompt_tokens": prompt_tokens,
            "output_tokens": output_tokens,
            "output_text": text,
            **summarize_output(text, candidates),
            "latency_sec": round(dt, 3),
            "ms_per_output_token": ms_per_tok,
        })

    # Release before next backend.
    del llm
    gc.collect()
    torch.cuda.empty_cache()
    return out


def run_gguf(samples: list[dict[str, Any]], gguf_path: Path,
             max_new_tokens: int = 1024, warmup: int = 1) -> list[dict[str, Any]]:
    """Load a GGUF file via llama-cpp-python and generate with greedy decoding.

    llama-cpp-python's ``Llama`` reports completion token counts in the
    response ``usage`` block, which feeds our ms/token metric.
    """
    from llama_cpp import Llama
    llm = Llama(
        model_path=str(gguf_path),
        n_gpu_layers=-1,  # full GPU offload where supported
        n_ctx=4096,
        verbose=False,
    )

    if warmup > 0 and samples:
        warmup_messages = build_chat(samples[0]["sample"])
        warmup_prompt = (
            f"<|im_start|>system\n{warmup_messages[0]['content']}<|im_end|>\n"
            f"<|im_start|>user\n{warmup_messages[1]['content']}<|im_end|>\n"
            f"<|im_start|>assistant\n"
        )
        for _ in range(warmup):
            llm(warmup_prompt, max_tokens=8, temperature=0.0, stop=["<|im_end|>"])

    out = []
    for ex in samples:
        sample = ex["sample"]
        messages = build_chat(sample)
        prompt = (
            f"<|im_start|>system\n{messages[0]['content']}<|im_end|>\n"
            f"<|im_start|>user\n{messages[1]['content']}<|im_end|>\n"
            f"<|im_start|>assistant\n"
        )
        t0 = time.perf_counter()
        resp = llm(
            prompt,
            max_tokens=max_new_tokens,
            temperature=0.0,
            stop=["<|im_end|>"],
        )
        dt = time.perf_counter() - t0
        text = resp["choices"][0]["text"]
        usage = resp.get("usage") or {}
        prompt_tokens = usage.get("prompt_tokens")
        output_tokens = usage.get("completion_tokens")
        ms_per_tok = round(1000.0 * dt / output_tokens, 3) if output_tokens else None
        candidates = sample["candidates"]
        out.append({
            "sample_id": sample["sample_id"],
            "positive_business_id": sample["positive_business_id"],
            "prompt_preview": messages[-1]["content"][:400],
            "prompt_length_chars": len(prompt),
            "prompt_tokens": prompt_tokens,
            "output_tokens": output_tokens,
            "output_text": text,
            **summarize_output(text, candidates),
            "latency_sec": round(dt, 3),
            "ms_per_output_token": ms_per_tok,
        })
    return out


# ---------- CLI ----------


def parse_args() -> argparse.Namespace:
    """Parse CLI args."""
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--backend",
        required=True,
        choices=["teacher", "fp16", "nf4", "w4a16", "gguf", "vllm"],
    )
    p.add_argument(
        "--variant",
        choices=["teacher", "fp16", "w4a16", "nf4"],
        default=None,
        help="variant to run when --backend vllm (teacher / fp16 / w4a16 / nf4)",
    )
    p.add_argument(
        "--tp", type=int, default=1,
        help="tensor-parallel size (vLLM only; teacher uses 4)",
    )
    p.add_argument(
        "--gpu-mem-util", type=float, default=0.85,
        help="vLLM --gpu-memory-utilization",
    )
    p.add_argument(
        "--max-model-len", type=int, default=4096,
        help="vLLM --max-model-len",
    )
    p.add_argument("--n-samples", type=int, default=5)
    p.add_argument("--max-new-tokens", type=int, default=1024)
    p.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    p.add_argument("--device", type=str, default="cuda:0")
    p.add_argument("--teacher-url", type=str, default=DEFAULT_TEACHER_URL)
    p.add_argument("--teacher-model", type=str, default=DEFAULT_TEACHER_MODEL)
    p.add_argument("--ckpt", type=Path, default=None,
                   help="override default checkpoint/URL for the chosen backend")
    return p.parse_args()


def main() -> int:
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    samples = pick_samples(args.n_samples)
    print(f"picked {len(samples)} samples")

    if args.backend == "teacher":
        tag = "teacher"
        src = args.teacher_url
        outputs = run_teacher(samples, args.teacher_url, args.teacher_model, args.max_new_tokens)
    elif args.backend == "fp16":
        tag = "v2-sft"
        src = args.ckpt or DEFAULT_STUDENT_CKPT["fp16"]
        outputs = run_hf_student(samples, src, quant="fp16",
                                 max_new_tokens=args.max_new_tokens, device=args.device)
    elif args.backend == "nf4":
        tag = "v2-sft-nf4"
        src = args.ckpt or DEFAULT_STUDENT_CKPT["nf4"]
        outputs = run_hf_student(samples, src, quant="nf4",
                                 max_new_tokens=args.max_new_tokens, device=args.device)
    elif args.backend == "w4a16":
        tag = "v2-sft-w4a16"
        src = args.ckpt or DEFAULT_STUDENT_CKPT["w4a16"]
        outputs = run_hf_student(samples, src, quant="w4a16",
                                 max_new_tokens=args.max_new_tokens, device=args.device)
    elif args.backend == "gguf":
        tag = "v2-sft-gguf-q4km"
        src = args.ckpt or DEFAULT_STUDENT_CKPT["gguf"]
        outputs = run_gguf(samples, src, max_new_tokens=args.max_new_tokens)
    elif args.backend == "vllm":
        if args.variant is None:
            raise SystemExit("--backend vllm requires --variant {teacher,fp16,w4a16,nf4}")
        variant_ckpt_map = {
            "teacher": "Qwen/Qwen3.5-35B-A3B",
            "fp16":    DEFAULT_STUDENT_CKPT["fp16"],
            "w4a16":   DEFAULT_STUDENT_CKPT["w4a16"],
            "nf4":     DEFAULT_STUDENT_CKPT["nf4"],
        }
        variant_tag_map = {
            "teacher": "vllm-teacher",
            "fp16":    "vllm-v2-sft",
            "w4a16":   "vllm-v2-sft-w4a16",
            "nf4":     "vllm-v2-sft-nf4",
        }
        tag = variant_tag_map[args.variant]
        src = args.ckpt or variant_ckpt_map[args.variant]
        outputs = run_vllm(
            samples, str(src), variant=args.variant,
            tp=args.tp, max_new_tokens=args.max_new_tokens,
            gpu_memory_utilization=args.gpu_mem_util,
            max_model_len=args.max_model_len,
        )
    else:
        raise ValueError(args.backend)

    doc = {
        "backend": tag,
        "model_path_or_url": str(src),
        "dtype": args.backend,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "samples": outputs,
    }
    out_path = args.out_dir / f"{tag}.json"
    out_path.write_text(json.dumps(doc, indent=2, ensure_ascii=False))
    print(f"wrote {out_path} ({len(outputs)} samples)")
    return 0


if __name__ == "__main__":
    sys.exit(main())

#!/usr/bin/env python
# ABOUTME: Stress test the vLLM teacher serve by replaying GKD-like traffic
# ABOUTME: (prompt_logprobs=50, long real prompts, sequential + concurrent).

"""
Teacher vLLM serve stress test.

Replicates the request pattern of scripts/serve/http_teacher_adapter.py which was
the trigger for the OOM crashes from teacher vLLM logprob workspace on the 4090 node. The goal here
is to confirm that on Pro 6000 (96 GB, single GPU) the same workload runs
without allocator pressure.

Phases
------
1. Single long prompt probe — tokenize one p99-ish sample, request
   prompt_logprobs=50, time the round trip.
2. Sequential hammer — 20 real prompts sampled from philly_samples.jsonl,
   each with prompt_logprobs=50, back-to-back, recording latencies + GPU
   memory after each call.
3. Concurrent burst — 8 parallel prompts via thread pool to exercise
   continuous batching under prompt_logprobs workspace pressure.

Args:
    --base-url (str): vLLM OpenAI endpoint root, default http://127.0.0.1:8100/v1
    --model (str): served-model-name, default qwen35-teacher
    --samples (Path): philly_samples.jsonl path
    --n-sequential (int): sequential request count, default 20
    --n-concurrent (int): concurrent batch size, default 8
    --top-k (int): prompt_logprobs value (teacher top-K), default 50
    --max-new-tokens (int): completion length (generation-side), default 64

Returns:
    Exits 0 on full success. Exits 1 if any request fails or latency p99
    exceeds a sanity threshold (60 s).
"""

from __future__ import annotations

import argparse
import concurrent.futures as cf
import json
import statistics
import subprocess
import sys
import time
from pathlib import Path

import requests


def _gpu_used_mib(device: int) -> int:
    """Return used memory (MiB) on the given GPU index via nvidia-smi.

    Args:
        device (int): GPU index to query.

    Returns:
        int: used memory in MiB, or -1 if nvidia-smi fails.
    """
    try:
        out = subprocess.check_output(
            [
                "nvidia-smi",
                f"--query-gpu=memory.used",
                "--format=csv,noheader,nounits",
                "-i",
                str(device),
            ],
            text=True,
            timeout=5,
        )
        return int(out.strip())
    except Exception:
        return -1


def _build_prompt(sample: dict) -> str:
    """Build the user prompt string from a preprocessed sample.

    Args:
        sample (dict): one record from data/processed/philly_samples.jsonl.

    Returns:
        str: the user-side prompt ready for the tokenizer.
    """
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    from configs.teacher_prompt import build_user_prompt  # local import

    return build_user_prompt(sample)


def _post_completion(
    base_url: str,
    model: str,
    prompt_tokens: list[int],
    top_k: int,
    max_new_tokens: int,
    timeout: float = 120.0,
) -> dict:
    """Send one /v1/completions request with prompt_logprobs.

    Args:
        base_url (str): vLLM endpoint root (ends with /v1).
        model (str): served-model-name.
        prompt_tokens (list[int]): token IDs to send as prompt.
        top_k (int): prompt_logprobs value.
        max_new_tokens (int): completion length.
        timeout (float): HTTP timeout seconds.

    Returns:
        dict with {latency_s, prompt_len, output_tokens, top_k_seen,
                   http_status}.
    """
    url = f"{base_url.rstrip('/')}/completions"
    payload = {
        "model": model,
        "prompt": [prompt_tokens],
        "max_tokens": max_new_tokens,
        "temperature": 0.0,
        "prompt_logprobs": top_k,
        "logprobs": top_k,
    }
    t0 = time.perf_counter()
    r = requests.post(url, json=payload, timeout=timeout)
    latency = time.perf_counter() - t0
    if r.status_code != 200:
        return {
            "latency_s": latency,
            "prompt_len": len(prompt_tokens),
            "output_tokens": None,
            "top_k_seen": None,
            "http_status": r.status_code,
            "error": r.text[:500],
        }
    body = r.json()
    choice0 = body["choices"][0]
    pl = choice0.get("prompt_logprobs") or []
    # Count how many prompt positions have a non-null dict with K slots.
    top_k_seen = None
    for entry in pl:
        if entry:
            top_k_seen = len(entry)
            break
    return {
        "latency_s": latency,
        "prompt_len": len(prompt_tokens),
        "output_tokens": len((choice0.get("logprobs") or {}).get("tokens") or choice0.get("text", "")),
        "top_k_seen": top_k_seen,
        "http_status": 200,
    }


def main() -> int:
    """Run the three-phase stress test and report.

    Returns:
        int: 0 on full success, 1 on any failure or p99 > 60 s.
    """
    ap = argparse.ArgumentParser()
    ap.add_argument("--base-url", default="http://127.0.0.1:8100/v1")
    ap.add_argument("--model", default="qwen35-teacher")
    ap.add_argument(
        "--samples",
        type=Path,
        default=Path("data/processed/philly_samples.jsonl"),
    )
    ap.add_argument("--n-sequential", type=int, default=20)
    ap.add_argument("--n-concurrent", type=int, default=8)
    ap.add_argument("--top-k", type=int, default=50)
    ap.add_argument("--max-new-tokens", type=int, default=64)
    ap.add_argument("--gpu-index", type=int, default=7)
    ap.add_argument(
        "--tokenizer-name",
        default="Qwen/Qwen3.5-35B-A3B",
        help="HF tokenizer to encode prompts (must match served model)",
    )
    args = ap.parse_args()

    from transformers import AutoTokenizer

    print("=== stress_teacher_vllm.py ===")
    print(f"endpoint:   {args.base_url}")
    print(f"model:      {args.model}")
    print(f"top-K:      {args.top_k}")
    print(f"samples:    {args.samples}")
    print(f"gpu index:  {args.gpu_index}")
    print()

    # --- sanity: /v1/models ---
    try:
        r = requests.get(f"{args.base_url}/models", timeout=10)
        r.raise_for_status()
        names = [m["id"] for m in r.json().get("data", [])]
        print(f"/v1/models: {names}")
        if args.model not in names:
            print(f"!! served-model-name {args.model!r} not in {names}")
            return 1
    except Exception as exc:
        print(f"!! /v1/models failed: {exc}")
        return 1

    # --- load samples + tokenizer ---
    tok = AutoTokenizer.from_pretrained(args.tokenizer_name, trust_remote_code=True)

    samples: list[dict] = []
    with args.samples.open() as fh:
        for line in fh:
            samples.append(json.loads(line))
    if not samples:
        print("!! no samples loaded")
        return 1

    encoded: list[tuple[dict, list[int]]] = []
    for s in samples:
        try:
            text = _build_prompt(s)
        except Exception as exc:
            print(f"!! prompt build failed for {s.get('sample_id')}: {exc}")
            continue
        ids = tok.encode(text, add_special_tokens=False)
        encoded.append((s, ids))
    # Sort by length descending so the "hammer" phase leads with longest prompts.
    encoded.sort(key=lambda p: -len(p[1]))
    lens = [len(e[1]) for e in encoded]
    print(
        f"encoded {len(encoded)} prompts — "
        f"min={min(lens)} p50={int(statistics.median(lens))} "
        f"p99={sorted(lens)[int(0.99 * len(lens))]} max={max(lens)}"
    )
    print(f"gpu {args.gpu_index} before: {_gpu_used_mib(args.gpu_index)} MiB")

    failures: list[dict] = []
    all_latencies: list[float] = []

    # --- Phase 1: single long probe ---
    print("\n--- Phase 1: single long prompt probe ---")
    longest_sample, longest_tokens = encoded[0]
    print(f"prompt_len={len(longest_tokens)} sample_id={longest_sample['sample_id']}")
    r1 = _post_completion(
        args.base_url, args.model, longest_tokens, args.top_k, args.max_new_tokens
    )
    all_latencies.append(r1["latency_s"])
    if r1["http_status"] != 200:
        failures.append({"phase": "1-single", **r1})
    print(f"  → {r1}")
    print(f"  gpu after: {_gpu_used_mib(args.gpu_index)} MiB")

    # --- Phase 2: sequential hammer ---
    print(f"\n--- Phase 2: sequential hammer ({args.n_sequential}) ---")
    gpu_peaks = []
    for i, (smp, ids) in enumerate(encoded[: args.n_sequential]):
        r = _post_completion(
            args.base_url, args.model, ids, args.top_k, args.max_new_tokens
        )
        all_latencies.append(r["latency_s"])
        gpu_peaks.append(_gpu_used_mib(args.gpu_index))
        tag = "OK " if r["http_status"] == 200 else "FAIL"
        print(
            f"  [{i+1:02d}/{args.n_sequential}] {tag} "
            f"len={r['prompt_len']:5d} lat={r['latency_s']:6.2f}s "
            f"topK_seen={r['top_k_seen']} gpu={gpu_peaks[-1]} MiB"
            + (f"  ERR={r.get('error','')[:200]}" if r["http_status"] != 200 else "")
        )
        if r["http_status"] != 200:
            failures.append({"phase": "2-sequential", "i": i, **r})

    # --- Phase 3: concurrent burst ---
    print(f"\n--- Phase 3: concurrent burst (width={args.n_concurrent}) ---")
    burst = encoded[: args.n_concurrent]
    t_burst = time.perf_counter()
    with cf.ThreadPoolExecutor(max_workers=args.n_concurrent) as ex:
        futs = [
            ex.submit(
                _post_completion,
                args.base_url,
                args.model,
                ids,
                args.top_k,
                args.max_new_tokens,
            )
            for (_, ids) in burst
        ]
        results = [f.result() for f in cf.as_completed(futs)]
    burst_wall = time.perf_counter() - t_burst
    for i, r in enumerate(results):
        all_latencies.append(r["latency_s"])
        tag = "OK " if r["http_status"] == 200 else "FAIL"
        print(
            f"  [{i+1:02d}] {tag} "
            f"len={r['prompt_len']:5d} lat={r['latency_s']:6.2f}s "
            f"topK_seen={r['top_k_seen']}"
            + (f"  ERR={r.get('error','')[:200]}" if r["http_status"] != 200 else "")
        )
        if r["http_status"] != 200:
            failures.append({"phase": "3-concurrent", "i": i, **r})
    print(
        f"  burst wall={burst_wall:.2f}s  "
        f"gpu after: {_gpu_used_mib(args.gpu_index)} MiB"
    )

    # --- Report ---
    print("\n=== REPORT ===")
    lat_sorted = sorted(all_latencies)
    p50 = lat_sorted[len(lat_sorted) // 2]
    p95 = lat_sorted[int(0.95 * len(lat_sorted))]
    p99 = lat_sorted[min(int(0.99 * len(lat_sorted)), len(lat_sorted) - 1)]
    print(f"requests:     {len(all_latencies)}")
    print(f"failures:     {len(failures)}")
    print(f"latency p50:  {p50:.2f}s")
    print(f"latency p95:  {p95:.2f}s")
    print(f"latency p99:  {p99:.2f}s")
    print(f"gpu peak (seq phase): {max(gpu_peaks) if gpu_peaks else -1} MiB")
    if failures:
        print("\nfailure samples:")
        for f in failures[:5]:
            print(f"  {f}")
    if failures or p99 > 60.0:
        return 1
    print("\nPASS")
    return 0


if __name__ == "__main__":
    sys.exit(main())

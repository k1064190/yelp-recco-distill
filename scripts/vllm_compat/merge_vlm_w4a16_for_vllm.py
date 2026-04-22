#!/usr/bin/env python
# ABOUTME: Merge an already-W4A16-quantized text-only ckpt back into VLM shape for vLLM.
# ABOUTME: Takes base vision + MTP weights (bf16, from cached HF snapshot), adds W4A16 text.

"""Wrap our W4A16 text-only checkpoint in a VLM-shaped container vLLM can load.

Why the companion script to merge_vlm_ckpt_for_vllm.py
------------------------------------------------------
The bf16 path merges only weights; the quant path also needs to merge config
fields. Compressed-tensors stores its recipe in ``quantization_config`` in
``config.json`` and lists modules it should NOT quantize under ``ignore``. For
a VLM shell that carries pretrained (bf16) vision + MTP weights alongside
W4A16 text, ``ignore`` needs to include the vision and MTP module patterns so
vLLM's loader leaves them alone while applying INT4 to the text side.

Inputs
------
    --w4a16-src     existing W4A16 ckpt (e.g. ckpt/student-v2-sft-merged_vllm_vlm_w4a16);
                    its safetensors have 692 `model.language_model.*` tensors
                    (packed int4 + scales + zeros) and its config.json carries
                    the compressed-tensors recipe with ignore=["lm_head"].

    --base-repo     HF repo id for the pretrained VLM (Qwen/Qwen3.5-0.8B).
                    Its snapshot supplies 153 `model.visual.*` + 15 `mtp.*`
                    tensors in bf16 and the VLM-shaped config.

    --dst           output directory. Refuses to overwrite.

Output
------
A VLM-shape directory with:

    - safetensors that contain (692 quantized text) ∪ (153 bf16 vision) ∪ (15 bf16 MTP)
    - config.json derived from base (architectures=Qwen3_5ForConditionalGeneration,
      vision_config present) with the W4A16 quantization_config re-attached and
      `ignore` extended to skip vision + mtp so compressed-tensors doesn't try
      to interpret their bf16 tensors as packed int4.
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

import torch
from safetensors import safe_open
from safetensors.torch import save_file


def load_all_safetensors(src: Path) -> dict[str, torch.Tensor]:
    shards = sorted(src.glob("*.safetensors"))
    if not shards:
        raise FileNotFoundError(f"no .safetensors under {src}")
    state: dict[str, torch.Tensor] = {}
    for shard in shards:
        with safe_open(shard, framework="pt") as f:
            for k in f.keys():
                state[k] = f.get_tensor(k)
    return state


def find_base_snapshot(base_repo: str) -> Path:
    owner, name = base_repo.split("/")
    root = Path.home() / ".cache" / "huggingface" / "hub" / f"models--{owner}--{name}" / "snapshots"
    snaps = sorted(root.iterdir())
    if not snaps:
        raise FileNotFoundError(f"no snapshot under {root}")
    return snaps[-1]


def build_merged_config(base_cfg: dict, w4a16_cfg: dict) -> dict:
    """Return a new config that is base (VLM-shaped) + W4A16 quantization_config.

    Extends ``ignore`` so the compressed-tensors loader leaves the bf16
    vision encoder, MTP head, and embedding/tie modules in place.
    """
    merged = dict(base_cfg)  # shallow copy
    qc = dict(w4a16_cfg["quantization_config"])
    ignore = list(qc.get("ignore") or [])
    # Regex-style patterns so compressed-tensors 0.x+ matches sub-modules.
    for extra in [
        "re:.*visual.*",
        "re:.*mtp.*",
    ]:
        if extra not in ignore:
            ignore.append(extra)
    qc["ignore"] = ignore
    merged["quantization_config"] = qc
    merged.setdefault(
        "_origin_note",
        "VLM shell (base Qwen3.5-0.8B vision + MTP) + W4A16 compressed-tensors "
        "text; config merged via scripts/vllm_compat/merge_vlm_w4a16_for_vllm.py so vLLM "
        "routes through Qwen3_5ForConditionalGeneration while the language "
        "model runs at W4A16.",
    )
    return merged


def merge(w4a16_src: Path, base_repo: str, dst: Path) -> None:
    if dst.exists():
        raise SystemExit(f"destination {dst} already exists; refusing to overwrite")

    base_dir = find_base_snapshot(base_repo)
    print(f"base snapshot: {base_dir}")

    base_state = load_all_safetensors(base_dir)
    w4_state = load_all_safetensors(w4a16_src)
    print(f"base: {len(base_state)} tensors / w4a16: {len(w4_state)} tensors")

    merged_state: dict[str, torch.Tensor] = {}

    # Take quantized text (692 keys) as-is.
    for k, v in w4_state.items():
        merged_state[k] = v

    # Add base's visual + MTP + anything else not already present.
    added = 0
    for k, v in base_state.items():
        if k.startswith("model.language_model."):
            continue  # the language model half lives in the quantized ckpt now
        if k in merged_state:
            continue
        merged_state[k] = v
        added += 1

    print(f"added from base: {added} non-language-model tensors")
    print(f"merged state: {len(merged_state)} tensors")

    # Configs.
    base_cfg = json.loads((base_dir / "config.json").read_text())
    w4a16_cfg = json.loads((w4a16_src / "config.json").read_text())
    merged_cfg = build_merged_config(base_cfg, w4a16_cfg)

    # Emit output.
    dst.mkdir(parents=True, exist_ok=True)
    # Copy tokenizer / chat_template / preprocessor from base.
    for f in base_dir.iterdir():
        if f.name.endswith(".safetensors") or f.name in ("config.json", "model.safetensors.index.json"):
            continue
        if f.is_file():
            shutil.copy(f, dst / f.name)
        else:
            shutil.copytree(f, dst / f.name, dirs_exist_ok=True)
    # Save merged weights + config.
    save_file(merged_state, str(dst / "model.safetensors"))
    (dst / "config.json").write_text(json.dumps(merged_cfg, indent=2, ensure_ascii=False))
    print(f"wrote {dst}")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--w4a16-src", type=Path, required=True,
                   help="existing compressed-tensors W4A16 ckpt (text-only, 692 tensors)")
    p.add_argument("--base-repo", type=str, default="Qwen/Qwen3.5-0.8B")
    p.add_argument("--dst", type=Path, required=True)
    return p.parse_args()


def main() -> int:
    args = parse_args()
    merge(args.w4a16_src, args.base_repo, args.dst)
    return 0


if __name__ == "__main__":
    sys.exit(main())

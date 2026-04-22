#!/usr/bin/env python
# ABOUTME: Merge a bnb NF4-quantized text-only ckpt back into VLM shape for vLLM.
# ABOUTME: Base vision + MTP stay bf16; language model keeps packed NF4 weights.

"""Wrap our NF4 text-only checkpoint in a VLM-shaped container vLLM can load.

Why
---
``quantize_nf4.py`` loads the bf16 VLM via ``AutoModelForCausalLM`` with
``load_in_4bit=True`` and saves a text-only bnb checkpoint. vLLM's Qwen3.5
text-only path is skeletal, so we can't just point vLLM at that
directory. We re-inject base vision + MTP bf16 weights and rewrite the
config so vLLM routes through ``Qwen3_5ForConditionalGeneration`` while
bnb still loads the NF4-packed language model.

Relationship to ``merge_vlm_w4a16_for_vllm.py``
-----------------------------------------------
Same pattern, different quant family. W4A16 uses ``compressed-tensors``
which exposes an ``ignore`` regex list to skip modules. bnb uses
``llm_int8_skip_modules`` (works for both 8-bit and 4-bit despite the
name) — non-regex module path prefixes.

Inputs
------
    --nf4-src       existing bnb-NF4 ckpt produced by ``quantize_nf4.py``.
                    Text-only (architectures=Qwen3_5ForCausalLM) with
                    ``quantization_config._load_in_4bit = true``.
    --base-repo     HF repo id for the pretrained VLM (Qwen/Qwen3.5-0.8B
                    or Qwen/Qwen3.5-9B). Its snapshot supplies the
                    ``model.visual.*`` + ``mtp.*`` tensors + the VLM-shaped
                    config with ``vision_config``.
    --dst           output directory. Refuses to overwrite.

Output
------
A VLM-shape directory with:
    - safetensors: (language model NF4 packed) ∪ (base visual bf16) ∪
      (base MTP bf16) + any other non-LM tensors from base.
    - config.json: architectures=``Qwen3_5ForConditionalGeneration`` +
      vision_config + ``quantization_config`` with NF4 flags and
      ``llm_int8_skip_modules`` extended to [``visual``, ``mtp``,
      ``lm_head``] so bnb leaves the bf16 vision/MTP alone.
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
    """Flatten every shard under ``src`` into a single state dict.

    Args:
        src (Path): directory containing one or more ``*.safetensors`` shards.

    Returns:
        dict[str, torch.Tensor]: concatenated parameter dict.
    """
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
    """Return the path of the most recent HF snapshot for ``base_repo``."""
    owner, name = base_repo.split("/")
    root = Path.home() / ".cache" / "huggingface" / "hub" / f"models--{owner}--{name}" / "snapshots"
    snaps = sorted(root.iterdir())
    if not snaps:
        raise FileNotFoundError(f"no snapshot under {root}")
    return snaps[-1]


def build_merged_config(base_cfg: dict, nf4_cfg: dict) -> dict:
    """Return a new config: base (VLM-shaped) + bnb NF4 quantization_config.

    Extends ``llm_int8_skip_modules`` so bnb's loader leaves the bf16
    vision encoder + MTP head + lm_head in place. ``llm_int8_skip_modules``
    applies to both 8-bit and 4-bit despite the name.
    """
    merged = dict(base_cfg)
    qc = dict(nf4_cfg["quantization_config"])
    skip = list(qc.get("llm_int8_skip_modules") or [])
    for extra in ["visual", "mtp", "lm_head"]:
        if extra not in skip:
            skip.append(extra)
    qc["llm_int8_skip_modules"] = skip
    merged["quantization_config"] = qc
    merged.setdefault(
        "_origin_note",
        "VLM shell (base vision + MTP in bf16) + bnb NF4 language model. "
        "Produced by scripts/vllm_compat/merge_vlm_nf4_for_vllm.py so vLLM "
        "routes through Qwen3_5ForConditionalGeneration while bnb loads "
        "the packed NF4 language model.",
    )
    return merged


def merge(nf4_src: Path, base_repo: str, dst: Path) -> None:
    """Merge NF4 text weights with base vision + MTP into a VLM shell."""
    if dst.exists():
        raise SystemExit(f"destination {dst} already exists; refusing to overwrite")

    base_dir = find_base_snapshot(base_repo)
    print(f"base snapshot: {base_dir}")

    base_state = load_all_safetensors(base_dir)
    nf4_state = load_all_safetensors(nf4_src)
    print(f"base: {len(base_state)} tensors / nf4: {len(nf4_state)} tensors")

    merged_state: dict[str, torch.Tensor] = {}

    # Text (language model) comes from the NF4 ckpt. bnb saves both packed
    # weights and companion `absmax`, `quant_map` etc under the same prefix;
    # include everything from the NF4 side as-is.
    for k, v in nf4_state.items():
        merged_state[k] = v

    # Pull in base's vision + MTP + any other non-language-model tensors.
    added = 0
    for k, v in base_state.items():
        if k.startswith("model.language_model."):
            continue
        if k in merged_state:
            continue
        merged_state[k] = v
        added += 1

    print(f"added from base: {added} non-language-model tensors")
    print(f"merged state: {len(merged_state)} tensors")

    base_cfg = json.loads((base_dir / "config.json").read_text())
    nf4_cfg = json.loads((nf4_src / "config.json").read_text())
    merged_cfg = build_merged_config(base_cfg, nf4_cfg)

    dst.mkdir(parents=True, exist_ok=True)
    for f in base_dir.iterdir():
        if f.name.endswith(".safetensors") or f.name in ("config.json", "model.safetensors.index.json"):
            continue
        if f.is_file():
            shutil.copy(f, dst / f.name)
        else:
            shutil.copytree(f, dst / f.name, dirs_exist_ok=True)
    save_file(merged_state, str(dst / "model.safetensors"))
    (dst / "config.json").write_text(json.dumps(merged_cfg, indent=2, ensure_ascii=False))
    print(f"wrote {dst}")


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments."""
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--nf4-src", type=Path, required=True,
                   help="existing bnb NF4 text-only ckpt from quantize_nf4.py")
    p.add_argument("--base-repo", type=str, default="Qwen/Qwen3.5-0.8B",
                   help="HF repo id of the pretrained VLM")
    p.add_argument("--dst", type=Path, required=True)
    return p.parse_args()


def main() -> int:
    """Entry point: run the merge with the given CLI args."""
    args = parse_args()
    merge(args.nf4_src, args.base_repo, args.dst)
    return 0


if __name__ == "__main__":
    sys.exit(main())

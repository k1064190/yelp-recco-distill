#!/usr/bin/env python
# ABOUTME: Merge fine-tuned text weights into base Qwen3.5 VLM checkpoint so vLLM
# ABOUTME: can load the result through its primary Qwen3_5ForConditionalGeneration path.

"""Stitch our text-only SFT checkpoint back into the VLM shape vLLM expects.

Why
---
Qwen3.5 ships as a VLM (``Qwen3_5ForConditionalGeneration``). vLLM's primary
path for this family — and the only one with complete support for the hybrid
linear_attention + full_attention KV cache layout — routes through the VLM
class. The text-only ``Qwen3_5ForCausalLM`` class in vLLM is a skeleton: no
``IsHybrid`` inheritance, not in the default ModelRegistry, and rejected by
the ``model_impl=transformers`` fallback. See logs/infer_vllm_fp16.log for
the cascade of errors when we try.

Our full-FT run loaded the base VLM as ``AutoModelForCausalLM``, trained on
text, and saved only the text half (320 tensors, ``model.language_model.*``).
That's a standalone ``Qwen3_5ForCausalLM`` checkpoint which vLLM can't load.

This script puts Humpty Dumpty back together:

    base Qwen/Qwen3.5-0.8B safetensors:
        320  model.language_model.*   (pretrained text)
        153  model.visual.*           (pretrained vision encoder)
         15  mtp.*                    (multi-token prediction head)
        ──
        488 total

    our ckpt (student-v2-sft-merged):
        320  model.language_model.*   (fine-tuned text)

Merge rule: copy base → overwrite any ``model.language_model.*`` key with the
fine-tuned value. The result has all 488 tensors, 320 of them fine-tuned and
the rest pretrained. The vision encoder stays dead weight at text-only
inference time (no ``pixel_values`` in the input), but its presence is what
makes the VLM loader happy.

Config: copied from base (declares ``Qwen3_5ForConditionalGeneration`` +
``vision_config``).

Disk cost: base model.safetensors is ~1.6 GB for Qwen3.5-0.8B (the visual
encoder adds only a few hundred MB to the text weights). Acceptable.
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path
from typing import Any

import torch
from safetensors import safe_open
from safetensors.torch import save_file


def load_all_safetensors(src: Path) -> dict[str, torch.Tensor]:
    """Load every .safetensors shard in ``src`` into one ``{key: tensor}`` dict."""
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
    """Locate the HF snapshot dir for ``base_repo`` in the user's HF cache."""
    owner, name = base_repo.split("/")
    root = Path.home() / ".cache" / "huggingface" / "hub" / f"models--{owner}--{name}" / "snapshots"
    snaps = sorted(root.iterdir())
    if not snaps:
        raise FileNotFoundError(
            f"no snapshots under {root}; download the base with `huggingface-cli "
            f"download {base_repo}` first."
        )
    return snaps[-1]  # most recent


def copy_non_weight_files(src: Path, dst: Path) -> None:
    """Copy every file in ``src`` except safetensors + index to ``dst``.

    Used to bring base's config.json / tokenizer / chat_template into the
    merged dir. The config.json from base is what makes vLLM route to the
    VLM class (architectures=Qwen3_5ForConditionalGeneration + vision_config).
    """
    dst.mkdir(parents=True, exist_ok=True)
    for f in src.iterdir():
        if f.name.endswith(".safetensors") or f.name == "model.safetensors.index.json":
            continue
        if f.is_file():
            shutil.copy(f, dst / f.name)
        else:
            shutil.copytree(f, dst / f.name, dirs_exist_ok=True)


def merge(base_repo: str, ft_src: Path, dst: Path) -> None:
    """Perform the VLM re-hydration described in the module docstring.

    Args:
        base_repo (str): HF repo id (e.g. ``"Qwen/Qwen3.5-0.8B"``) whose cached
            snapshot supplies vision + MTP weights and the VLM config.
        ft_src (Path): fine-tuned text-only checkpoint dir (our SFT output).
        dst (Path): destination directory; must not already exist.
    """
    if dst.exists():
        raise SystemExit(f"destination {dst} already exists; refusing to overwrite")

    base_dir = find_base_snapshot(base_repo)
    print(f"base snapshot: {base_dir}")

    base_state = load_all_safetensors(base_dir)
    ft_state = load_all_safetensors(ft_src)
    print(f"base: {len(base_state)} tensors / ft: {len(ft_state)} tensors")

    overwritten = 0
    missing_from_ft: list[str] = []
    merged: dict[str, torch.Tensor] = {}

    for k, v_base in base_state.items():
        if k in ft_state:
            merged[k] = ft_state[k]
            overwritten += 1
        elif k.startswith("model.language_model."):
            # A language_model.* key that the fine-tune did not save (e.g. because
            # AutoModelForCausalLM dropped something). Keep the base value but
            # record it.
            missing_from_ft.append(k)
            merged[k] = v_base
        else:
            merged[k] = v_base

    extra_in_ft = set(ft_state) - set(base_state)
    if extra_in_ft:
        print(f"WARN: {len(extra_in_ft)} fine-tuned keys not in base — adding as-is")
        for k in extra_in_ft:
            merged[k] = ft_state[k]

    copy_non_weight_files(base_dir, dst)
    save_file(merged, str(dst / "model.safetensors"))

    print(f"merged: {len(merged)} tensors → {dst/'model.safetensors'}")
    print(f"  overwritten with fine-tuned weights: {overwritten}")
    print(f"  kept from base (missing in ft):      {len(missing_from_ft)}")
    print(f"  extra from ft (not in base):          {len(extra_in_ft)}")
    if missing_from_ft[:3]:
        print(f"  sample missing-from-ft: {missing_from_ft[:3]}")


def parse_args() -> argparse.Namespace:
    """Parse CLI args."""
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--base-repo", type=str, default="Qwen/Qwen3.5-0.8B",
                   help="HF repo id for the base VLM checkpoint (default Qwen/Qwen3.5-0.8B)")
    p.add_argument("--ft-src", type=Path, required=True,
                   help="fine-tuned text-only ckpt dir (e.g. ckpt/student-v2-sft-merged)")
    p.add_argument("--dst", type=Path, required=True,
                   help="destination merged ckpt dir")
    return p.parse_args()


def main() -> int:
    args = parse_args()
    merge(args.base_repo, args.ft_src, args.dst)
    return 0


if __name__ == "__main__":
    sys.exit(main())

#!/usr/bin/env python
# ABOUTME: Remap an HF-saved Qwen3.5 Text checkpoint to the weight names vLLM's
# ABOUTME: Qwen3_5ForCausalLM class expects. Writes a sibling ckpt dir, never overwrites.

"""Translate an HF-saved Qwen3.5 checkpoint into the name scheme vLLM expects.

Why this exists
---------------

The base Qwen/Qwen3.5-0.8B ships as a VLM (``Qwen3_5ForConditionalGeneration``),
so its safetensors use the ``model.language_model.…`` prefix and keep the
Gated DeltaNet projections split into ``in_proj_a``, ``in_proj_b``,
``in_proj_qkv``, ``in_proj_z``. When we fine-tune via
``AutoModelForCausalLM``, HF saves a text-only checkpoint with architecture
``Qwen3_5ForCausalLM`` — but the weight names stay in the VLM form.

vLLM 0.19.1rc1's ``Qwen3_5ForCausalLM`` class expects the prefix-free names
but leaves fusion to its OWN weight loader (``stacked_params_mapping`` in
``qwen3_5.py`` lines 286-301 declares ``in_proj_b`` / ``in_proj_a`` as the
checkpoint-side names that get packed into ``in_proj_ba`` internally; same
for ``in_proj_qkv`` + ``in_proj_z`` → ``in_proj_qkvz``). So the only shim
this script does is drop the ``model.language_model.`` prefix:

    model.language_model.layers.N.linear_attn.in_proj_a.weight  (HF)
        → model.layers.N.linear_attn.in_proj_a.weight           (vLLM)
    model.language_model.layers.N.linear_attn.in_proj_b.weight
        → model.layers.N.linear_attn.in_proj_b.weight
    (same for in_proj_qkv / in_proj_z / every other layer tensor)

The Qwen3_5ForConditionalGeneration (VLM) loader wraps the text model under
``language_model`` and strips that prefix internally. The pure ``ForCausalLM``
loader does not. This script does the strip offline, once, and writes a
sibling ``<orig>_vllm_rename/`` directory so the original checkpoint is
untouched.

Supported source checkpoints
----------------------------

- bf16 merged (``ckpt/student-v2-sft-merged``): full weights, concat works trivially.
- NF4 (bitsandbytes saved dir): concat of packed 4-bit tensors is not meaningful;
  this script refuses NF4. Inference NF4 via vLLM needs a separate re-quant
  pass.
- W4A16 (compressed-tensors): same — packed int4 + scales can't be concatenated
  without de-quantizing first. Refused.

So this is only a bf16 rename path. Follow-up: re-quantize from the renamed
bf16 into vLLM-compatible W4A16 / NF4 inside the matching environment if needed.
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


HF_TO_VLLM_PREFIX_STRIP = "model.language_model."
VLLM_PREFIX = "model."


def load_all_safetensors(src: Path) -> dict[str, torch.Tensor]:
    """Load every .safetensors shard in ``src`` into one ``{key: tensor}`` dict.

    Args:
        src (Path): source checkpoint directory.

    Returns:
        dict[str, torch.Tensor]: merged state dict.
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


def strip_language_model_prefix(state: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    """Rename ``model.language_model.X`` → ``model.X`` (everything else untouched).

    Args:
        state (dict): input state dict.

    Returns:
        dict: new dict with keys rewritten in place (new object).
    """
    out: dict[str, torch.Tensor] = {}
    for k, v in state.items():
        if k.startswith(HF_TO_VLLM_PREFIX_STRIP):
            new_k = VLLM_PREFIX + k[len(HF_TO_VLLM_PREFIX_STRIP) :]
        else:
            new_k = k
        out[new_k] = v
    return out


def fuse_gated_deltanet(state: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    """Fuse ``in_proj_b + in_proj_a`` → ``in_proj_ba`` and ``in_proj_qkv + in_proj_z`` → ``in_proj_qkvz``.

    Concat along dim 0. Order matches vLLM's ``packed_modules_mapping`` in
    ``qwen3_5.py``:
        in_proj_ba   = cat([in_proj_b, in_proj_a], dim=0)
        in_proj_qkvz = cat([in_proj_qkv, in_proj_z], dim=0)

    Args:
        state (dict): input state dict.

    Returns:
        dict: new dict with fused keys; original components removed.
    """
    out: dict[str, torch.Tensor] = dict(state)
    fused_ba = 0
    fused_qkvz = 0

    # Find layer prefixes by scanning for in_proj_a.
    prefixes_ba: set[str] = set()
    for k in list(out.keys()):
        if k.endswith(".linear_attn.in_proj_a.weight"):
            prefixes_ba.add(k[: -len(".in_proj_a.weight")])

    for prefix in sorted(prefixes_ba):
        a_key = f"{prefix}.in_proj_a.weight"
        b_key = f"{prefix}.in_proj_b.weight"
        ba_key = f"{prefix}.in_proj_ba.weight"
        if a_key in out and b_key in out:
            a = out.pop(a_key)
            b = out.pop(b_key)
            out[ba_key] = torch.cat([b, a], dim=0)
            fused_ba += 1

    prefixes_qkvz: set[str] = set()
    for k in list(out.keys()):
        if k.endswith(".linear_attn.in_proj_qkv.weight"):
            prefixes_qkvz.add(k[: -len(".in_proj_qkv.weight")])

    for prefix in sorted(prefixes_qkvz):
        qkv_key = f"{prefix}.in_proj_qkv.weight"
        z_key = f"{prefix}.in_proj_z.weight"
        qkvz_key = f"{prefix}.in_proj_qkvz.weight"
        if qkv_key in out and z_key in out:
            qkv = out.pop(qkv_key)
            z = out.pop(z_key)
            out[qkvz_key] = torch.cat([qkv, z], dim=0)
            fused_qkvz += 1

    print(f"fused {fused_ba} in_proj_ba pairs, {fused_qkvz} in_proj_qkvz pairs")
    return out


def copy_aux_files(src: Path, dst: Path) -> None:
    """Copy everything except safetensors shards + index from ``src`` to ``dst``."""
    dst.mkdir(parents=True, exist_ok=True)
    for f in src.iterdir():
        if f.name.endswith(".safetensors"):
            continue
        if f.name == "model.safetensors.index.json":
            continue
        if f.is_file():
            shutil.copy(f, dst / f.name)
        else:
            shutil.copytree(f, dst / f.name, dirs_exist_ok=True)


def rewrite_config(dst: Path) -> None:
    """Adjust config.json for vLLM compatibility.

    vLLM's ``Qwen3_5ForCausalLM`` class is happy with the already-declared
    ``architectures: ["Qwen3_5ForCausalLM"]``, but we also drop any
    ``torch_dtype`` coercion mismatch and mark the origin for debugging.
    """
    cfg_path = dst / "config.json"
    if not cfg_path.exists():
        return
    cfg = json.loads(cfg_path.read_text())
    cfg.setdefault("architectures", ["Qwen3_5ForCausalLM"])
    cfg.setdefault(
        "_origin_note",
        "Renamed for vLLM via scripts/vllm_compat/rename_ckpt_for_vllm.py "
        "(fused in_proj_ba / in_proj_qkvz, stripped model.language_model. prefix).",
    )
    cfg_path.write_text(json.dumps(cfg, indent=2, ensure_ascii=False))


def rename_bf16_checkpoint(src: Path, dst: Path) -> None:
    """Full rename pipeline for a bf16 merged checkpoint.

    Refuses quantized source directories (W4A16 / NF4): packed int4 weights
    and their scale tensors cannot be meaningfully concatenated without a
    round-trip dequantization pass, so those variants should be produced by
    re-quantizing the renamed bf16 output, not by renaming the quantized
    tensors directly.

    Args:
        src (Path): source ckpt directory.
        dst (Path): destination ckpt directory (must not already exist).
    """
    # Guard against quantized sources.
    config = json.loads((src / "config.json").read_text())
    qcfg = config.get("quantization_config")
    if qcfg is not None:
        raise SystemExit(
            f"{src} is a {qcfg.get('quant_method', '?')} quantized checkpoint — "
            "this script only handles bf16. Re-quantize from the renamed bf16 if "
            "you need a vLLM-compatible W4A16/NF4."
        )

    if dst.exists():
        raise SystemExit(f"destination {dst} already exists; refusing to overwrite")

    state = load_all_safetensors(src)
    print(f"loaded {len(state)} tensors from {src}")

    state = strip_language_model_prefix(state)
    # NB: we intentionally do NOT fuse in_proj_a/b → in_proj_ba here. vLLM's
    # Qwen3_5ForCausalLM weight loader expects the separate names in the
    # checkpoint and stacks them into the fused parameter at load time
    # (see qwen3_5.py stacked_params_mapping). Fusing here would trigger
    # "Following weights were not initialized from checkpoint" on the
    # separate entries vLLM is looking for.
    print(f"final state has {len(state)} tensors")

    copy_aux_files(src, dst)
    save_file(state, str(dst / "model.safetensors"))
    rewrite_config(dst)
    print(f"wrote {dst}")


def parse_args() -> argparse.Namespace:
    """Parse CLI args."""
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--src", type=Path, required=True, help="source HF ckpt directory")
    p.add_argument("--dst", type=Path, required=True, help="destination ckpt directory")
    return p.parse_args()


def main() -> int:
    args = parse_args()
    rename_bf16_checkpoint(args.src, args.dst)
    return 0


if __name__ == "__main__":
    sys.exit(main())

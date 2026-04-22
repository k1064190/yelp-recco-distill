#!/usr/bin/env python
# ABOUTME: Convert merged Qwen3-4B student from HF format to GGUF and quantize
# ABOUTME: to Q4_K_M using llama.cpp tools (convert_hf_to_gguf.py + llama-quantize).

"""
Convert the merged Qwen3-4B student to GGUF format and quantize to Q4_K_M.

Two-step process:
  1. ``convert_hf_to_gguf.py`` (from llama.cpp repo) converts the HF
     SafeTensors checkpoint into a GGUF F16 file.
  2. ``llama-quantize`` (compiled from llama.cpp) applies the Q4_K_M
     quantization scheme (4-bit with k-quant medium, good balance of
     quality and compression).

The GGUF format is the native format for llama.cpp inference and is widely
supported by consumer-oriented LLM runtimes (Ollama, LM Studio, etc.).

Prerequisites:
  - llama.cpp cloned at /workspace/projects/llama.cpp
  - llama-quantize binary built at /workspace/projects/llama.cpp/build/bin/llama-quantize
  - gguf Python package installed (pip install gguf)

Example:
    $ python scripts/quantize/quantize_gguf.py \\
        --model ckpt/student-merged \\
        --output ckpt/student-gguf-q4km \\
        --quant-type Q4_K_M
"""

from __future__ import annotations

import argparse
import logging
import shutil
import subprocess
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
LLAMA_CPP_DIR = Path("/workspace/projects/llama.cpp")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("quantize_gguf")


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments.

    Returns:
        argparse.Namespace: parsed arguments with model, output, quant_type,
            llama_cpp_dir, and python_bin fields.
    """
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--model",
        type=Path,
        default=PROJECT_ROOT / "ckpt/student-merged",
        help="LoRA-merged Qwen3-4B checkpoint to convert",
    )
    p.add_argument(
        "--output",
        type=Path,
        default=PROJECT_ROOT / "ckpt/student-gguf-q4km",
        help="output directory for GGUF files",
    )
    p.add_argument(
        "--quant-type",
        type=str,
        default="Q4_K_M",
        help="GGUF quantization type (default Q4_K_M)",
    )
    p.add_argument(
        "--llama-cpp-dir",
        type=Path,
        default=LLAMA_CPP_DIR,
        help="path to llama.cpp repository root",
    )
    p.add_argument(
        "--python-bin",
        type=Path,
        default=Path(sys.executable),
        help="Python binary for running convert_hf_to_gguf.py",
    )
    return p.parse_args()


def get_disk_size_mb(path: Path) -> float:
    """Compute total size of all files under a directory or a single file in MB.

    Args:
        path (Path): directory or file to measure.

    Returns:
        float: total size in megabytes.
    """
    if path.is_file():
        return path.stat().st_size / (1024 * 1024)
    total = sum(f.stat().st_size for f in path.rglob("*") if f.is_file())
    return total / (1024 * 1024)


def main() -> int:
    """Orchestrate HF->GGUF F16 conversion and Q4_K_M quantization.

    Returns:
        int: exit code (0 = success, 2 = input/tool error).
    """
    args = parse_args()

    if not args.model.exists():
        log.error("merged model path not found: %s", args.model)
        return 2

    convert_script = args.llama_cpp_dir / "convert_hf_to_gguf.py"
    quantize_bin = args.llama_cpp_dir / "build" / "bin" / "llama-quantize"

    if not convert_script.exists():
        log.error("convert_hf_to_gguf.py not found at %s", convert_script)
        return 2
    if not quantize_bin.exists():
        log.error(
            "llama-quantize binary not found at %s — build llama.cpp first",
            quantize_bin,
        )
        return 2

    args.output.mkdir(parents=True, exist_ok=True)

    # ---- Step 1: HF → GGUF F16 ------------------------------------------------
    f16_gguf = args.output / "student-f16.gguf"
    log.info("step 1: converting HF model to GGUF F16 → %s", f16_gguf)

    convert_cmd = [
        str(args.python_bin),
        str(convert_script),
        str(args.model),
        "--outfile", str(f16_gguf),
        "--outtype", "f16",
    ]
    log.info("running: %s", " ".join(convert_cmd))
    result = subprocess.run(
        convert_cmd,
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        log.error("convert_hf_to_gguf.py failed (exit %d)", result.returncode)
        log.error("stderr:\n%s", result.stderr[-2000:] if result.stderr else "(empty)")
        return 2

    f16_size_mb = get_disk_size_mb(f16_gguf)
    log.info("GGUF F16 created: %.1f MB (%.2f GB)", f16_size_mb, f16_size_mb / 1024)

    # ---- Step 2: GGUF F16 → quantized -----------------------------------------
    quant_gguf = args.output / f"student-{args.quant_type.lower().replace('_', '-')}.gguf"
    log.info(
        "step 2: quantizing GGUF F16 → %s (%s)", quant_gguf, args.quant_type
    )

    quantize_cmd = [
        str(quantize_bin),
        str(f16_gguf),
        str(quant_gguf),
        args.quant_type,
    ]
    log.info("running: %s", " ".join(quantize_cmd))
    result = subprocess.run(
        quantize_cmd,
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        log.error("llama-quantize failed (exit %d)", result.returncode)
        log.error("stderr:\n%s", result.stderr[-2000:] if result.stderr else "(empty)")
        return 2

    quant_size_mb = get_disk_size_mb(quant_gguf)
    log.info(
        "%s GGUF created: %.1f MB (%.2f GB)",
        args.quant_type,
        quant_size_mb,
        quant_size_mb / 1024,
    )

    # ---- Cleanup: remove F16 intermediate to save disk -------------------------
    log.info("removing F16 intermediate (%s) to save disk", f16_gguf)
    f16_gguf.unlink(missing_ok=True)

    # ---- Summary ---------------------------------------------------------------
    input_size_mb = get_disk_size_mb(args.model)
    log.info(
        "summary: input=%.2f GB, %s=%.2f GB, compression=%.1fx",
        input_size_mb / 1024,
        args.quant_type,
        quant_size_mb / 1024,
        input_size_mb / max(quant_size_mb, 1),
    )
    log.info("GGUF quantization complete: %s", quant_gguf)
    return 0


if __name__ == "__main__":
    sys.exit(main())

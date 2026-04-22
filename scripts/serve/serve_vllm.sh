#!/usr/bin/env bash
# ABOUTME: Launch the FP16 and W4A16 Qwen3-4B student vLLM servers in the
# ABOUTME: background, each pinned to its own GPU, with latency-optimized flags.

# Plan Key Design Decisions:
#
#   * TP=1 for both. Qwen3-4B fits on a single RTX 3090, and TP>1 adds
#     per-layer all-reduce overhead that hurts single-request latency.
#   * --max-num-seqs 1 prevents continuous-batching from piggybacking on
#     the benchmark's serial calls (we want per-request latency, not
#     throughput under load).
#   * --enforce-eager disables CUDA graph capture so warmup noise is
#     consistent across FP16 and W4A16 runs — the same flag must be set
#     on both servers or the comparison is unfair.
#   * The W4A16 server runs with `--dtype float16` because the
#     compressed-tensors scheme requires fp16 compute dtype; the FP16
#     server uses bf16 because that's what training produced.
#
# Usage:
#
#   bash scripts/serve/serve_vllm.sh              # launch both, write logs/vllm-*.log
#   bash scripts/serve/serve_vllm.sh stop         # stop any running vLLM servers
#   bash scripts/serve/serve_vllm.sh status       # ping both ports
#
# Expected successful startup takes ~60 s per server (weight load +
# CUDA kernel compile). Check logs/vllm-fp16.log and logs/vllm-w4a16.log
# for "Uvicorn running on" before running the benchmark or judge.

set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_ROOT"

# Force FLASH_ATTN backend to avoid the flashinfer decode-wrapper
# assertion triggered by compressed-tensors W4A16 on Ampere (SM 8.6).
# Hard override (unconditional export, not `:-` default) because the
# shell may already export FLASHINFER globally which would trip the
# assertion during the first decode. Keeping the same backend family on
# the FP16 server too keeps the latency benchmark apples-to-apples.
# See the portfolio README.
export VLLM_ATTENTION_BACKEND=FLASH_ATTN

VLLM="vllm"
LOG_DIR="$PROJECT_ROOT/logs"
PID_DIR="$PROJECT_ROOT/logs/.pids"
mkdir -p "$LOG_DIR" "$PID_DIR"

FP16_MODEL="${FP16_MODEL:-$PROJECT_ROOT/ckpt/student-merged}"
W4A16_MODEL="${W4A16_MODEL:-$PROJECT_ROOT/ckpt/student-w4a16}"
FP16_PORT="${FP16_PORT:-8000}"
W4A16_PORT="${W4A16_PORT:-8001}"
FP16_GPU="${FP16_GPU:-0}"
W4A16_GPU="${W4A16_GPU:-1}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-8192}"

start_fp16() {
    if [[ ! -d "$FP16_MODEL" ]]; then
        echo "ERROR: FP16 model dir not found: $FP16_MODEL" >&2
        return 1
    fi
    echo "starting FP16 server on GPU $FP16_GPU port $FP16_PORT ..."
    CUDA_VISIBLE_DEVICES="$FP16_GPU" nohup "$VLLM" serve "$FP16_MODEL" \
        --dtype bfloat16 \
        --tensor-parallel-size 1 \
        --gpu-memory-utilization 0.80 \
        --max-model-len "$MAX_MODEL_LEN" \
        --max-num-seqs 1 \
        --enforce-eager \
        --port "$FP16_PORT" \
        --served-model-name student-fp16 \
        > "$LOG_DIR/vllm-fp16.log" 2>&1 &
    echo $! > "$PID_DIR/vllm-fp16.pid"
    echo "  pid=$(cat "$PID_DIR/vllm-fp16.pid") log=$LOG_DIR/vllm-fp16.log"
}

start_w4a16() {
    if [[ ! -d "$W4A16_MODEL" ]]; then
        echo "ERROR: W4A16 model dir not found: $W4A16_MODEL" >&2
        return 1
    fi
    echo "starting W4A16 server on GPU $W4A16_GPU port $W4A16_PORT ..."
    CUDA_VISIBLE_DEVICES="$W4A16_GPU" nohup "$VLLM" serve "$W4A16_MODEL" \
        --quantization compressed-tensors \
        --dtype float16 \
        --tensor-parallel-size 1 \
        --gpu-memory-utilization 0.60 \
        --max-model-len "$MAX_MODEL_LEN" \
        --max-num-seqs 1 \
        --enforce-eager \
        --port "$W4A16_PORT" \
        --served-model-name student-w4a16 \
        > "$LOG_DIR/vllm-w4a16.log" 2>&1 &
    echo $! > "$PID_DIR/vllm-w4a16.pid"
    echo "  pid=$(cat "$PID_DIR/vllm-w4a16.pid") log=$LOG_DIR/vllm-w4a16.log"
}

stop_one() {
    local name="$1"
    local pidfile="$PID_DIR/vllm-$name.pid"
    if [[ ! -f "$pidfile" ]]; then
        echo "$name: no pid file, nothing to stop"
        return 0
    fi
    local pid
    pid="$(cat "$pidfile")"
    if kill -0 "$pid" 2>/dev/null; then
        echo "stopping $name (pid $pid) ..."
        kill "$pid" 2>/dev/null || true
        sleep 1
        if kill -0 "$pid" 2>/dev/null; then
            kill -9 "$pid" 2>/dev/null || true
        fi
    else
        echo "$name: pid $pid not running"
    fi
    rm -f "$pidfile"
}

status_one() {
    local name="$1"
    local port="$2"
    local url="http://localhost:$port/v1/models"
    local out
    if out=$(curl --silent --max-time 2 "$url" 2>&1); then
        if echo "$out" | grep -q '"id"'; then
            echo "$name: UP   ($url)"
            return 0
        fi
    fi
    echo "$name: DOWN ($url)"
    return 1
}

case "${1:-start}" in
    start)
        start_fp16
        start_w4a16
        echo
        echo "both launched. tail logs to watch startup:"
        echo "  tail -F $LOG_DIR/vllm-fp16.log  $LOG_DIR/vllm-w4a16.log"
        ;;
    stop)
        stop_one fp16
        stop_one w4a16
        ;;
    status)
        status_one fp16  "$FP16_PORT"
        status_one w4a16 "$W4A16_PORT"
        ;;
    *)
        echo "usage: $0 {start|stop|status}" >&2
        exit 2
        ;;
esac

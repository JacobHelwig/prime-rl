#!/usr/bin/env bash

set -euo pipefail

DATA_PATH=/scratch/user/jacob.a.helwig_tamu.edu/primeRL_data

export HF_HOME=$DATA_PATH
export TRITON_CACHE_DIR=$DATA_PATH/triton_cache


MODEL="${MODEL:-Qwen/Qwen3-VL-2B-Instruct}"
PORT="${PORT:-8000}"
SERVER_GPU="${SERVER_GPU:-0}"
COMPARE_GPU="${COMPARE_GPU:-1}"
LOG_DIR="${LOG_DIR:-/scratch/user/jacob.a.helwig_tamu.edu/prime-rl/logs/vlm_suffix_test}"
TIMEOUT_SECONDS="${TIMEOUT_SECONDS:-300}"

mkdir -p "$LOG_DIR"

SERVER_LOG="$LOG_DIR/server.log"
COMPARE_LOG="$LOG_DIR/compare.log"

cleanup() {
  if [[ -n "${SERVER_PID:-}" ]] && kill -0 "$SERVER_PID" 2>/dev/null; then
    kill "$SERVER_PID" 2>/dev/null || true
    wait "$SERVER_PID" 2>/dev/null || true
  fi
}

trap cleanup EXIT

echo "Starting inference server"
echo "  model: $MODEL"
echo "  port: $PORT"
echo "  server_gpu: $SERVER_GPU"
echo "  compare_gpu: $COMPARE_GPU"
echo "  log_dir: $LOG_DIR"

CUDA_VISIBLE_DEVICES="$SERVER_GPU" \
uv run inference @ configs/debug/infer.toml \
  --model.name "$MODEL" \
  --server.port "$PORT" \
  >"$SERVER_LOG" 2>&1 &
SERVER_PID=$!

echo "Waiting for server health check"
SECONDS_WAITED=0
until curl -fsS "http://127.0.0.1:${PORT}/health" >/dev/null 2>&1; do
  if ! kill -0 "$SERVER_PID" 2>/dev/null; then
    echo "Inference server exited early. Log:"
    cat "$SERVER_LOG"
    exit 1
  fi
  if (( SECONDS_WAITED >= TIMEOUT_SECONDS )); then
    echo "Timed out waiting for server after ${TIMEOUT_SECONDS}s. Log:"
    cat "$SERVER_LOG"
    exit 1
  fi
  sleep 2
  SECONDS_WAITED=$((SECONDS_WAITED + 2))
done

echo "Server is healthy"

CUDA_VISIBLE_DEVICES="$COMPARE_GPU" \
uv run python scripts/compare_vlm_prefill_logprobs.py \
  --base-url "http://127.0.0.1:${PORT}" \
  --model "$MODEL" \
  >"$COMPARE_LOG" 2>&1

echo
echo "Comparison output:"
cat "$COMPARE_LOG"
echo
echo "Logs:"
echo "  server:  $SERVER_LOG"
echo "  compare: $COMPARE_LOG"

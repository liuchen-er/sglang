#!/usr/bin/env bash
set -euo pipefail

source /root/autodl-tmp/venvs/sglang015/bin/activate
source "$(dirname "$0")/env.sh"

CHUNK_SIZES=(512 1024 2048 4096)
REPEATS="${REPEATS:-1}"

SERVER_PID=""

stop_server() {
  if [[ -n "${SERVER_PID}" ]]; then
    kill -TERM -- "-${SERVER_PID}" 2>/dev/null || true
    wait "${SERVER_PID}" 2>/dev/null || true
    SERVER_PID=""
  fi
}

trap stop_server EXIT

for chunk in "${CHUNK_SIZES[@]}"; do
  for repeat in $(seq 1 "${REPEATS}"); do
    RESULT_DIR="${PROJECT_ROOT}/results/fixed_chunk/chunk_${chunk}/run_${repeat}"
    mkdir -p "${RESULT_DIR}"

    echo "=================================================="
    echo "Chunk size: ${chunk}"
    echo "Repeat: ${repeat}"
    echo "=================================================="

    setsid env \
      CHUNK_SIZE="${chunk}" \
      ADAPTIVE=0 \
      bash "${PROJECT_ROOT}/scripts/launch_server.sh" \
      > "${RESULT_DIR}/server.log" 2>&1 &

    SERVER_PID=$!

    bash "${PROJECT_ROOT}/scripts/wait_server.sh"
    sleep 8

    bash "${PROJECT_ROOT}/scripts/bench_case.sh" \
      decode \
      "${RESULT_DIR}/decode.jsonl" \
      > "${RESULT_DIR}/decode.log" 2>&1

	sleep 3
    bash "${PROJECT_ROOT}/scripts/bench_case.sh" \
      prefill \
      "${RESULT_DIR}/prefill.jsonl" \
      > "${RESULT_DIR}/prefill.log" 2>&1

    bash "${PROJECT_ROOT}/scripts/bench_case.sh" \
      balanced \
      "${RESULT_DIR}/balanced.jsonl" \
      > "${RESULT_DIR}/balanced.log" 2>&1

    stop_server
    sleep 6
  done
done
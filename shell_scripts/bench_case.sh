#!/usr/bin/env bash
set -euo pipefail

source /root/autodl-tmp/venvs/sglang015/bin/activate
source "$(dirname "$0")/env.sh"

CASE_NAME="${1:?Usage: bench_case.sh <decode|prefill|balanced> <output.jsonl>}"
OUTPUT_FILE="${2:?Usage: bench_case.sh <decode|prefill|balanced> <output.jsonl>}"

case "${CASE_NAME}" in
  decode)
    INPUT_LEN=256
    OUTPUT_LEN=1024
    NUM_PROMPTS=160
    CONCURRENCY=32
    REQUEST_RATE=inf
    ;;

  prefill)
    INPUT_LEN=16384
    OUTPUT_LEN=8
    NUM_PROMPTS=80
    CONCURRENCY=4
    REQUEST_RATE=inf
    ;;

  balanced)
    INPUT_LEN=1024
    OUTPUT_LEN=256
    NUM_PROMPTS=480
    CONCURRENCY=32
    REQUEST_RATE=inf
    ;;

  *)
    echo "Unknown case: ${CASE_NAME}" >&2
    exit 2
    ;;
esac

mkdir -p "$(dirname "${OUTPUT_FILE}")"
rm -f "${OUTPUT_FILE}"

python -m sglang.benchmark.serving \
  --backend sglang \
  --host "${HOST}" \
  --port "${PORT}" \
  --model "${MODEL_PATH}" \
  --dataset-name random-ids \
  --random-input-len "${INPUT_LEN}" \
  --random-output-len "${OUTPUT_LEN}" \
  --random-range-ratio 1 \
  --num-prompts "${NUM_PROMPTS}" \
  --max-concurrency "${CONCURRENCY}" \
  --request-rate "${REQUEST_RATE}" \
  --temperature 0 \
  --top-p 1 \
  --warmup-requests 3 \
  --seed 42 \
  --output-file "${OUTPUT_FILE}" \
  --output-details
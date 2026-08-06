#!/usr/bin/env bash

set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-/root/projects/sglang-qwen2-adaptive-prefill}"

source /root/autodl-tmp/venvs/sglang015/bin/activate
source "${PROJECT_ROOT}/scripts/env.sh"
source "${PROJECT_ROOT}/scripts/workload_profiles.sh"

RESULT_DIR="${1:?Usage: bench_mixed.sh RESULT_DIR [PROFILE]}"
PROFILE_NAME="${2:-standard}"

load_workload_profile "${PROFILE_NAME}"

mkdir -p "${RESULT_DIR}"

rm -f \
    "${RESULT_DIR}/decode.jsonl" \
    "${RESULT_DIR}/decode.log" \
    "${RESULT_DIR}/prefill.jsonl" \
    "${RESULT_DIR}/prefill.log"

cat > "${RESULT_DIR}/config.txt" <<EOF
experiment=mixed
profile=${PROFILE}

DECODE_INPUT_LEN=${DECODE_INPUT_LEN}
DECODE_OUTPUT_LEN=${DECODE_OUTPUT_LEN}
DECODE_PROMPTS=${DECODE_PROMPTS}
DECODE_CONCURRENCY=${DECODE_CONCURRENCY}
DECODE_REQUEST_RATE=${DECODE_REQUEST_RATE}
DECODE_SEED=${DECODE_SEED}

PREFILL_INPUT_LEN=${PREFILL_INPUT_LEN}
PREFILL_OUTPUT_LEN=${PREFILL_OUTPUT_LEN}
PREFILL_PROMPTS=${PREFILL_PROMPTS}
PREFILL_CONCURRENCY=${PREFILL_CONCURRENCY}
PREFILL_REQUEST_RATE=${PREFILL_REQUEST_RATE}
PREFILL_SEED=${PREFILL_SEED}

PREFILL_DELAY=${PREFILL_DELAY}
EOF

echo "=================================================="
echo "Mixed workload"
echo "=================================================="
echo "Profile              : ${PROFILE}"
echo "Result dir           : ${RESULT_DIR}"
echo
echo "Decode input/output  : ${DECODE_INPUT_LEN}/${DECODE_OUTPUT_LEN}"
echo "Decode prompts       : ${DECODE_PROMPTS}"
echo "Decode concurrency   : ${DECODE_CONCURRENCY}"
echo "Decode request rate  : ${DECODE_REQUEST_RATE}"
echo
echo "Prefill input/output : ${PREFILL_INPUT_LEN}/${PREFILL_OUTPUT_LEN}"
echo "Prefill prompts      : ${PREFILL_PROMPTS}"
echo "Prefill concurrency  : ${PREFILL_CONCURRENCY}"
echo "Prefill request rate : ${PREFILL_REQUEST_RATE}"
echo "Prefill delay        : ${PREFILL_DELAY}s"
echo "=================================================="

DECODE_PID=""
PREFILL_PID=""

cleanup_clients() {
    if [[ -n "${DECODE_PID}" ]]; then
        kill "${DECODE_PID}" 2>/dev/null || true
    fi

    if [[ -n "${PREFILL_PID}" ]]; then
        kill "${PREFILL_PID}" 2>/dev/null || true
    fi
}

trap cleanup_clients INT TERM

echo
echo "[1/2] Starting Decode client..."

(
    python -m sglang.benchmark.serving \
        --backend sglang \
        --host "${HOST}" \
        --port "${PORT}" \
        --dataset-name random-ids \
        --tokenize-prompt \
        --random-input-len "${DECODE_INPUT_LEN}" \
        --random-output-len "${DECODE_OUTPUT_LEN}" \
        --random-range-ratio 1.0 \
        --num-prompts "${DECODE_PROMPTS}" \
        --max-concurrency "${DECODE_CONCURRENCY}" \
        --request-rate "${DECODE_REQUEST_RATE}" \
        --seed "${DECODE_SEED}" \
        --output-file "${RESULT_DIR}/decode.jsonl" \
        --output-details \
        2>&1 | tee "${RESULT_DIR}/decode.log"
) &

DECODE_PID=$!

echo "Decode PID: ${DECODE_PID}"
echo "Waiting ${PREFILL_DELAY}s before Prefill injection..."

sleep "${PREFILL_DELAY}"

echo
echo "[2/2] Starting Prefill client..."

(
    python -m sglang.benchmark.serving \
        --backend sglang \
        --host "${HOST}" \
        --port "${PORT}" \
        --dataset-name random-ids \
        --tokenize-prompt \
        --random-input-len "${PREFILL_INPUT_LEN}" \
        --random-output-len "${PREFILL_OUTPUT_LEN}" \
        --random-range-ratio 1.0 \
        --num-prompts "${PREFILL_PROMPTS}" \
        --max-concurrency "${PREFILL_CONCURRENCY}" \
        --request-rate "${PREFILL_REQUEST_RATE}" \
        --seed "${PREFILL_SEED}" \
        --output-file "${RESULT_DIR}/prefill.jsonl" \
        --output-details \
        2>&1 | tee "${RESULT_DIR}/prefill.log"
) &

PREFILL_PID=$!

echo "Prefill PID: ${PREFILL_PID}"
echo "Decode and Prefill are now running concurrently."

set +e

wait "${PREFILL_PID}"
PREFILL_STATUS=$?

wait "${DECODE_PID}"
DECODE_STATUS=$?

set -e

trap - INT TERM

echo
echo "=================================================="
echo "Mixed workload finished"
echo "=================================================="
echo "Decode status  : ${DECODE_STATUS}"
echo "Prefill status : ${PREFILL_STATUS}"
echo "Decode result  : ${RESULT_DIR}/decode.jsonl"
echo "Prefill result : ${RESULT_DIR}/prefill.jsonl"
echo "=================================================="

if [[ "${DECODE_STATUS}" -ne 0 || "${PREFILL_STATUS}" -ne 0 ]]; then
    echo "ERROR: One or more benchmark clients failed." >&2
    exit 1
fi
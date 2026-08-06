#!/usr/bin/env bash

set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-/root/projects/sglang-qwen2-adaptive-prefill}"

source /root/autodl-tmp/venvs/sglang015/bin/activate
source "${PROJECT_ROOT}/scripts/env.sh"
source "${PROJECT_ROOT}/scripts/workload_profiles.sh"

RESULT_DIR="${1:?Usage: bench_decode_fixed_rate.sh RESULT_DIR [PROFILE]}"
PROFILE_NAME="${2:-standard}"

load_workload_profile "${PROFILE_NAME}"

mkdir -p "${RESULT_DIR}"

# 防止重新执行时继续向旧JSONL追加结果。
rm -f \
    "${RESULT_DIR}/decode.jsonl" \
    "${RESULT_DIR}/decode.log"

cat > "${RESULT_DIR}/config.txt" <<EOF
experiment=decode_only
profile=${PROFILE}

DECODE_INPUT_LEN=${DECODE_INPUT_LEN}
DECODE_OUTPUT_LEN=${DECODE_OUTPUT_LEN}
DECODE_PROMPTS=${DECODE_PROMPTS}
DECODE_CONCURRENCY=${DECODE_CONCURRENCY}
DECODE_REQUEST_RATE=${DECODE_REQUEST_RATE}
DECODE_SEED=${DECODE_SEED}
EOF

echo "=================================================="
echo "Fixed-rate Decode-only"
echo "=================================================="
echo "Profile       : ${PROFILE}"
echo "Result dir    : ${RESULT_DIR}"
echo "Input/output  : ${DECODE_INPUT_LEN}/${DECODE_OUTPUT_LEN}"
echo "Prompts       : ${DECODE_PROMPTS}"
echo "Concurrency   : ${DECODE_CONCURRENCY}"
echo "Request rate  : ${DECODE_REQUEST_RATE} req/s"
echo "Seed          : ${DECODE_SEED}"
echo "=================================================="

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
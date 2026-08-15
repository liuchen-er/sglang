#!/usr/bin/env bash
set -euo pipefail

if [ $# -ne 5 ]; then
    echo "Usage:"
    echo "$0 <colocated|pd> <input_len> <output_len> <concurrency> <request_rate>"
    exit 1
fi

MODE=$1
INPUT_LEN=$2
OUTPUT_LEN=$3
CONC=$4
RATE=$5

SCRIPT_DIR=$(cd "$(dirname "$0")" && pwd)
source "$SCRIPT_DIR/00_common.sh"

NUM_PROMPTS=$((CONC * 8))
TS=$(date +%Y%m%d_%H%M%S)

TAG="${MODE}_i${INPUT_LEN}_o${OUTPUT_LEN}_c${CONC}_r${RATE}_${TS}"

RESULT_DIR="$RESULT_ROOT/raw/$MODE"
BENCH_LOG_DIR="$LOG_ROOT/benchmark/$MODE"

mkdir -p "$RESULT_DIR" "$BENCH_LOG_DIR"

RESULT_FILE="$RESULT_DIR/${TAG}.jsonl"
LOG_FILE="$BENCH_LOG_DIR/${TAG}.log"

PD_ARGS=()

if [ "$MODE" = "pd" ]; then
    PD_ARGS+=(--pd-separated)
fi

echo
echo "==========================================="
echo "Mode         : $MODE"
echo "Input tokens : $INPUT_LEN"
echo "Output tokens: $OUTPUT_LEN"
echo "Concurrency  : $CONC"
echo "Request rate : $RATE"
echo "Prompts      : $NUM_PROMPTS"
echo "Result       : $RESULT_FILE"
echo "==========================================="

PYTHONPATH="$REPO/python${PYTHONPATH:+:$PYTHONPATH}" \
python -c 'from sglang.benchmark.serving import cli_main; cli_main()' \
    --backend sglang \
    --base-url "http://${HOST}:${ROUTER_PORT}" \
    --model "$MODEL_PATH" \
    --dataset-name random-ids \
    --num-prompts "$NUM_PROMPTS" \
    --random-input-len "$INPUT_LEN" \
    --random-output-len "$OUTPUT_LEN" \
    --random-range-ratio 0 \
    --request-rate "$RATE" \
    --max-concurrency "$CONC" \
    --warmup-requests 4 \
    --flush-cache \
    --seed 42 \
    --temperature 0 \
    --output-file "$RESULT_FILE" \
    --output-details \
    --tag "$TAG" \
    "${PD_ARGS[@]}" \
    2>&1 | tee "$LOG_FILE"

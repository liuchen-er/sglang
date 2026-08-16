#!/usr/bin/env bash
set -euo pipefail

# ============================================================
# Arguments
# ============================================================
if [ $# -lt 5 ] || [ $# -gt 6 ]; then
    echo "Usage:"
    echo "$0 <colocated|pd> <input_len> <output_len> <concurrency> <request_rate> [num_prompts]"
    echo
    echo "Examples:"
    echo "$0 colocated 4096 256 8 inf"
    echo "$0 colocated 4096 256 8 inf 300"
    echo "$0 pd        8192 256 16 0.54 300"
    exit 1
fi

MODE=$1
INPUT_LEN=$2
OUTPUT_LEN=$3
CONC=$4
RATE=$5
NUM_PROMPTS=${6:-300}
WARMUP_REQUESTS=${WARMUP_REQUESTS:-4}

# Experiment phase:
#   capacity
#   offered_load
#   smoke
#   profile
#   manual
PHASE=${BENCH_PHASE:-manual}

# ============================================================
# Validation
# ============================================================
if [ "$MODE" != "colocated" ] && [ "$MODE" != "pd" ]; then
    echo "ERROR: mode must be 'colocated' or 'pd'"
    exit 1
fi

case "$PHASE" in
    capacity|offered_load|smoke|profile|manual)
        ;;
    *)
        echo "ERROR: invalid BENCH_PHASE=$PHASE"
        echo "Valid values: capacity offered_load smoke profile manual"
        exit 1
        ;;
esac

if ! [[ "$INPUT_LEN" =~ ^[1-9][0-9]*$ ]]; then
    echo "ERROR: input_len must be a positive integer"
    exit 1
fi

if ! [[ "$OUTPUT_LEN" =~ ^[1-9][0-9]*$ ]]; then
    echo "ERROR: output_len must be a positive integer"
    exit 1
fi

if ! [[ "$CONC" =~ ^[1-9][0-9]*$ ]]; then
    echo "ERROR: concurrency must be a positive integer"
    exit 1
fi

if ! [[ "$NUM_PROMPTS" =~ ^[1-9][0-9]*$ ]]; then
    echo "ERROR: num_prompts must be a positive integer"
    exit 1
fi

# ============================================================
# Common environment
# ============================================================
SCRIPT_DIR=$(cd "$(dirname "$0")" && pwd)
source "$SCRIPT_DIR/00_common.sh"

# ============================================================
# Output paths
# ============================================================
TS=$(date +%Y%m%d_%H%M%S)

TAG="${MODE}_i${INPUT_LEN}_o${OUTPUT_LEN}_c${CONC}_r${RATE}_n${NUM_PROMPTS}_${PHASE}_${TS}"

RESULT_DIR="$RESULT_ROOT/raw/$PHASE/$MODE"
BENCH_LOG_DIR="$LOG_ROOT/benchmark/$PHASE/$MODE"

mkdir -p \
    "$RESULT_DIR" \
    "$BENCH_LOG_DIR"

RESULT_FILE="$RESULT_DIR/${TAG}.jsonl"
LOG_FILE="$BENCH_LOG_DIR/${TAG}.log"

# ============================================================
# PD-specific benchmark arguments
# ============================================================
PD_ARGS=()

if [ "$MODE" = "pd" ]; then
    PD_ARGS+=(--pd-separated)
fi

# ============================================================
# Print configuration
# ============================================================
echo
echo "==========================================="
echo "Phase        : $PHASE"
echo "Mode         : $MODE"
echo "Input tokens : $INPUT_LEN"
echo "Output tokens: $OUTPUT_LEN"
echo "Concurrency  : $CONC"
echo "Request rate : $RATE"
echo "Prompts      : $NUM_PROMPTS"
echo "Result       : $RESULT_FILE"
echo "Log          : $LOG_FILE"
echo "==========================================="

# ============================================================
# Run benchmark
#
# random-range-ratio=1.0:
#   fixed input/output lengths
#
# tokenize-prompt:
#   directly send token IDs, avoiding decode -> re-tokenize drift
# ============================================================
PYTHONPATH="$REPO/python${PYTHONPATH:+:$PYTHONPATH}" \
python -c 'from sglang.benchmark.serving import cli_main; cli_main()' \
    --backend sglang \
    --base-url "http://${HOST}:${ROUTER_PORT}" \
    --model "$MODEL_PATH" \
    --dataset-name random-ids \
    --num-prompts "$NUM_PROMPTS" \
    --random-input-len "$INPUT_LEN" \
    --random-output-len "$OUTPUT_LEN" \
    --random-range-ratio 1.0 \
    --tokenize-prompt \
    --request-rate "$RATE" \
    --max-concurrency "$CONC" \
    --warmup-requests "$WARMUP_REQUESTS" \
    --flush-cache \
    --seed 42 \
    --temperature 0 \
    --output-file "$RESULT_FILE" \
    --output-details \
    --tag "$TAG" \
    "${PD_ARGS[@]}" \
    2>&1 | tee "$LOG_FILE"

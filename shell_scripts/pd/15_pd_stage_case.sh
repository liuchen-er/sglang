#!/usr/bin/env bash
set -euo pipefail

if [ $# -ne 4 ]; then
    echo "Usage:"
    echo "$0 <case_name> <input_len> <request_rate> <num_prompts>"
    echo
    echo "Example:"
    echo "$0 caseB 8192 0.30 30"
    exit 1
fi

CASE=$1
INPUT_LEN=$2
RATE=$3
NUM_PROMPTS=$4

SCRIPT_DIR=$(cd "$(dirname "$0")" && pwd)

OUT_DIR="/root/autodl-tmp/sglang_pd_exp/results/raw/profile/stage_breakdown/$CASE"

mkdir -p "$OUT_DIR"

echo
echo "=========================================="
echo "PD STAGE BREAKDOWN"
echo "Case     : $CASE"
echo "Input    : $INPUT_LEN"
echo "Output   : 256"
echo "Rate     : $RATE"
echo "Requests : $NUM_PROMPTS"
echo "=========================================="

echo "[1/3] Snapshot metrics before benchmark..."

curl -sf http://127.0.0.1:30000/metrics \
    > "$OUT_DIR/prefill_before.metrics"

curl -sf http://127.0.0.1:30001/metrics \
    > "$OUT_DIR/decode_before.metrics"


echo "[2/3] Run benchmark..."

WARMUP_REQUESTS=0 \
BENCH_PHASE=profile \
"$SCRIPT_DIR/10_bench_one.sh" \
    pd \
    "$INPUT_LEN" \
    256 \
    24 \
    "$RATE" \
    "$NUM_PROMPTS"


echo "[3/3] Snapshot metrics after benchmark..."

curl -sf http://127.0.0.1:30000/metrics \
    > "$OUT_DIR/prefill_after.metrics"

curl -sf http://127.0.0.1:30001/metrics \
    > "$OUT_DIR/decode_after.metrics"

echo
echo "Saved:"
echo "  $OUT_DIR"

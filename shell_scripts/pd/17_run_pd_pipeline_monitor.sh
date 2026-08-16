#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "$0")" && pwd)

EXP_ROOT=/root/autodl-tmp/sglang_pd_exp

OUT_DIR="$EXP_ROOT/results/raw/profile/pipeline_monitor"

mkdir -p "$OUT_DIR"

TS=$(date +%Y%m%d_%H%M%S)

TAG="pd_i8192_o256_c24_r0.54_n100_${TS}"

CSV="$OUT_DIR/${TAG}.csv"
STOP_FILE="$OUT_DIR/${TAG}.stop"

rm -f "$STOP_FILE"

echo "=============================================="
echo "PD PIPELINE MONITOR"
echo "=============================================="
echo "Input       : 8192"
echo "Output      : 256"
echo "Rate        : 0.54 req/s"
echo "Concurrency : 24"
echo "Requests    : 100"
echo "Interval    : 1 s"
echo "CSV         : $CSV"
echo "=============================================="

cleanup() {
    touch "$STOP_FILE" 2>/dev/null || true

    if [[ -n "${MON_PID:-}" ]]; then
        wait "$MON_PID" 2>/dev/null || true
    fi
}

trap cleanup EXIT INT TERM


echo
echo "[1/3] Start metrics monitor..."

python "$SCRIPT_DIR/17_monitor_pd_pipeline.py" \
    --interval 1 \
    --output "$CSV" \
    --stop-file "$STOP_FILE" &

MON_PID=$!

# Give monitor time to obtain baseline sample
sleep 2


echo
echo "[2/3] Run Case C workload..."

WARMUP_REQUESTS=0 \
BENCH_PHASE=profile \
"$SCRIPT_DIR/10_bench_one.sh" \
    pd \
    8192 \
    256 \
    24 \
    0.54 \
    100


echo
echo "[3/3] Keep monitoring for 5 seconds..."

sleep 5

touch "$STOP_FILE"

wait "$MON_PID"

trap - EXIT INT TERM

rm -f "$STOP_FILE"

echo
echo "Done."
echo "CSV:"
echo "$CSV"

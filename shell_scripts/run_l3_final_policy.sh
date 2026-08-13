#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 1 ]]; then
    echo "Usage: $0 {always_restore|always_recompute|cost_model}"
    exit 1
fi

POLICY="$1"

case "$POLICY" in
    always_restore|always_recompute|cost_model)
        ;;
    *)
        echo "Invalid policy: $POLICY"
        exit 1
        ;;
esac

cd /root/projects/sglang-qwen2-adaptive-prefill/sglang
source /root/autodl-tmp/venvs/sglang015/bin/activate

export ADMIN_API_KEY=hicache-admin-local
export HICACHE_BENCH_TIMEOUT=600

RESULT_DIR=/root/projects/sglang-qwen2-adaptive-prefill/hicache/results/l3_final
mkdir -p "$RESULT_DIR"

run_case_group() {
    local concurrency="$1"
    local prefixes="$2"
    local result_file="$3"

    echo
    echo "============================================================"
    echo "FINAL L3 EXPERIMENT"
    echo "policy=$POLICY"
    echo "concurrency=$concurrency"
    echo "prefixes=$prefixes"
    echo "result=$result_file"
    echo "============================================================"

    TARGET_CACHE_TIER=L3 \
    L3_PREP_MODE=flush \
    CLEAR_L3=1 \
    VALIDATE_CACHE_TIER=0 \
    POLICY="$POLICY" \
    PREFIX_LENGTHS="$prefixes" \
    SESSIONS_PER_LEN=32 \
    TRIALS=3 \
    MAX_CONCURRENCY="$concurrency" \
    REQUEST_RATE=inf \
    OUTPUT_LEN=1 \
    PAGE_SIZE=64 \
    L3_BACKUP_TIMEOUT_S=180 \
    RESET_RESULT=1 \
    RESULT_FILE="$result_file" \
    python shell_scripts/bench_hicache_agent.py
}

run_case_group \
    1 \
    "256,8192,16384" \
    "$RESULT_DIR/l3_final_${POLICY}_c1.jsonl"

run_case_group \
    16 \
    "256,4096,8192,16384" \
    "$RESULT_DIR/l3_final_${POLICY}_c16.jsonl"

run_case_group \
    32 \
    "256,4096,8192" \
    "$RESULT_DIR/l3_final_${POLICY}_c32.jsonl"

echo
echo "============================================================"
echo "FINAL L3 EXPERIMENT FINISHED"
echo "policy=$POLICY"
echo "============================================================"

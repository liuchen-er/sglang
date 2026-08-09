#!/usr/bin/env bash
set -euo pipefail

cd /root/projects/sglang-qwen2-adaptive-prefill/sglang
source /root/autodl-tmp/venvs/sglang015/bin/activate

export ADMIN_API_KEY=hicache-admin-local
export HICACHE_BENCH_TIMEOUT=600

RESULT_DIR=/root/projects/sglang-qwen2-adaptive-prefill/hicache/results/l3_v31
mkdir -p "$RESULT_DIR"

run_group() {
    local concurrency="$1"
    local prefixes="$2"

    TARGET_CACHE_TIER=L3 \
    L3_PREP_MODE=flush \
    CLEAR_L3=1 \
    VALIDATE_CACHE_TIER=0 \
    POLICY=cost_model \
    PREFIX_LENGTHS="$prefixes" \
    SESSIONS_PER_LEN=32 \
    TRIALS=1 \
    MAX_CONCURRENCY="$concurrency" \
    REQUEST_RATE=inf \
    OUTPUT_LEN=1 \
    PAGE_SIZE=64 \
    L3_BACKUP_TIMEOUT_S=180 \
    RESET_RESULT=1 \
    RESULT_FILE="$RESULT_DIR/l3_v31_c${concurrency}.jsonl" \
    python shell_scripts/bench_hicache_agent.py
}

run_group 16 "4096,16384"
run_group 32 "4096,8192"

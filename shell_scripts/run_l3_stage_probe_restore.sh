#!/usr/bin/env bash
set -euo pipefail

cd /root/projects/sglang-qwen2-adaptive-prefill/sglang
source /root/autodl-tmp/venvs/sglang015/bin/activate

export MODEL_PATH=/root/autodl-tmp/models/Qwen2.5-1.5B-Instruct-ms
export ADMIN_API_KEY=hicache-admin-local

RESULT_DIR=/root/projects/sglang-qwen2-adaptive-prefill/hicache/results/l3_next
mkdir -p "$RESULT_DIR"

echo "============================================================"
echo "L3 Stage Probe: c16 / 8192"
echo "============================================================"

TARGET_CACHE_TIER=L3 \
L3_PREP_MODE=flush \
CLEAR_L3=1 \
VALIDATE_CACHE_TIER=0 \
POLICY=always_restore \
PREFIX_LENGTHS=8192 \
SESSIONS_PER_LEN=32 \
TRIALS=3 \
MAX_CONCURRENCY=16 \
REQUEST_RATE=inf \
OUTPUT_LEN=1 \
PAGE_SIZE=64 \
L3_BACKUP_TIMEOUT_S=180 \
RESET_RESULT=1 \
RESULT_FILE="$RESULT_DIR/l3_stage_restore_c16_p8192.jsonl" \
python shell_scripts/bench_hicache_agent.py

echo "============================================================"
echo "L3 Stage Probe: c32 / 512"
echo "============================================================"

TARGET_CACHE_TIER=L3 \
L3_PREP_MODE=flush \
CLEAR_L3=1 \
VALIDATE_CACHE_TIER=0 \
POLICY=always_restore \
PREFIX_LENGTHS=512 \
SESSIONS_PER_LEN=32 \
TRIALS=3 \
MAX_CONCURRENCY=32 \
REQUEST_RATE=inf \
OUTPUT_LEN=1 \
PAGE_SIZE=64 \
L3_BACKUP_TIMEOUT_S=180 \
RESET_RESULT=1 \
RESULT_FILE="$RESULT_DIR/l3_stage_restore_c32_p512.jsonl" \
python shell_scripts/bench_hicache_agent.py

echo "============================================================"
echo "L3 Stage Probe: c32 / 2048"
echo "============================================================"

TARGET_CACHE_TIER=L3 \
L3_PREP_MODE=flush \
CLEAR_L3=1 \
VALIDATE_CACHE_TIER=0 \
POLICY=always_restore \
PREFIX_LENGTHS=2048 \
SESSIONS_PER_LEN=32 \
TRIALS=3 \
MAX_CONCURRENCY=32 \
REQUEST_RATE=inf \
OUTPUT_LEN=1 \
PAGE_SIZE=64 \
L3_BACKUP_TIMEOUT_S=180 \
RESET_RESULT=1 \
RESULT_FILE="$RESULT_DIR/l3_stage_restore_c32_p2048.jsonl" \
python shell_scripts/bench_hicache_agent.py

echo "============================================================"
echo "L3 stage probe finished"
echo "============================================================"

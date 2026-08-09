#!/usr/bin/env bash
set -euo pipefail

cd /root/projects/sglang-qwen2-adaptive-prefill/sglang
source /root/autodl-tmp/venvs/sglang015/bin/activate

export MODEL_PATH=/root/autodl-tmp/models/Qwen2.5-1.5B-Instruct-ms
export ADMIN_API_KEY=hicache-admin-local

RESULT_DIR=/root/projects/sglang-qwen2-adaptive-prefill/hicache/results/l3_refine
mkdir -p "$RESULT_DIR"

echo
echo "============================================================"
echo "L3 REFINE: always_recompute, concurrency=1"
echo "prefix=1024,2048,3072,4096"
echo "============================================================"

TARGET_CACHE_TIER=L3 \
L3_PREP_MODE=flush \
CLEAR_L3=1 \
VALIDATE_CACHE_TIER=0 \
POLICY=always_recompute \
PREFIX_LENGTHS=1024,2048,3072,4096 \
SESSIONS_PER_LEN=32 \
TRIALS=3 \
MAX_CONCURRENCY=1 \
REQUEST_RATE=inf \
OUTPUT_LEN=1 \
PAGE_SIZE=64 \
L3_BACKUP_TIMEOUT_S=180 \
RESET_RESULT=1 \
RESULT_FILE="$RESULT_DIR/l3_recompute_c1_refine.jsonl" \
python shell_scripts/bench_hicache_agent.py

echo
echo "============================================================"
echo "L3 REFINE: always_recompute, concurrency=16"
echo "prefix=1024,2048,4096,8192"
echo "============================================================"

TARGET_CACHE_TIER=L3 \
L3_PREP_MODE=flush \
CLEAR_L3=1 \
VALIDATE_CACHE_TIER=0 \
POLICY=always_recompute \
PREFIX_LENGTHS=1024,2048,4096,8192 \
SESSIONS_PER_LEN=32 \
TRIALS=3 \
MAX_CONCURRENCY=16 \
REQUEST_RATE=inf \
OUTPUT_LEN=1 \
PAGE_SIZE=64 \
L3_BACKUP_TIMEOUT_S=180 \
RESET_RESULT=1 \
RESULT_FILE="$RESULT_DIR/l3_recompute_c16_refine.jsonl" \
python shell_scripts/bench_hicache_agent.py

echo
echo "============================================================"
echo "L3 REFINE: always_recompute, concurrency=32"
echo "prefix=256,512,1024,2048"
echo "============================================================"

TARGET_CACHE_TIER=L3 \
L3_PREP_MODE=flush \
CLEAR_L3=1 \
VALIDATE_CACHE_TIER=0 \
POLICY=always_recompute \
PREFIX_LENGTHS=256,512,1024,2048 \
SESSIONS_PER_LEN=32 \
TRIALS=3 \
MAX_CONCURRENCY=32 \
REQUEST_RATE=inf \
OUTPUT_LEN=1 \
PAGE_SIZE=64 \
L3_BACKUP_TIMEOUT_S=180 \
RESET_RESULT=1 \
RESULT_FILE="$RESULT_DIR/l3_recompute_c32_refine.jsonl" \
python shell_scripts/bench_hicache_agent.py

echo
echo "============================================================"
echo "ALL L3 RECOMPUTE REFINE CASES FINISHED"
echo "============================================================"

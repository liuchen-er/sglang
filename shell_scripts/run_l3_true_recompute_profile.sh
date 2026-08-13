#!/usr/bin/env bash
set -euo pipefail

cd /root/projects/sglang-qwen2-adaptive-prefill/sglang
source /root/autodl-tmp/venvs/sglang015/bin/activate

export MODEL_PATH=/root/autodl-tmp/models/Qwen2.5-1.5B-Instruct-ms
export ADMIN_API_KEY=hicache-admin-local

RESULT_DIR=/root/projects/sglang-qwen2-adaptive-prefill/hicache/results/l3_true_profile
mkdir -p "$RESULT_DIR"

echo "============================================================"
echo "TRUE L3 PROFILE: always_recompute c=1"
echo "============================================================"

TARGET_CACHE_TIER=L3 \
L3_PREP_MODE=flush \
CLEAR_L3=1 \
VALIDATE_CACHE_TIER=0 \
POLICY=always_recompute \
PREFIX_LENGTHS=256,512,1024,2048,4096,8192,16384 \
SESSIONS_PER_LEN=32 \
TRIALS=3 \
MAX_CONCURRENCY=1 \
REQUEST_RATE=inf \
OUTPUT_LEN=1 \
PAGE_SIZE=64 \
L3_BACKUP_TIMEOUT_S=180 \
RESET_RESULT=1 \
RESULT_FILE="$RESULT_DIR/l3_true_recompute_c1.jsonl" \
python shell_scripts/bench_hicache_agent.py

echo "============================================================"
echo "TRUE L3 PROFILE: always_recompute c=16"
echo "============================================================"

TARGET_CACHE_TIER=L3 \
L3_PREP_MODE=flush \
CLEAR_L3=1 \
VALIDATE_CACHE_TIER=0 \
POLICY=always_recompute \
PREFIX_LENGTHS=256,512,1024,2048,4096,8192,16384 \
SESSIONS_PER_LEN=32 \
TRIALS=3 \
MAX_CONCURRENCY=16 \
REQUEST_RATE=inf \
OUTPUT_LEN=1 \
PAGE_SIZE=64 \
L3_BACKUP_TIMEOUT_S=180 \
RESET_RESULT=1 \
RESULT_FILE="$RESULT_DIR/l3_true_recompute_c16.jsonl" \
python shell_scripts/bench_hicache_agent.py

echo "============================================================"
echo "TRUE L3 PROFILE: always_recompute c=32"
echo "============================================================"

TARGET_CACHE_TIER=L3 \
L3_PREP_MODE=flush \
CLEAR_L3=1 \
VALIDATE_CACHE_TIER=0 \
POLICY=always_recompute \
PREFIX_LENGTHS=256,512,1024,2048,4096,8192 \
SESSIONS_PER_LEN=32 \
TRIALS=3 \
MAX_CONCURRENCY=32 \
REQUEST_RATE=inf \
OUTPUT_LEN=1 \
PAGE_SIZE=64 \
L3_BACKUP_TIMEOUT_S=180 \
RESET_RESULT=1 \
RESULT_FILE="$RESULT_DIR/l3_true_recompute_c32.jsonl" \
python shell_scripts/bench_hicache_agent.py

echo "============================================================"
echo "ALL TRUE RECOMPUTE PROFILE CASES FINISHED"
echo "============================================================"

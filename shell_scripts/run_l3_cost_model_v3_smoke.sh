#!/usr/bin/env bash
set -euo pipefail

cd /root/projects/sglang-qwen2-adaptive-prefill/sglang
source /root/autodl-tmp/venvs/sglang015/bin/activate

export MODEL_PATH=/root/autodl-tmp/models/Qwen2.5-1.5B-Instruct-ms
export ADMIN_API_KEY=hicache-admin-local
export HICACHE_COST_PROFILE=/root/projects/sglang-qwen2-adaptive-prefill/hicache/profiles/hicache_cost_profile_v3.json
export HICACHE_MARGIN_MS=2.0

RESULT_DIR=/root/projects/sglang-qwen2-adaptive-prefill/hicache/results/l3_next
mkdir -p "$RESULT_DIR"

echo "============================================================"
echo "V3 SMOKE: c1 / prefix=256"
echo "Expected action: recompute"
echo "============================================================"

TARGET_CACHE_TIER=L3 \
L3_PREP_MODE=flush \
CLEAR_L3=1 \
VALIDATE_CACHE_TIER=0 \
POLICY=cost_model \
PREFIX_LENGTHS=256 \
SESSIONS_PER_LEN=4 \
TRIALS=1 \
MAX_CONCURRENCY=1 \
REQUEST_RATE=inf \
OUTPUT_LEN=1 \
PAGE_SIZE=64 \
L3_BACKUP_TIMEOUT_S=180 \
RESET_RESULT=1 \
RESULT_FILE="$RESULT_DIR/l3_cost_model_v3_c1_p256.jsonl" \
python shell_scripts/bench_hicache_agent.py

echo
echo "============================================================"
echo "V3 SMOKE: c1 / prefix=16384"
echo "Expected action: restore"
echo "============================================================"

TARGET_CACHE_TIER=L3 \
L3_PREP_MODE=flush \
CLEAR_L3=1 \
VALIDATE_CACHE_TIER=0 \
POLICY=cost_model \
PREFIX_LENGTHS=16384 \
SESSIONS_PER_LEN=4 \
TRIALS=1 \
MAX_CONCURRENCY=1 \
REQUEST_RATE=inf \
OUTPUT_LEN=1 \
PAGE_SIZE=64 \
L3_BACKUP_TIMEOUT_S=180 \
RESET_RESULT=1 \
RESULT_FILE="$RESULT_DIR/l3_cost_model_v3_c1_p16384.jsonl" \
python shell_scripts/bench_hicache_agent.py

echo
echo "============================================================"
echo "V3 smoke test finished"
echo "============================================================"

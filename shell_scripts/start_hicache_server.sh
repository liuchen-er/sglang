#!/usr/bin/env bash
set -euo pipefail

POLICY="${1:-always_restore}"
THRESHOLD="${2:-0}"
REPO=/root/projects/sglang-qwen2-adaptive-prefill/sglang
DATA=/root/projects/sglang-qwen2-adaptive-prefill/hicache
LOG_TAG="${LOG_TAG:-$POLICY}"

source /root/autodl-tmp/venvs/sglang015/bin/activate
cd "$REPO"
: "${MODEL_PATH:?Please export MODEL_PATH first}"

ARGS=(
    --model-path "$MODEL_PATH"
    --host 127.0.0.1
    --port 30000
    --page-size 64
    --mem-fraction-static 0.65
    --enable-hierarchical-cache
    --hicache-ratio 2
    --hicache-io-backend kernel
    --hicache-mem-layout page_first
    --hicache-write-policy write_through
    --hicache-restore-policy "$POLICY"
    --enable-metrics
    --enable-cache-report
)

if [[ "$POLICY" == "token_threshold" ]]; then
    ARGS+=(--hicache-restore-token-threshold "$THRESHOLD")
fi

if [[ "$POLICY" == "cost_model" ]]; then
    PROFILE="${HICACHE_COST_PROFILE:-/root/projects/sglang-qwen2-adaptive-prefill/hicache/profiles/hicache_cost_profile.json}"
    MARGIN="${HICACHE_MARGIN_MS:-1.0}"
    ARGS+=(--hicache-cost-profile "$PROFILE" --hicache-restore-safety-margin-ms "$MARGIN")
fi

python -m sglang.launch_server "${ARGS[@]}" 2>&1 | tee "$DATA/logs/server_${LOG_TAG}.log"

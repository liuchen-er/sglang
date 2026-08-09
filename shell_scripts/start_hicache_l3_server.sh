#!/usr/bin/env bash
set -euo pipefail

: "${MODEL_PATH:?Please export MODEL_PATH first}"

REPO=/root/projects/sglang-qwen2-adaptive-prefill/sglang
DATA=/root/projects/sglang-qwen2-adaptive-prefill/hicache
L3_PATH="${HICACHE_L3_PATH:-/root/autodl-tmp/hicache_l3}"
POLICY="${1:-always_restore}"
THRESHOLD="${2:-960}"
LOG_TAG="${LOG_TAG:-l3_${POLICY}}"
ADMIN_API_KEY="${ADMIN_API_KEY:-hicache-admin-local}"

mkdir -p "$DATA/logs" "$L3_PATH"
cd "$REPO"

ARGS=(
    --model-path "$MODEL_PATH"
    --host 127.0.0.1
    --port 30000
    --admin-api-key "$ADMIN_API_KEY"
    --page-size 64
    --mem-fraction-static 0.65
    --enable-hierarchical-cache
    --hicache-size 20
    --hicache-io-backend kernel
    --hicache-mem-layout page_first
    --hicache-write-policy write_through
    --hicache-storage-backend file
    --file-storage-path "$L3_PATH"
    --hicache-storage-prefetch-policy wait_complete
    --enable-metrics
    --enable-cache-report
)

case "$POLICY" in
    always_restore)
        ARGS+=(--hicache-restore-policy always_restore)
        ;;
    always_recompute)
        ARGS+=(--hicache-restore-policy always_recompute)
        ;;
    token_threshold)
        ARGS+=(--hicache-restore-policy token_threshold --hicache-restore-token-threshold "$THRESHOLD")
        ;;
    cost_model)
        : "${HICACHE_COST_PROFILE:?Please export HICACHE_COST_PROFILE}"
        ARGS+=(
            --hicache-restore-policy cost_model
            --hicache-cost-profile "$HICACHE_COST_PROFILE"
            --hicache-restore-safety-margin-ms "${HICACHE_MARGIN_MS:-2.0}"
        )
        ;;
    *)
        echo "Unknown policy: $POLICY"
        exit 1
        ;;
esac

echo "============================================================"
echo "HiCache L3 Server"
echo "policy=$POLICY"
echo "L3_PATH=$L3_PATH"
echo "============================================================"

python -m sglang.launch_server "${ARGS[@]}" \
    2>&1 | tee "$DATA/logs/server_${LOG_TAG}.log"
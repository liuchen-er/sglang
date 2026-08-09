#!/usr/bin/env bash
set -euo pipefail

LOG=/root/projects/sglang-qwen2-adaptive-prefill/hicache/logs/server_l3_cost_model_v3.log

echo "============================================================"
echo "V3 SMOKE VALIDATION"
echo "============================================================"

P256_RECOMPUTE=$(
    grep '\[HiCacheEarlyDecision\]' "$LOG" \
    | grep 'policy=cost_model' \
    | grep 'storage_hit_length=256' \
    | grep 'action=recompute' \
    | wc -l
)

P256_IO=$(
    grep '\[HiCachePrefetchIO\]' "$LOG" \
    | grep 'completed_tokens=256' \
    | wc -l
)

P16384_RESTORE=$(
    grep '\[HiCacheEarlyDecision\]' "$LOG" \
    | grep 'policy=cost_model' \
    | grep 'storage_hit_length=16384' \
    | grep 'action=restore' \
    | wc -l
)

P16384_PRECOMPUTED=$(
    grep '\[HiCachePrefetchQuery\]' "$LOG" \
    | grep 'storage_hit_tokens=16384' \
    | grep 'query_source=precomputed' \
    | wc -l
)

P16384_WORKER=$(
    grep '\[HiCachePrefetchQuery\]' "$LOG" \
    | grep 'storage_hit_tokens=16384' \
    | grep 'query_source=worker' \
    | wc -l
)

P16384_IO=$(
    grep '\[HiCachePrefetchIO\]' "$LOG" \
    | grep 'completed_tokens=16384' \
    | wc -l
)

P16384_LATE_RECOMPUTE=$(
    grep '\[HiCacheDecision\]' "$LOG" \
    | grep 'host_hit_length=16384' \
    | grep 'action=recompute' \
    | wc -l
)

echo "prefix=256:"
echo "  early recompute       = $P256_RECOMPUTE"
echo "  payload io            = $P256_IO"

echo
echo "prefix=16384:"
echo "  early restore         = $P16384_RESTORE"
echo "  precomputed query     = $P16384_PRECOMPUTED"
echo "  worker re-query       = $P16384_WORKER"
echo "  payload io            = $P16384_IO"
echo "  late recompute        = $P16384_LATE_RECOMPUTE"

echo
echo "Expected:"
echo "  p256 early recompute  = 4"
echo "  p256 payload io       = 0"
echo "  p16384 early restore  = 4"
echo "  p16384 precomputed    = 4"
echo "  p16384 worker query   = 0"
echo "  p16384 payload io     = 4"
echo "  p16384 late recompute = 0"

if [[ "$P256_RECOMPUTE" -eq 4 \
   && "$P256_IO" -eq 0 \
   && "$P16384_RESTORE" -eq 4 \
   && "$P16384_PRECOMPUTED" -eq 4 \
   && "$P16384_WORKER" -eq 0 \
   && "$P16384_IO" -eq 4 \
   && "$P16384_LATE_RECOMPUTE" -eq 0 ]]; then
    echo
    echo "[PASS] L3 Cost Model V3 smoke validation passed."
else
    echo
    echo "[FAIL] L3 Cost Model V3 smoke validation failed."
    exit 1
fi

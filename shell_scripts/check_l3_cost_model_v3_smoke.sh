#!/usr/bin/env bash
set -euo pipefail

LOG=/root/projects/sglang-qwen2-adaptive-prefill/hicache/logs/server_l3_cost_model_v3.log

if [[ ! -f "$LOG" ]]; then
    echo "[FAIL] Log file not found: $LOG"
    exit 1
fi

echo "============================================================"
echo "V3 SMOKE VALIDATION"
echo "============================================================"

P256_RECOMPUTE=$(awk '
    /\[HiCacheEarlyDecision\]/ &&
    /policy=cost_model/ &&
    /storage_hit_length=256/ &&
    /action=recompute/ {n++}
    END {print n+0}
' "$LOG")

P256_IO=$(awk '
    /\[HiCachePrefetchIO\]/ &&
    /completed_tokens=256/ {n++}
    END {print n+0}
' "$LOG")

P16384_RESTORE=$(awk '
    /\[HiCacheEarlyDecision\]/ &&
    /policy=cost_model/ &&
    /storage_hit_length=16384/ &&
    /action=restore/ {n++}
    END {print n+0}
' "$LOG")

P16384_PRECOMPUTED=$(awk '
    /\[HiCachePrefetchQuery\]/ &&
    /storage_hit_tokens=16384/ &&
    /query_source=precomputed/ {n++}
    END {print n+0}
' "$LOG")

P16384_WORKER=$(awk '
    /\[HiCachePrefetchQuery\]/ &&
    /storage_hit_tokens=16384/ &&
    /query_source=worker/ {n++}
    END {print n+0}
' "$LOG")

P16384_IO=$(awk '
    /\[HiCachePrefetchIO\]/ &&
    /completed_tokens=16384/ {n++}
    END {print n+0}
' "$LOG")

P16384_LATE_RECOMPUTE=$(awk '
    /\[HiCacheDecision\]/ &&
    /host_hit_length=16384/ &&
    /action=recompute/ {n++}
    END {print n+0}
' "$LOG")

echo
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

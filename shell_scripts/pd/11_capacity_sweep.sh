#!/usr/bin/env bash
set -euo pipefail

# ============================================================
# Arguments
# ============================================================
if [ $# -lt 1 ] || [ $# -gt 3 ]; then
    echo "Usage:"
    echo "$0 <colocated|pd> [repeats] [num_prompts]"
    echo
    echo "Defaults:"
    echo "  repeats     = 3"
    echo "  num_prompts = 300"
    echo
    echo "Examples:"
    echo "$0 colocated"
    echo "$0 colocated 3 300"
    echo "$0 pd 3 300"
    exit 1
fi

MODE=$1
REPEATS=${2:-3}
NUM_PROMPTS=${3:-300}

# ============================================================
# Validation
# ============================================================
if [ "$MODE" != "colocated" ] && [ "$MODE" != "pd" ]; then
    echo "ERROR: mode must be 'colocated' or 'pd'"
    exit 1
fi

if ! [[ "$REPEATS" =~ ^[1-9][0-9]*$ ]]; then
    echo "ERROR: repeats must be a positive integer"
    exit 1
fi

if ! [[ "$NUM_PROMPTS" =~ ^[1-9][0-9]*$ ]]; then
    echo "ERROR: num_prompts must be a positive integer"
    exit 1
fi

SCRIPT_DIR=$(cd "$(dirname "$0")" && pwd)

# ============================================================
# Capacity Sweep
#
# Workloads:
#   4K / 256, C = 8 / 16 / 24
#   8K / 256, C = 8 / 16 / 24
#
# request-rate = inf
# ============================================================
echo
echo "============================================================"
echo "CAPACITY SWEEP"
echo "============================================================"
echo "Mode          : $MODE"
echo "Repeats       : $REPEATS"
echo "Requests/run  : $NUM_PROMPTS"
echo "Input lengths : 4096, 8192"
echo "Output length : 256"
echo "Concurrency   : 8, 16, 24"
echo "Request rate  : inf"
echo "============================================================"

for INPUT in 4096 8192; do
    for CONC in 8 16 24; do
        for REP in $(seq 1 "$REPEATS"); do

            echo
            echo "################################################"
            echo "# CAPACITY SWEEP"
            echo "# MODE        = $MODE"
            echo "# INPUT       = $INPUT"
            echo "# OUTPUT      = 256"
            echo "# CONCURRENCY = $CONC"
            echo "# REQUESTS    = $NUM_PROMPTS"
            echo "# REPEAT      = $REP / $REPEATS"
            echo "################################################"

            BENCH_PHASE=capacity \
            "$SCRIPT_DIR/10_bench_one.sh" \
                "$MODE" \
                "$INPUT" \
                256 \
                "$CONC" \
                inf \
                "$NUM_PROMPTS"

            # Benchmark itself is synchronous.
            # Only leave a short interval for log/process cleanup.
            sleep 1
        done
    done
done

echo
echo "============================================================"
echo "CAPACITY SWEEP FINISHED"
echo "============================================================"
echo "Mode     : $MODE"
echo "Runs     : $((2 * 3 * REPEATS))"
echo "Requests : $NUM_PROMPTS per run"
echo "Results  : /root/autodl-tmp/sglang_pd_exp/results/raw/capacity/$MODE"
echo "Logs     : /root/autodl-tmp/sglang_pd_exp/logs/benchmark/capacity/$MODE"
echo "============================================================"

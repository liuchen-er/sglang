#!/usr/bin/env bash
set -euo pipefail

if [ $# -lt 1 ] || [ $# -gt 3 ]; then
    echo "Usage:"
    echo "$0 <colocated|pd> [repeats] [num_prompts]"
    echo
    echo "Defaults:"
    echo "  repeats     = 3"
    echo "  num_prompts = 300"
    exit 1
fi

MODE=$1
REPEATS=${2:-3}
NUM_PROMPTS=${3:-300}

SCRIPT_DIR=$(cd "$(dirname "$0")" && pwd)

run_case() {
    local INPUT=$1
    local RATE=$2

    for REP in $(seq 1 "$REPEATS"); do

        echo
        echo "################################################"
        echo "# EQUAL OFFERED LOAD"
        echo "# MODE         = $MODE"
        echo "# INPUT        = $INPUT"
        echo "# OUTPUT       = 256"
        echo "# CONCURRENCY  = 24"
        echo "# REQUEST RATE = $RATE req/s"
        echo "# REQUESTS     = $NUM_PROMPTS"
        echo "# REPEAT       = $REP / $REPEATS"
        echo "################################################"

        "$SCRIPT_DIR/10_bench_one.sh" \
            "$MODE" \
            "$INPUT" \
            256 \
            24 \
            "$RATE" \
            "$NUM_PROMPTS"

        sleep 1
    done
}

# 50% 75% 90%
# 4K: based on PD saturation ≈ 1.45 req/s
for RATE in 0.73 1.09 1.30; do
    run_case 4096 "$RATE"
done

# 8K: based on PD saturation ≈ 0.60 req/s
for RATE in 0.30 0.45 0.54; do
    run_case 8192 "$RATE"
done

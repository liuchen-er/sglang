#!/usr/bin/env bash
set -euo pipefail

if [ $# -lt 1 ] || [ $# -gt 3 ]; then
    echo "Usage:"
    echo "$0 <colocated|pd> [repeats] [num_prompts]"
    echo
    echo "Defaults:"
    echo "  repeats     = 3"
    echo "  num_prompts = 300"
    echo
    echo "Example:"
    echo "$0 colocated"
    echo "$0 colocated 3 300"
    echo "$0 pd 3 500"
    exit 1
fi

MODE=$1
REPEATS=${2:-3}
NUM_PROMPTS=${3:-300}

if [ "$MODE" != "colocated" ] && [ "$MODE" != "pd" ]; then
    echo "ERROR: mode must be colocated or pd"
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

for INPUT in 4096 8192; do
    for CONC in 8 16 24; do
        for REP in $(seq 1 "$REPEATS"); do

            echo
            echo "################################################"
            echo "# MODE        = $MODE"
            echo "# INPUT       = $INPUT"
            echo "# OUTPUT      = 256"
            echo "# CONCURRENCY = $CONC"
            echo "# REQUESTS    = $NUM_PROMPTS"
            echo "# REPEAT      = $REP / $REPEATS"
            echo "################################################"

            "$SCRIPT_DIR/10_bench_one.sh" \
                "$MODE" \
                "$INPUT" \
                256 \
                "$CONC" \
                inf \
                "$NUM_PROMPTS"

            sleep 5
        done
    done
done

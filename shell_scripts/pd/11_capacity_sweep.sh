#!/usr/bin/env bash
set -euo pipefail

if [ $# -ne 1 ]; then
    echo "Usage: $0 <colocated|pd>"
    exit 1
fi

MODE=$1

SCRIPT_DIR=$(cd "$(dirname "$0")" && pwd)

for INPUT in 4096 8192; do
    for CONC in 8 16 24; do

        echo
        echo "############################################"
        echo "# MODE=$MODE"
        echo "# INPUT=$INPUT"
        echo "# OUTPUT=256"
        echo "# CONCURRENCY=$CONC"
        echo "############################################"

        "$SCRIPT_DIR/10_bench_one.sh" \
            "$MODE" \
            "$INPUT" \
            256 \
            "$CONC" \
            inf

        sleep 5
    done
done

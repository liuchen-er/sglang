#!/bin/bash
set -euo pipefail

SCRIPT_DIR=/root/projects/sglang-qwen2-adaptive-prefill/sglang/shell_scripts/quantization

# 默认先跑 1 轮，确认所有配置都正常。
# 正式实验时可以传 3，表示每个配置重复 3 次。
REPEAT=${1:-1}
NUM_PROMPTS=${2:-64}

CONFIGS=(
    "4096 128 1"
    "4096 128 4"
    "4096 128 16"

    "1024 512 1"
    "1024 512 4"
    "1024 512 16"

    "128 1024 1"
    "128 1024 4"
    "128 1024 16"
)

echo "========================================"
echo "Qwen2.5-7B BF16 Benchmark Matrix"
echo "Repeat:      $REPEAT"
echo "Num Prompts: $NUM_PROMPTS"
echo "========================================"

for ((run=1; run<=REPEAT; run++)); do

    echo
    echo "========== Repeat $run / $REPEAT =========="

    for config in "${CONFIGS[@]}"; do

        read -r INPUT_LEN OUTPUT_LEN CONCURRENCY <<< "$config"

        echo
        echo "----------------------------------------"
        echo "Run:         $run"
        echo "Input Len:   $INPUT_LEN"
        echo "Output Len:  $OUTPUT_LEN"
        echo "Concurrency: $CONCURRENCY"
        echo "----------------------------------------"

        "$SCRIPT_DIR/02_benchmark_bf16.sh" \
            "$INPUT_LEN" \
            "$OUTPUT_LEN" \
            "$CONCURRENCY" \
            "$NUM_PROMPTS"

        sleep 3
    done
done
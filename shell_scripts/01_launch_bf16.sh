#!/bin/bash
set -e

source /root/autodl-tmp/venvs/sglang015/bin/activate

SGLANG_ROOT=/root/projects/sglang-qwen2-adaptive-prefill/sglang
EXP_ROOT=/root/autodl-tmp/qwen25_quant
MODEL=/root/autodl-tmp/models/Qwen2.5-7B-Instruct
PORT=30000

cd "$SGLANG_ROOT"

mkdir -p "$EXP_ROOT/logs"

echo "========================================"
echo "Qwen2.5-7B-Instruct BF16 Baseline"
echo "MODEL: $MODEL"
echo "PORT:  $PORT"
echo "========================================"

python -m sglang.launch_server \
    --model-path "$MODEL" \
    --dtype bfloat16 \
    --host 0.0.0.0 \
    --port "$PORT" \
    --mem-fraction-static 0.80 \
    2>&1 | tee "$EXP_ROOT/logs/bf16_server.log"
#!/bin/bash
set -euo pipefail

source /root/autodl-tmp/venvs/sglang015/bin/activate

export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1

SGLANG_ROOT=/root/projects/sglang-qwen2-adaptive-prefill/sglang
EXP_ROOT=/root/autodl-tmp/qwen25_quant

MODEL=$EXP_ROOT/quantized_models/Qwen2.5-7B-Instruct-W8A8-smoke

PORT=30000

mkdir -p "$EXP_ROOT/logs/w8a8"

TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
LOG="$EXP_ROOT/logs/w8a8/w8a8_smoke_server_${TIMESTAMP}.log"

cd "$SGLANG_ROOT"

echo "========================================"
echo "Qwen2.5-7B W8A8 Smoke Server"
echo "MODEL: $MODEL"
echo "PORT:  $PORT"
echo "LOG:   $LOG"
echo "========================================"

python -m sglang.launch_server \
    --model-path "$MODEL" \
    --dtype bfloat16 \
    --quantization compressed-tensors \
    --host 0.0.0.0 \
    --port "$PORT" \
    --mem-fraction-static 0.80 \
    2>&1 | tee "$LOG"
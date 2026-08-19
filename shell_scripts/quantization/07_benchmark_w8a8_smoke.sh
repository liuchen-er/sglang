#!/bin/bash
set -euo pipefail

source /root/autodl-tmp/venvs/sglang015/bin/activate

SGLANG_ROOT=/root/projects/sglang-qwen2-adaptive-prefill/sglang
EXP_ROOT=/root/autodl-tmp/qwen25_quant

MODEL=$EXP_ROOT/quantized_models/Qwen2.5-7B-Instruct-W8A8-smoke

INPUT_LEN=${1:-128}
OUTPUT_LEN=${2:-128}
CONCURRENCY=${3:-1}
NUM_PROMPTS=${4:-8}

cd "$SGLANG_ROOT"

export PYTHONPATH="$SGLANG_ROOT/python:${PYTHONPATH:-}"

RESULT_DIR="$EXP_ROOT/results/w8a8_smoke"
mkdir -p "$RESULT_DIR"

TIMESTAMP=$(date +"%Y%m%d_%H%M%S")

TAG="w8a8_smoke_in${INPUT_LEN}_out${OUTPUT_LEN}_c${CONCURRENCY}_n${NUM_PROMPTS}_${TIMESTAMP}"

OUT="$RESULT_DIR/${TAG}.jsonl"
LOG="$RESULT_DIR/${TAG}.log"

echo "========================================"
echo "W8A8 Smoke Benchmark"
echo "MODEL:        $MODEL"
echo "INPUT_LEN:    $INPUT_LEN"
echo "OUTPUT_LEN:   $OUTPUT_LEN"
echo "CONCURRENCY:  $CONCURRENCY"
echo "NUM_PROMPTS:  $NUM_PROMPTS"
echo "OUTPUT:       $OUT"
echo "LOG:          $LOG"
echo "========================================"

python -m sglang.benchmark.serving \
    --backend sglang \
    --host 127.0.0.1 \
    --port 30000 \
    --model "$MODEL" \
    --tokenizer "$MODEL" \
    --dataset-name random-ids \
    --tokenize-prompt \
    --seed 42 \
    --num-prompts "$NUM_PROMPTS" \
    --random-input-len "$INPUT_LEN" \
    --random-output-len "$OUTPUT_LEN" \
    --random-range-ratio 1.0 \
    --max-concurrency "$CONCURRENCY" \
    --flush-cache \
    --output-file "$OUT" \
    2>&1 | tee "$LOG"
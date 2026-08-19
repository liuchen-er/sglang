#!/bin/bash
set -e

source /root/autodl-tmp/venvs/sglang015/bin/activate

SGLANG_ROOT=/root/projects/sglang-qwen2-adaptive-prefill/sglang
EXP_ROOT=/root/autodl-tmp/qwen25_quant
MODEL=/root/autodl-tmp/models/Qwen2.5-7B-Instruct

INPUT_LEN=${1:-512}
OUTPUT_LEN=${2:-128}
CONCURRENCY=${3:-4}
NUM_PROMPTS=${4:-20}

cd "$SGLANG_ROOT"

RESULT_DIR="$EXP_ROOT/results/bf16"
mkdir -p "$RESULT_DIR"

TIMESTAMP=$(date +"%Y%m%d_%H%M%S")

TAG="bf16_in${INPUT_LEN}_out${OUTPUT_LEN}_c${CONCURRENCY}_n${NUM_PROMPTS}_${TIMESTAMP}"

OUT="$RESULT_DIR/${TAG}.jsonl"
LOG="$RESULT_DIR/${TAG}.log"

echo "========================================"
echo "Qwen2.5-7B BF16 Benchmark"
echo "Input Len:       $INPUT_LEN"
echo "Output Len:      $OUTPUT_LEN"
echo "Concurrency:     $CONCURRENCY"
echo "Num Prompts:     $NUM_PROMPTS"
echo "Seed:            42"
echo "Result JSONL:    $OUT"
echo "Log:             $LOG"
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
    --output-file "$OUT" \
    2>&1 | tee "$LOG"
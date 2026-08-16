#!/usr/bin/env bash

# ============================================================
# Source code
# ============================================================
export REPO=/root/projects/sglang-qwen2-adaptive-prefill/sglang

# ============================================================
# Model / experiment data disk
# ============================================================
export MODEL_PATH=/root/autodl-tmp/models/Qwen2.5-7B-Instruct
export EXP_ROOT=/root/autodl-tmp/sglang_pd_exp

export LOG_ROOT="$EXP_ROOT/logs"
export RESULT_ROOT="$EXP_ROOT/results"
export SERVER_LOG_ROOT="$LOG_ROOT/server"
export BENCH_LOG_ROOT="$LOG_ROOT/benchmark"

export RESULT_RAW_ROOT="$RESULT_ROOT/raw"
export RESULT_SUMMARY_ROOT="$RESULT_ROOT/summary"
export PROFILE_ROOT="$EXP_ROOT/profiles"
export RUN_ROOT="$EXP_ROOT/run"
export META_ROOT="$EXP_ROOT/meta"
export TMP_ROOT="$EXP_ROOT/tmp"
export CACHE_ROOT="$EXP_ROOT/cache"

mkdir -p \
    "$LOG_ROOT" \
    "$RESULT_ROOT/raw" \
    "$RESULT_ROOT/summary" \
    "$PROFILE_ROOT" \
    "$RUN_ROOT" \
    "$META_ROOT" \
    "$TMP_ROOT" \
    "$CACHE_ROOT"

# ============================================================
# Move runtime/cache writes to data disk
# ============================================================
export TMPDIR="$TMP_ROOT"
export XDG_CACHE_HOME="$CACHE_ROOT/xdg"
export HF_HOME="$CACHE_ROOT/huggingface"
export TRANSFORMERS_CACHE="$CACHE_ROOT/huggingface/transformers"
export TRITON_CACHE_DIR="$CACHE_ROOT/triton"
export TORCHINDUCTOR_CACHE_DIR="$CACHE_ROOT/torchinductor"
export CUDA_CACHE_PATH="$CACHE_ROOT/cuda"
export PYTHONPYCACHEPREFIX="$CACHE_ROOT/pycache"
export SGLANG_TORCH_PROFILER_DIR="$PROFILE_ROOT"

mkdir -p \
    "$XDG_CACHE_HOME" \
    "$HF_HOME" \
    "$TRITON_CACHE_DIR" \
    "$TORCHINDUCTOR_CACHE_DIR" \
    "$CUDA_CACHE_PATH" \
    "$PYTHONPYCACHEPREFIX"

# ============================================================
# Network / ports
# ============================================================
export HOST=127.0.0.1
export WORKER0_PORT=30000
export WORKER1_PORT=30001
export ROUTER_PORT=8000

# ============================================================
# Python environment
# ============================================================
source /root/autodl-tmp/venvs/sglang015/bin/activate

# ============================================================
# Fixed baseline parameters
# ============================================================
COMMON_SERVER_ARGS=(
    --model-path "$MODEL_PATH"
    --host "$HOST"
    --chunked-prefill-size 2048
    --max-prefill-tokens 16384
    --context-length 32768
    --page-size 16
)

# ============================================================
# Router CLI
# ============================================================
if command -v sglang-router >/dev/null 2>&1; then
    ROUTER_BIN="sglang-router"
elif command -v smg >/dev/null 2>&1; then
    ROUTER_BIN="smg"
else
    echo "ERROR: sglang-router / smg not found"
    return 1 2>/dev/null || exit 1
fi

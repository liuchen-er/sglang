#!/usr/bin/env bash

export PROJECT_ROOT="/root/projects/sglang-qwen2-adaptive-prefill"
export SGLANG_ROOT="${PROJECT_ROOT}/sglang"
export MODEL_PATH="/root/autodl-tmp/models/Qwen2.5-1.5B-Instruct-ms"

export HOST="${HOST:-127.0.0.1}"
export PORT="${PORT:-30000}"
export BASE_URL="http://${HOST}:${PORT}"

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"

# 必须加入的是包含 sglang 包目录的父目录。
export PYTHONPATH="${SGLANG_ROOT}/python${PYTHONPATH:+:${PYTHONPATH}}"

export PYTHONUNBUFFERED=1
export PYTHONHASHSEED=0
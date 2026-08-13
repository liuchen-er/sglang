#!/usr/bin/env bash
set -euo pipefail

REPO=/root/projects/sglang-qwen2-adaptive-prefill/sglang
DATA=/root/projects/sglang-qwen2-adaptive-prefill/hicache
VENV=/root/autodl-tmp/venvs/sglang015/bin/activate
BASE_URL=http://127.0.0.1:30000

: "${MODEL_PATH:?Please export MODEL_PATH first}"

TRIALS="${TRIALS:-3}"
TARGETS_PER_STATE="${TARGETS_PER_STATE:-3}"
PREFIX_LENGTHS="${PREFIX_LENGTHS:-512,1024,2048,4096,8192,16384}"
RUNNING_BS_LIST="${RUNNING_BS_LIST:-0,4,16,32}"
NUM_EVICTORS="${NUM_EVICTORS:-22}"
EVICTOR_LEN="${EVICTOR_LEN:-29000}"
BG_OUTPUT_LEN="${BG_OUTPUT_LEN:-2048}"

SERVER_PID=""

source "$VENV"
cd "$REPO"
mkdir -p "$DATA/results" "$DATA/logs" "$DATA/profiles"

cleanup_server() {
    if [[ -n "${SERVER_PID:-}" ]]; then
        kill -TERM -- "-$SERVER_PID" 2>/dev/null || true
        for _ in $(seq 1 30); do
            curl -sf "$BASE_URL/health" >/dev/null 2>&1 || break
            sleep 1
        done
        kill -KILL -- "-$SERVER_PID" 2>/dev/null || true
        wait "$SERVER_PID" 2>/dev/null || true
        SERVER_PID=""
        sleep 2
    fi
}

trap cleanup_server EXIT INT TERM

wait_server() {
    for i in $(seq 1 600); do
        if curl -sf "$BASE_URL/health" >/dev/null 2>&1; then
            echo "[READY] server"
            sleep 2
            return
        fi
        if ! kill -0 "$SERVER_PID" 2>/dev/null; then
            echo "[ERROR] server exited"
            exit 1
        fi
        sleep 1
    done
    echo "[ERROR] server startup timeout"
    exit 1
}

run_policy() {
    local policy="$1"
    local tag="paired_${policy}"
    local result="$DATA/results/paired_${policy}.jsonl"

    rm -f "$result" "$DATA/logs/server_${tag}.log"

    echo "============================================================"
    echo "START PROFILE: $policy"
    echo "============================================================"

    setsid bash -c "
        source '$VENV'
        cd '$REPO'
        export MODEL_PATH='$MODEL_PATH'
        LOG_TAG='$tag' ./shell_scripts/start_hicache_server.sh '$policy'
    " >"$DATA/logs/launcher_${tag}.log" 2>&1 &

    SERVER_PID=$!
    wait_server

    POLICY="$policy" \
    PREFIX_LENGTHS="$PREFIX_LENGTHS" \
    RUNNING_BS_LIST="$RUNNING_BS_LIST" \
    TARGETS_PER_STATE="$TARGETS_PER_STATE" \
    TRIALS="$TRIALS" \
    NUM_EVICTORS="$NUM_EVICTORS" \
    EVICTOR_LEN="$EVICTOR_LEN" \
    BG_OUTPUT_LEN="$BG_OUTPUT_LEN" \
    RESULT_FILE="$result" \
    PYTHONUNBUFFERED=1 \
    python shell_scripts/bench_hicache_paired.py \
        2>&1 | tee "$DATA/logs/bench_paired_${policy}.log"

    cleanup_server
}

rm -f \
    "$DATA/results/paired_always_restore.jsonl" \
    "$DATA/results/paired_always_recompute.jsonl"

run_policy always_restore
run_policy always_recompute

python shell_scripts/build_hicache_cost_profile_v2.py

echo
echo "============================================================"
echo "PAIR PROFILE COMPLETE"
echo "$DATA/profiles/hicache_cost_profile_v2.json"
echo "============================================================"

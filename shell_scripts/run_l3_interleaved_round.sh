#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 1 ]]; then
    echo "Usage: $0 <round_id: 1|2|3>"
    exit 1
fi

ROUND="$1"

case "$ROUND" in
    1)
        ORDER=("restore" "recompute" "v3" "v31")
        ;;
    2)
        ORDER=("recompute" "v31" "restore" "v3")
        ;;
    3)
        ORDER=("v31" "v3" "recompute" "restore")
        ;;
    *)
        echo "Invalid round: $ROUND"
        echo "Expected: 1, 2, or 3"
        exit 1
        ;;
esac

REPO=/root/projects/sglang-qwen2-adaptive-prefill/sglang
DATA=/root/projects/sglang-qwen2-adaptive-prefill/hicache
RESULT_DIR="$DATA/results/l3_interleaved/round${ROUND}"
LOG_DIR="$DATA/logs"

V3_PROFILE="$DATA/profiles/hicache_cost_profile_v3.json"
V31_PROFILE="$DATA/profiles/hicache_cost_profile_v31.json"

BASE_URL=http://127.0.0.1:30000

cd "$REPO"
source /root/autodl-tmp/venvs/sglang015/bin/activate

export MODEL_PATH=/root/autodl-tmp/models/Qwen2.5-1.5B-Instruct-ms
export ADMIN_API_KEY=hicache-admin-local
export SGLANG_HICACHE_FILE_BACKEND_STORAGE_DIR=/root/autodl-tmp/hicache_l3
export HICACHE_MARGIN_MS=2.0
export HICACHE_BENCH_TIMEOUT=600

mkdir -p "$RESULT_DIR" "$LOG_DIR"

SERVER_PID=""

cleanup_server() {
    if [[ -n "${SERVER_PID:-}" ]]; then
        if kill -0 "$SERVER_PID" 2>/dev/null; then
            echo
            echo "[Server] stopping process group pid=$SERVER_PID ..."
            kill -TERM -- "-$SERVER_PID" 2>/dev/null || true

            for _ in $(seq 1 20); do
                if ! kill -0 "$SERVER_PID" 2>/dev/null; then
                    break
                fi
                sleep 1
            done

            if kill -0 "$SERVER_PID" 2>/dev/null; then
                echo "[Server] TERM timeout, sending KILL ..."
                kill -KILL -- "-$SERVER_PID" 2>/dev/null || true
            fi
        fi

        wait "$SERVER_PID" 2>/dev/null || true
        SERVER_PID=""
    fi

    for _ in $(seq 1 30); do
        HTTP_CODE=$(
            curl -sS --max-time 1 \
                -o /dev/null \
                -w '%{http_code}' \
                "$BASE_URL/health" \
                2>/dev/null || true
        )

        if [[ "$HTTP_CODE" != "200" ]]; then
            break
        fi

        sleep 1
    done

    sleep 3
}

trap cleanup_server EXIT INT TERM

ensure_port_free() {
    local code

    code=$(
        curl -s \
            --connect-timeout 1 \
            --max-time 3 \
            -o /dev/null \
            -w '%{http_code}' \
            "$BASE_URL/health" \
            2>/dev/null || true
    )

    if [[ "$code" == "200" ]]; then
        echo "[ERROR] Port 30000 already has a running SGLang server."
        echo "Stop the existing server before running this script."
        exit 1
    fi
}

wait_server_ready() {
    local launcher_log="$1"
    local start_time
    local now
    local elapsed
    local http_code

    start_time=$(date +%s)

    echo "[Server] waiting for /health ..."
    echo "[Server] startup timeout=600s, curl timeout=5s"

    while true; do
        if ! kill -0 "$SERVER_PID" 2>/dev/null; then
            echo "[ERROR] Server exited before becoming healthy."
            echo "===== launcher log tail ====="
            tail -80 "$launcher_log" || true
            exit 1
        fi

        now=$(date +%s)
        elapsed=$((now - start_time))

        if (( elapsed >= 600 )); then
            echo "[ERROR] Server health check timed out after ${elapsed}s."
            echo "===== launcher log tail ====="
            tail -80 "$launcher_log" || true
            exit 1
        fi

        http_code=$(
            curl -s \
                --connect-timeout 2 \
                --max-time 5 \
                -o /dev/null \
                -w '%{http_code}' \
                "$BASE_URL/health" \
                2>/dev/null || true
        )

        if [[ "$http_code" == "200" ]]; then
            now=$(date +%s)
            elapsed=$((now - start_time))
            echo "[Server] ready after ${elapsed}s."
            sleep 3
            return
        fi

        if (( elapsed % 30 < 3 )); then
            echo "[Server] still starting... elapsed=${elapsed}s http=${http_code:-none}"
        fi

        sleep 2
    done
}

start_server() {
    local label="$1"
    local policy="$2"
    local profile="${3:-}"

    ensure_port_free

    local log_tag="final_r${ROUND}_${label}"
    local launcher_log="$LOG_DIR/launcher_${log_tag}.log"

    echo
    echo "============================================================"
    echo "START SERVER"
    echo "round=$ROUND"
    echo "label=$label"
    echo "policy=$policy"
    if [[ -n "$profile" ]]; then
        echo "profile=$profile"
    fi
    echo "============================================================"

    if [[ "$policy" == "cost_model" ]]; then
        if [[ ! -f "$profile" ]]; then
            echo "[ERROR] Cost profile not found: $profile"
            exit 1
        fi
        export HICACHE_COST_PROFILE="$profile"
    else
        unset HICACHE_COST_PROFILE || true
    fi

    setsid env \
        LOG_TAG="$log_tag" \
        bash ./shell_scripts/start_hicache_l3_server.sh "$policy" \
        >"$launcher_log" 2>&1 &

    SERVER_PID=$!

    echo "[Server] launcher pid=$SERVER_PID"
    echo "[Server] launcher log=$launcher_log"
    echo "[Server] server log=$LOG_DIR/server_${log_tag}.log"

    wait_server_ready "$launcher_log"
}

run_group() {
    local label="$1"
    local policy="$2"
    local concurrency="$3"
    local prefixes="$4"

    local result_file="$RESULT_DIR/${label}_c${concurrency}.jsonl"

    echo
    echo "------------------------------------------------------------"
    echo "BENCHMARK"
    echo "round=$ROUND"
    echo "label=$label"
    echo "policy=$policy"
    echo "concurrency=$concurrency"
    echo "prefixes=$prefixes"
    echo "result=$result_file"
    echo "------------------------------------------------------------"

    TARGET_CACHE_TIER=L3 \
    L3_PREP_MODE=flush \
    CLEAR_L3=1 \
    VALIDATE_CACHE_TIER=0 \
    POLICY="$policy" \
    PREFIX_LENGTHS="$prefixes" \
    SESSIONS_PER_LEN=32 \
    TRIALS=1 \
    MAX_CONCURRENCY="$concurrency" \
    REQUEST_RATE=inf \
    OUTPUT_LEN=1 \
    PAGE_SIZE=64 \
    L3_BACKUP_TIMEOUT_S=180 \
    RESET_RESULT=1 \
    RESULT_FILE="$result_file" \
    python shell_scripts/bench_hicache_agent.py
}

run_policy_grid() {
    local label="$1"
    local policy="$2"

    run_group "$label" "$policy" 1 "256,8192,16384"
    run_group "$label" "$policy" 16 "256,4096,8192,16384"
    run_group "$label" "$policy" 32 "256,4096,8192"
}

run_one_policy() {
    local label="$1"

    case "$label" in
        restore)
            start_server "restore" "always_restore"
            run_policy_grid "restore" "always_restore"
            ;;

        recompute)
            start_server "recompute" "always_recompute"
            run_policy_grid "recompute" "always_recompute"
            ;;

        v3)
            start_server "v3" "cost_model" "$V3_PROFILE"
            run_policy_grid "v3" "cost_model"
            ;;

        v31)
            start_server "v31" "cost_model" "$V31_PROFILE"
            run_policy_grid "v31" "cost_model"
            ;;

        *)
            echo "[ERROR] Unknown label: $label"
            exit 1
            ;;
    esac

    cleanup_server

    echo
    echo "[DONE] round=$ROUND policy=$label"
}

echo "============================================================"
echo "L3 INTERLEAVED FINAL BENCHMARK"
echo "Round: $ROUND"
echo "Order: ${ORDER[*]}"
echo "Result dir: $RESULT_DIR"
echo "============================================================"

ensure_port_free

for label in "${ORDER[@]}"; do
    run_one_policy "$label"
done

echo
echo "============================================================"
echo "ROUND $ROUND FINISHED"
echo "============================================================"
echo "Order: ${ORDER[*]}"
echo "Results:"
find "$RESULT_DIR" -maxdepth 1 -type f -name '*.jsonl' -printf '  %f\n' | sort

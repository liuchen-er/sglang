#!/usr/bin/env bash
set -euo pipefail

REPO=/root/projects/sglang-qwen2-adaptive-prefill/sglang
DATA=/root/projects/sglang-qwen2-adaptive-prefill/hicache
VENV=/root/autodl-tmp/venvs/sglang015/bin/activate
BASE_URL=http://127.0.0.1:30000

: "${MODEL_PATH:?Please export MODEL_PATH before running this script}"

PREFIX_LENGTHS="${PREFIX_LENGTHS:-512,1024,2048,4096,8192,16384}"
SESSIONS_PER_LEN="${SESSIONS_PER_LEN:-32}"
TRIALS="${TRIALS:-8}"
NUM_EVICTORS="${NUM_EVICTORS:-22}"
EVICTOR_LEN="${EVICTOR_LEN:-29000}"
OUTPUT_LEN="${OUTPUT_LEN:-128}"
TAIL_LEN="${TAIL_LEN:-64}"
REQUEST_RATE="${REQUEST_RATE:-inf}"
THRESHOLD="${THRESHOLD:-960}"
COST_PROFILE="${HICACHE_COST_PROFILE:-$DATA/profiles/hicache_cost_profile_v2.json}"
MARGIN_MS="${HICACHE_MARGIN_MS:-1.0}"
CONCURRENCIES=(4 16 32)
POLICIES=(always_restore token_threshold cost_model)

TIMESTAMP=$(date +%Y%m%d_%H%M%S)
RUN_LOG="$DATA/logs/run_agent_all_${TIMESTAMP}.log"
ARCHIVE="$DATA/archive/agent_b_${TIMESTAMP}"

SERVER_PID=""

source "$VENV"
cd "$REPO"
mkdir -p "$DATA/results" "$DATA/logs" "$DATA/archive" "$ARCHIVE/results" "$ARCHIVE/logs"

log() {
    echo "[$(date '+%F %T')] $*" | tee -a "$RUN_LOG"
}

cleanup_server() {
    if [[ -n "${SERVER_PID:-}" ]]; then
        log "Stopping server process group: $SERVER_PID"
        kill -TERM -- "-$SERVER_PID" 2>/dev/null || true
        for _ in $(seq 1 30); do
            if ! curl -sf "$BASE_URL/health" >/dev/null 2>&1; then
                break
            fi
            sleep 1
        done
        kill -KILL -- "-$SERVER_PID" 2>/dev/null || true
        wait "$SERVER_PID" 2>/dev/null || true
        SERVER_PID=""
        sleep 2
    fi
}

cleanup() {
    cleanup_server
}
trap cleanup EXIT INT TERM

archive_old_files() {
    log "Archiving previous B experiment files to $ARCHIVE"

    shopt -s nullglob
    for f in "$DATA"/results/agent_always_restore_c*.jsonl \
             "$DATA"/results/agent_token_threshold_c*.jsonl \
             "$DATA"/results/agent_cost_model_c*.jsonl; do
        mv "$f" "$ARCHIVE/results/"
    done

    for f in "$DATA"/logs/server_agent_always_restore.log \
             "$DATA"/logs/server_agent_token_threshold.log \
             "$DATA"/logs/server_agent_cost_model.log \
             "$DATA"/logs/bench_agent_*.log; do
        [[ -e "$f" ]] && mv "$f" "$ARCHIVE/logs/"
    done
    shopt -u nullglob
}

check_environment() {
    [[ -f "$REPO/shell_scripts/start_hicache_server.sh" ]] || {
        echo "Missing start_hicache_server.sh"
        exit 1
    }

    [[ -f "$REPO/shell_scripts/bench_hicache_agent.py" ]] || {
        echo "Missing bench_hicache_agent.py"
        exit 1
    }

    [[ -f "$REPO/shell_scripts/analyze_hicache_agent.py" ]] || {
        echo "Missing analyze_hicache_agent.py"
        exit 1
    }

    [[ -f "$COST_PROFILE" ]] || {
        echo "Missing cost profile: $COST_PROFILE"
        exit 1
    }

    python -m py_compile shell_scripts/bench_hicache_agent.py
    python -m py_compile shell_scripts/analyze_hicache_agent.py

    if curl -sf "$BASE_URL/health" >/dev/null 2>&1; then
        echo "Port 30000 already has a running SGLang server. Stop it before running this script."
        exit 1
    fi
}

wait_server() {
    local log_file="$1"
    log "Waiting for server startup..."

    for i in $(seq 1 600); do
        if curl -sf "$BASE_URL/health" >/dev/null 2>&1; then
            log "Server is ready."
            sleep 2
            return 0
        fi

        if ! kill -0 "$SERVER_PID" 2>/dev/null; then
            log "Server process exited during startup."
            tail -80 "$log_file" || true
            return 1
        fi

        if (( i % 30 == 0 )); then
            log "Still waiting for server... ${i}s"
        fi
        sleep 1
    done

    log "Server startup timeout."
    tail -100 "$log_file" || true
    return 1
}

start_server() {
    local policy="$1"
    local tag="agent_${policy}"
    local launcher_log="$DATA/logs/launcher_${tag}_${TIMESTAMP}.log"

    log "============================================================"
    log "Starting server: policy=$policy"
    log "============================================================"

    if [[ "$policy" == "always_restore" ]]; then
        setsid bash -c "source '$VENV'; cd '$REPO'; export MODEL_PATH='$MODEL_PATH'; LOG_TAG='$tag' ./shell_scripts/start_hicache_server.sh always_restore" \
            >"$launcher_log" 2>&1 &
    elif [[ "$policy" == "token_threshold" ]]; then
        setsid bash -c "source '$VENV'; cd '$REPO'; export MODEL_PATH='$MODEL_PATH'; LOG_TAG='$tag' ./shell_scripts/start_hicache_server.sh token_threshold '$THRESHOLD'" \
            >"$launcher_log" 2>&1 &
    elif [[ "$policy" == "cost_model" ]]; then
        setsid bash -c "source '$VENV'; cd '$REPO'; export MODEL_PATH='$MODEL_PATH'; export HICACHE_COST_PROFILE='$COST_PROFILE'; export HICACHE_MARGIN_MS='$MARGIN_MS'; LOG_TAG='$tag' ./shell_scripts/start_hicache_server.sh cost_model" \
            >"$launcher_log" 2>&1 &
    else
        log "Unknown policy: $policy"
        exit 1
    fi

    SERVER_PID=$!
    log "Server process group PID=$SERVER_PID"
    wait_server "$launcher_log"
}

run_benchmark() {
    local policy="$1"
    local concurrency="$2"
    local result="$DATA/results/agent_${policy}_c${concurrency}.jsonl"
    local bench_log="$DATA/logs/bench_agent_${policy}_c${concurrency}.log"

    log "------------------------------------------------------------"
    log "Benchmark: policy=$policy concurrency=$concurrency"
    log "result=$result"
    log "------------------------------------------------------------"

    rm -f "$result"

    POLICY="$policy" \
    MAX_CONCURRENCY="$concurrency" \
    PREFIX_LENGTHS="$PREFIX_LENGTHS" \
    SESSIONS_PER_LEN="$SESSIONS_PER_LEN" \
    TRIALS="$TRIALS" \
    REQUEST_RATE="$REQUEST_RATE" \
    NUM_EVICTORS="$NUM_EVICTORS" \
    EVICTOR_LEN="$EVICTOR_LEN" \
    OUTPUT_LEN="$OUTPUT_LEN" \
    TAIL_LEN="$TAIL_LEN" \
    RESULT_FILE="$result" \
    PYTHONUNBUFFERED=1 \
    python shell_scripts/bench_hicache_agent.py 2>&1 | tee "$bench_log"

    log "Benchmark finished: policy=$policy concurrency=$concurrency"

    local server_log="$DATA/logs/server_agent_${policy}.log"
    if [[ -f "$server_log" ]]; then
        local decisions restores recomputes
        decisions=$(grep "HiCacheDecision" "$server_log" | grep -c "agent_target_${policy}_c${concurrency}_" || true)
        restores=$(grep "HiCacheDecision" "$server_log" | grep "agent_target_${policy}_c${concurrency}_" | grep -c "action=restore" || true)
        recomputes=$(grep "HiCacheDecision" "$server_log" | grep "agent_target_${policy}_c${concurrency}_" | grep -c "action=recompute" || true)

        log "Decision check: total=$decisions restore=$restores recompute=$recomputes"

        if [[ "$policy" == "always_restore" && "$restores" -eq 0 ]]; then
            log "ERROR: always_restore produced zero restore decisions."
            exit 1
        fi
    fi
}

run_policy() {
    local policy="$1"

    start_server "$policy"

    for concurrency in "${CONCURRENCIES[@]}"; do
        run_benchmark "$policy" "$concurrency"
    done

    cleanup_server
    log "Policy completed: $policy"
}

analyze_results() {
    log "============================================================"
    log "Analyzing all B experiment results"
    log "============================================================"

    POLICIES=always_restore,token_threshold,cost_model \
    CONCURRENCIES=4,16,32 \
    python shell_scripts/analyze_hicache_agent.py \
        2>&1 | tee "$DATA/results/agent_analysis_${TIMESTAMP}.txt"

    log "Analysis saved to $DATA/results/agent_analysis_${TIMESTAMP}.txt"
}

main() {
    log "============================================================"
    log "SGLang HiCache Agent L2-hit Full Benchmark"
    log "============================================================"
    log "MODEL_PATH=$MODEL_PATH"
    log "PREFIX_LENGTHS=$PREFIX_LENGTHS"
    log "SESSIONS_PER_LEN=$SESSIONS_PER_LEN"
    log "TRIALS=$TRIALS"
    log "CONCURRENCIES=${CONCURRENCIES[*]}"
    log "EVICTORS=${NUM_EVICTORS}x${EVICTOR_LEN}"
    log "THRESHOLD=$THRESHOLD"
    log "COST_PROFILE=$COST_PROFILE"
    log "MARGIN_MS=$MARGIN_MS"

    check_environment
    archive_old_files

    for policy in "${POLICIES[@]}"; do
        run_policy "$policy"
    done

    analyze_results

    log "============================================================"
    log "ALL B EXPERIMENTS COMPLETED"
    log "============================================================"
}

main "$@"

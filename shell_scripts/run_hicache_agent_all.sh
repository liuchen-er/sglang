#!/usr/bin/env bash
set -euo pipefail

REPO=/root/projects/sglang-qwen2-adaptive-prefill/sglang
DATA=/root/projects/sglang-qwen2-adaptive-prefill/hicache
VENV=/root/autodl-tmp/venvs/sglang015/bin/activate
BASE_URL=http://127.0.0.1:30000

: "${MODEL_PATH:?Please export MODEL_PATH before running}"

# ---------------- Experiment config ----------------
PREFIX_LENGTHS="${PREFIX_LENGTHS:-256,512,1024,4096,16384}"
SESSIONS_PER_LEN="${SESSIONS_PER_LEN:-32}"
TRIALS="${TRIALS:-1}"
NUM_EVICTORS="${NUM_EVICTORS:-22}"
EVICTOR_LEN="${EVICTOR_LEN:-29000}"
OUTPUT_LEN="${OUTPUT_LEN:-128}"
TAIL_LEN="${TAIL_LEN:-64}"
REQUEST_RATE="${REQUEST_RATE:-inf}"

THRESHOLD="${THRESHOLD:-960}"
COST_PROFILE="${HICACHE_COST_PROFILE:-$DATA/profiles/hicache_cost_profile_v2.json}"
MARGIN_MS="${HICACHE_MARGIN_MS:-2.0}"

# Selectable policies/concurrencies.
# Example:
# POLICIES_CSV=always_restore,always_recompute
# CONCURRENCIES_CSV=4,16,32
POLICIES_CSV="${POLICIES_CSV:-always_restore,token_threshold,cost_model}"
CONCURRENCIES_CSV="${CONCURRENCIES_CSV:-16,32}"
IFS=',' read -r -a POLICIES <<< "$POLICIES_CSV"
IFS=',' read -r -a CONCURRENCIES <<< "$CONCURRENCIES_CSV"

# RESULT_TAG is strongly recommended for development runs.
RESULT_TAG="${RESULT_TAG:-}"
ARCHIVE_RESULTS="${ARCHIVE_RESULTS:-1}"
RUN_ANALYSIS="${RUN_ANALYSIS:-1}"

TIMESTAMP=$(date +%Y%m%d_%H%M%S)
RUN_LOG="$DATA/logs/run_agent_all_${TIMESTAMP}.log"
ARCHIVE="$DATA/archive/agent_b_${TIMESTAMP}"
SERVER_PID=""
CURRENT_SERVER_LOG=""

source "$VENV"
cd "$REPO"
mkdir -p "$DATA/results" "$DATA/logs" "$DATA/archive" "$ARCHIVE/results" "$ARCHIVE/logs"

log() { echo "[$(date '+%F %T')] $*" | tee -a "$RUN_LOG"; }

result_suffix() {
    [[ -n "$RESULT_TAG" ]] && echo "_$RESULT_TAG" || true
}

cleanup_server() {
    if [[ -n "${SERVER_PID:-}" ]]; then
        log "Stopping server process group: $SERVER_PID"
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

contains_policy() {
    local wanted="$1"
    for p in "${POLICIES[@]}"; do
        [[ "$p" == "$wanted" ]] && return 0
    done
    return 1
}

check_environment() {
    [[ -f shell_scripts/start_hicache_server.sh ]] || { echo "Missing start_hicache_server.sh"; exit 1; }
    [[ -f shell_scripts/bench_hicache_agent.py ]] || { echo "Missing bench_hicache_agent.py"; exit 1; }
    python -m py_compile shell_scripts/bench_hicache_agent.py

    for policy in "${POLICIES[@]}"; do
        case "$policy" in
            always_restore|always_recompute|token_threshold|cost_model) ;;
            *) echo "Unsupported policy: $policy"; exit 1 ;;
        esac
    done

    if contains_policy cost_model && [[ ! -f "$COST_PROFILE" ]]; then
        echo "Missing cost profile: $COST_PROFILE"
        exit 1
    fi

    if curl -sf "$BASE_URL/health" >/dev/null 2>&1; then
        echo "Port 30000 already has a running SGLang server."
        exit 1
    fi
}

archive_old_files() {
    [[ "$ARCHIVE_RESULTS" == "1" ]] || return 0
    [[ -z "$RESULT_TAG" ]] || return 0

    shopt -s nullglob
    for policy in "${POLICIES[@]}"; do
        for c in "${CONCURRENCIES[@]}"; do
            f="$DATA/results/agent_${policy}_c${c}.jsonl"
            [[ -f "$f" ]] && mv "$f" "$ARCHIVE/results/"
        done
        for f in "$DATA/logs/server_agent_${policy}.log" "$DATA/logs/launcher_agent_${policy}_"*.log "$DATA/logs/bench_agent_${policy}_c"*.log; do
            [[ -e "$f" ]] && mv "$f" "$ARCHIVE/logs/"
        done
    done
    shopt -u nullglob
}

wait_server() {
    local launcher_log="$1"
    log "Waiting for server startup..."
    for i in $(seq 1 600); do
        if curl -sf "$BASE_URL/health" >/dev/null 2>&1; then
            log "Server ready."
            sleep 2
            return 0
        fi
        if ! kill -0 "$SERVER_PID" 2>/dev/null; then
            log "Server exited during startup."
            tail -100 "$launcher_log" || true
            return 1
        fi
        (( i % 30 == 0 )) && log "Still waiting... ${i}s"
        sleep 1
    done
    log "Server startup timeout."
    tail -100 "$launcher_log" || true
    return 1
}

start_server() {
    local policy="$1"
    local suffix=""
    [[ -n "$RESULT_TAG" ]] && suffix="_${RESULT_TAG}"
    local tag="agent_${policy}${suffix}"
    local launcher_log="$DATA/logs/launcher_${tag}_${TIMESTAMP}.log"

    CURRENT_SERVER_LOG="$DATA/logs/server_${tag}.log"
    rm -f "$CURRENT_SERVER_LOG"

    log "============================================================"
    log "Starting server: policy=$policy tag=$tag"
    log "============================================================"

    case "$policy" in
        always_restore)
            setsid bash -c "source '$VENV'; cd '$REPO'; export MODEL_PATH='$MODEL_PATH'; LOG_TAG='$tag' ./shell_scripts/start_hicache_server.sh always_restore" >"$launcher_log" 2>&1 &
            ;;
        always_recompute)
            setsid bash -c "source '$VENV'; cd '$REPO'; export MODEL_PATH='$MODEL_PATH'; LOG_TAG='$tag' ./shell_scripts/start_hicache_server.sh always_recompute" >"$launcher_log" 2>&1 &
            ;;
        token_threshold)
            setsid bash -c "source '$VENV'; cd '$REPO'; export MODEL_PATH='$MODEL_PATH'; LOG_TAG='$tag' ./shell_scripts/start_hicache_server.sh token_threshold '$THRESHOLD'" >"$launcher_log" 2>&1 &
            ;;
        cost_model)
            setsid bash -c "source '$VENV'; cd '$REPO'; export MODEL_PATH='$MODEL_PATH'; export HICACHE_COST_PROFILE='$COST_PROFILE'; export HICACHE_MARGIN_MS='$MARGIN_MS'; LOG_TAG='$tag' ./shell_scripts/start_hicache_server.sh cost_model" >"$launcher_log" 2>&1 &
            ;;
    esac

    SERVER_PID=$!
    log "Server PID=$SERVER_PID"
    wait_server "$launcher_log"
}

check_decisions() {
    local policy="$1" concurrency="$2"
    [[ -f "$CURRENT_SERVER_LOG" ]] || return 0

    local key="agent_target_${policy}_c${concurrency}_"
    local total restore recompute
    total=$(grep "HiCacheDecision" "$CURRENT_SERVER_LOG" | grep -c "$key" || true)
    restore=$(grep "HiCacheDecision" "$CURRENT_SERVER_LOG" | grep "$key" | grep -c "action=restore" || true)
    recompute=$(grep "HiCacheDecision" "$CURRENT_SERVER_LOG" | grep "$key" | grep -c "action=recompute" || true)

    log "Decision check: policy=$policy c=$concurrency total=$total restore=$restore recompute=$recompute"

    if [[ "$policy" == "always_restore" && "$total" -gt 0 && "$restore" -ne "$total" ]]; then
        log "ERROR: always_restore contains non-restore decisions."
        exit 1
    fi
    if [[ "$policy" == "always_recompute" && "$total" -gt 0 && "$recompute" -ne "$total" ]]; then
        log "ERROR: always_recompute contains non-recompute decisions."
        exit 1
    fi
}

run_benchmark() {
    local policy="$1" concurrency="$2"
    local suffix
    suffix=$(result_suffix)
    local result="$DATA/results/agent_${policy}_c${concurrency}${suffix}.jsonl"
    local bench_log="$DATA/logs/bench_agent_${policy}_c${concurrency}${suffix}.log"

    log "------------------------------------------------------------"
    log "Benchmark: policy=$policy c=$concurrency"
    log "prefix=$PREFIX_LENGTHS sessions=$SESSIONS_PER_LEN trials=$TRIALS output=$OUTPUT_LEN"
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

    check_decisions "$policy" "$concurrency"
}

run_policy() {
    local policy="$1"
    start_server "$policy"
    for c in "${CONCURRENCIES[@]}"; do
        run_benchmark "$policy" "$c"
    done
    cleanup_server
}

print_quick_summary() {
    local suffix
    suffix=$(result_suffix)

    echo
    echo "================================================================================================================"
    echo "QUICK RESULT SUMMARY"
    echo "================================================================================================================"

    POLICIES_CSV="$POLICIES_CSV" CONCURRENCIES_CSV="$CONCURRENCIES_CSV" RESULT_SUFFIX="$suffix" DATA="$DATA" python - <<'PY'
import json, math, os
from pathlib import Path

base=Path(os.environ["DATA"])/"results"
policies=os.environ["POLICIES_CSV"].split(",")
concurrencies=[int(x) for x in os.environ["CONCURRENCIES_CSV"].split(",")]
suffix=os.environ.get("RESULT_SUFFIX","")

def pct(v,q):
    if not v: return float("nan")
    v=sorted(v)
    i=min(len(v)-1,max(0,math.ceil(q*len(v))-1))
    return v[i]

print(f"{'Policy':20s} {'C':>4s} {'Req':>6s} {'P50':>10s} {'P95':>10s} {'MAX':>10s} {'LoadBack':>12s} {'Evicted':>12s}")
print("-"*100)

for policy in policies:
    for c in concurrencies:
        path=base/f"agent_{policy}_c{c}{suffix}.jsonl"
        if not path.exists():
            continue

        ttft=[]
        load_back=0
        evicted=0
        summaries=0

        with path.open() as f:
            for line in f:
                x=json.loads(line)
                if x.get("record_type")=="request":
                    ttft.append(float(x["ttft_ms"]))
                elif x.get("record_type")=="summary":
                    load_back += float(x.get("load_back_delta",0))
                    evicted += float(x.get("measurement_evicted_delta",0))
                    summaries += 1

        print(
            f"{policy:20s} {c:4d} {len(ttft):6d} "
            f"{pct(ttft,.50):10.3f} {pct(ttft,.95):10.3f} {max(ttft) if ttft else float('nan'):10.3f} "
            f"{load_back:12.0f} {evicted:12.0f}"
        )
PY
}

run_legacy_analysis() {
    [[ "$RUN_ANALYSIS" == "1" ]] || return 0

    # Existing analyze_hicache_agent.py is for the original
    # always_restore/token_threshold/cost_model formal comparison.
    if [[ -n "$RESULT_TAG" ]]; then
        log "Skip legacy analysis because RESULT_TAG=$RESULT_TAG."
        return 0
    fi
    if ! contains_policy always_restore || ! contains_policy token_threshold || ! contains_policy cost_model; then
        log "Skip legacy analysis: required formal policies are not all selected."
        return 0
    fi
    if [[ ! -f shell_scripts/analyze_hicache_agent.py ]]; then
        log "Skip legacy analysis: analyzer not found."
        return 0
    fi

    log "Running analyze_hicache_agent.py..."
    python shell_scripts/analyze_hicache_agent.py 2>&1 | tee "$DATA/results/agent_analysis_${TIMESTAMP}.txt"
}

main() {
    log "============================================================"
    log "SGLang HiCache Agent Benchmark"
    log "============================================================"
    log "MODEL_PATH=$MODEL_PATH"
    log "POLICIES=$POLICIES_CSV"
    log "CONCURRENCIES=$CONCURRENCIES_CSV"
    log "PREFIX_LENGTHS=$PREFIX_LENGTHS"
    log "SESSIONS_PER_LEN=$SESSIONS_PER_LEN TRIALS=$TRIALS"
    log "EVICTORS=${NUM_EVICTORS}x${EVICTOR_LEN}"
    log "OUTPUT_LEN=$OUTPUT_LEN"
    log "THRESHOLD=$THRESHOLD MARGIN_MS=$MARGIN_MS"
    log "RESULT_TAG=${RESULT_TAG:-<none>}"

    check_environment
    archive_old_files

    for policy in "${POLICIES[@]}"; do
        run_policy "$policy"
    done

    print_quick_summary
    run_legacy_analysis

    log "============================================================"
    log "ALL SELECTED EXPERIMENTS COMPLETED"
    log "============================================================"
}

main "$@"

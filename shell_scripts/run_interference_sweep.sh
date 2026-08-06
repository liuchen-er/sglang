#!/usr/bin/env bash

set -euo pipefail


PROJECT_ROOT="${PROJECT_ROOT:-/root/projects/sglang-qwen2-adaptive-prefill}"
RESULTS_ROOT="${RESULTS_ROOT:-${PROJECT_ROOT}/results/interference}"

RUNS="${RUNS:-3}"
CHUNKS="${CHUNKS:-512 1024 2048 4096}"
PROFILES="${PROFILES:-standard}"

# fixed：执行原有 Baseline
# adaptive：只执行 Adaptive Mixed
MODE="${MODE:-fixed}"


source /root/autodl-tmp/venvs/sglang015/bin/activate
source "${PROJECT_ROOT}/scripts/env.sh"


SERVER_PID=""


stop_server() {
    if [[ -z "${SERVER_PID}" ]]; then
        return
    fi

    if ! kill -0 "${SERVER_PID}" 2>/dev/null; then
        wait "${SERVER_PID}" 2>/dev/null || true
        SERVER_PID=""
        return
    fi

    echo "Stopping SGLang server: ${SERVER_PID}"

    # 先通知主进程正常退出。
    kill -TERM "${SERVER_PID}" 2>/dev/null || true

    for _ in $(seq 1 30); do
        if ! kill -0 "${SERVER_PID}" 2>/dev/null; then
            break
        fi

        sleep 1
    done

    # 超时后清理整个进程组。
    if kill -0 "${SERVER_PID}" 2>/dev/null; then
        echo "Server did not exit gracefully, killing process group."

        kill -KILL -- "-${SERVER_PID}" 2>/dev/null || true
    fi

    wait "${SERVER_PID}" 2>/dev/null || true

    SERVER_PID=""

    # 给端口和GPU资源留出释放时间。
    sleep 3
}


cleanup() {
    stop_server || true
}


trap cleanup EXIT INT TERM


start_fixed_server() {
    local chunk_size="$1"
    local server_log="$2"

    mkdir -p "$(dirname "${server_log}")"

    echo
    echo "Starting fixed SGLang server"
    echo "CHUNK_SIZE=${chunk_size}"
    echo "Server log: ${server_log}"

    setsid env \
        ADAPTIVE=0 \
        CHUNK_SIZE="${chunk_size}" \
        bash "${PROJECT_ROOT}/scripts/launch_server.sh" \
        >"${server_log}" 2>&1 &

    SERVER_PID=$!

    echo "Server PID: ${SERVER_PID}"

    bash "${PROJECT_ROOT}/scripts/wait_server.sh"

    echo "Fixed SGLang server is ready."
}


start_adaptive_server() {
    local server_log="$1"

    mkdir -p "$(dirname "${server_log}")"

    echo
    echo "Starting adaptive SGLang server"
    echo "ADAPTIVE=1"
    echo "Server log: ${server_log}"

    setsid env \
        ADAPTIVE=1 \
        bash "${PROJECT_ROOT}/scripts/launch_server.sh" \
        >"${server_log}" 2>&1 &

    SERVER_PID=$!

    echo "Server PID: ${SERVER_PID}"

    bash "${PROJECT_ROOT}/scripts/wait_server.sh"

    echo "Adaptive SGLang server is ready."
}


warmup_server() {
    local warmup_log="$1"

    mkdir -p "$(dirname "${warmup_log}")"

    echo "Running warmup..."

    set +e

    python -m sglang.benchmark.serving \
        --backend sglang \
        --host "${HOST}" \
        --port "${PORT}" \
        --dataset-name random-ids \
        --tokenize-prompt \
        --random-input-len 256 \
        --random-output-len 64 \
        --random-range-ratio 1.0 \
        --num-prompts 8 \
        --max-concurrency 8 \
        --request-rate inf \
        --seed 9999 \
        >"${warmup_log}" 2>&1

    local status=$?

    set -e

    if [[ "${status}" -ne 0 ]]; then
        echo "ERROR: Warmup failed, status=${status}" >&2
        echo "Warmup log: ${warmup_log}" >&2

        tail -n 100 "${warmup_log}" >&2 || true

        return "${status}"
    fi

    echo "Warmup finished."
}


run_fixed_once() {
    local profile="$1"
    local chunk_size="$2"
    local run_id="$3"

    # 保持原有 Baseline 目录结构。
    local decode_dir
    local mixed_dir

    decode_dir="${RESULTS_ROOT}/${profile}/decode_only/chunk_${chunk_size}/run_${run_id}"
    mixed_dir="${RESULTS_ROOT}/${profile}/mixed/chunk_${chunk_size}/run_${run_id}"

    echo
    echo "=================================================="
    echo "Fixed Decode-only"
    echo "profile=${profile}"
    echo "chunk_size=${chunk_size}"
    echo "run_id=${run_id}"
    echo "=================================================="

    mkdir -p "${decode_dir}"

    start_fixed_server \
        "${chunk_size}" \
        "${decode_dir}/server.log"

    warmup_server \
        "${decode_dir}/warmup.log"

    bash "${PROJECT_ROOT}/scripts/bench_decode_fixed_rate.sh" \
        "${decode_dir}" \
        "${profile}"

    stop_server

    echo
    echo "=================================================="
    echo "Fixed Mixed"
    echo "profile=${profile}"
    echo "chunk_size=${chunk_size}"
    echo "run_id=${run_id}"
    echo "=================================================="

    mkdir -p "${mixed_dir}"

    start_fixed_server \
        "${chunk_size}" \
        "${mixed_dir}/server.log"

    warmup_server \
        "${mixed_dir}/warmup.log"

    bash "${PROJECT_ROOT}/scripts/bench_mixed.sh" \
        "${mixed_dir}" \
        "${profile}"

    stop_server
}


run_adaptive_once() {
    local profile="$1"
    local run_id="$2"

    # Adaptive只新增到Mixed下，不改动任何Baseline目录。
    local mixed_dir

    mixed_dir="${RESULTS_ROOT}/${profile}/mixed/chunk_adaptive/run_${run_id}"

    echo
    echo "=================================================="
    echo "Adaptive Mixed"
    echo "profile=${profile}"
    echo "chunk_policy=adaptive"
    echo "run_id=${run_id}"
    echo "=================================================="

    mkdir -p "${mixed_dir}"

    start_adaptive_server \
        "${mixed_dir}/server.log"

    warmup_server \
        "${mixed_dir}/warmup.log"

    bash "${PROJECT_ROOT}/scripts/bench_mixed.sh" \
        "${mixed_dir}" \
        "${profile}"

    # 单独记录策略信息，不修改bench_mixed.sh的原有配置。
    cat >"${mixed_dir}/strategy.txt" <<EOF
MODE=adaptive
ADAPTIVE=1
PROFILE=${profile}
RUN_ID=${run_id}
EOF

    stop_server
}


run_fixed_sweep() {
    local profile
    local chunk_size
    local run_id

    for profile in ${PROFILES}; do
        for chunk_size in ${CHUNKS}; do
            for run_id in $(seq 1 "${RUNS}"); do
                run_fixed_once \
                    "${profile}" \
                    "${chunk_size}" \
                    "${run_id}"
            done
        done
    done
}


run_adaptive_sweep() {
    local profile
    local run_id

    for profile in ${PROFILES}; do
        for run_id in $(seq 1 "${RUNS}"); do
            run_adaptive_once \
                "${profile}" \
                "${run_id}"
        done
    done
}


echo "=================================================="
echo "SGLang Interference Experiment"
echo "MODE=${MODE}"
echo "RUNS=${RUNS}"
echo "PROFILES=${PROFILES}"
echo "CHUNKS=${CHUNKS}"
echo "RESULTS_ROOT=${RESULTS_ROOT}"
echo "=================================================="


case "${MODE}" in
    fixed)
        run_fixed_sweep
        ;;

    adaptive)
        run_adaptive_sweep
        ;;

    *)
        echo "ERROR: Unsupported MODE=${MODE}" >&2
        echo "Supported modes: fixed adaptive" >&2
        exit 1
        ;;
esac


echo
echo "=================================================="
echo "Experiments completed."
echo "MODE=${MODE}"
echo "=================================================="
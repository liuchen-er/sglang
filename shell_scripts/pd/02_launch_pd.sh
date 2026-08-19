#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "$0")" && pwd)
source "$SCRIPT_DIR/00_common.sh"

PHASE=${1:-manual}
MODE=pd
LOG_DIR="$SERVER_LOG_ROOT/$PHASE/$MODE"
RUN_DIR="$RUN_ROOT/$MODE"

mkdir -p "$LOG_DIR" "$RUN_DIR"

python -c "import nixl" >/dev/null 2>&1 || {
    echo "ERROR: nixl is not installed"
    echo "Run: pip install nixl"
    exit 1
}

echo "Launching Prefill worker on GPU0..."

nohup env CUDA_VISIBLE_DEVICES=0 \
    sglang serve \
    "${COMMON_SERVER_ARGS[@]}" \
    --port "$WORKER0_PORT" \
    --disaggregation-mode prefill \
    --disaggregation-transfer-backend nixl \
    > "$LOG_DIR/prefill.log" 2>&1 &

echo $! > "$RUN_DIR/prefill.pid"

echo "Launching Decode worker on GPU1..."

nohup env CUDA_VISIBLE_DEVICES=1 \
    sglang serve \
    "${COMMON_SERVER_ARGS[@]}" \
    --port "$WORKER1_PORT" \
    --disaggregation-mode decode \
    --disaggregation-transfer-backend nixl \
    > "$LOG_DIR/decode.log" 2>&1 &

echo $! > "$RUN_DIR/decode.pid"

wait_worker() {
    local port=$1
    local name=$2

    echo "Waiting for $name..."

    until curl -sf "http://127.0.0.1:${port}/model_info" >/dev/null; do
        sleep 1
    done

    echo "$name ready."
}

wait_worker "$WORKER0_PORT" "Prefill worker"
wait_worker "$WORKER1_PORT" "Decode worker"

echo "Launching PD router..."

nohup "$ROUTER_BIN" launch \
    --pd-disaggregation \
    --prefill "http://127.0.0.1:${WORKER0_PORT}" \
    --decode "http://127.0.0.1:${WORKER1_PORT}" \
    --host "$HOST" \
    --port "$ROUTER_PORT" \
    > "$LOG_DIR/router.log" 2>&1 &

echo $! > "$RUN_DIR/router.pid"

echo "Waiting for router..."

until curl -sf "http://${HOST}:${ROUTER_PORT}/v1/models" >/dev/null; do
    sleep 0.5
done

echo "Router ready."
echo
echo "========================================="
echo "PD deployment ready"
echo "GPU0 : Prefill"
echo "GPU1 : Decode"
echo "Router: http://${HOST}:${ROUTER_PORT}"
echo "Logs  : $LOG_DIR"
echo "========================================="

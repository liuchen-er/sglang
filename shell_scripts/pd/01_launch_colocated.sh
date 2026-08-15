#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "$0")" && pwd)
source "$SCRIPT_DIR/00_common.sh"

MODE=colocated
LOG_DIR="$LOG_ROOT/$MODE"
RUN_DIR="$RUN_ROOT/$MODE"

mkdir -p "$LOG_DIR" "$RUN_DIR"

echo "Launching GPU0 colocated worker..."

nohup env CUDA_VISIBLE_DEVICES=0 \
    sglang serve \
    "${COMMON_SERVER_ARGS[@]}" \
    --port "$WORKER0_PORT" \
    > "$LOG_DIR/worker0.log" 2>&1 &

echo $! > "$RUN_DIR/worker0.pid"

echo "Launching GPU1 colocated worker..."

nohup env CUDA_VISIBLE_DEVICES=1 \
    sglang serve \
    "${COMMON_SERVER_ARGS[@]}" \
    --port "$WORKER1_PORT" \
    > "$LOG_DIR/worker1.log" 2>&1 &

echo $! > "$RUN_DIR/worker1.pid"

wait_worker() {
    local port=$1
    local name=$2

    echo "Waiting for $name..."

    until curl -sf "http://127.0.0.1:${port}/model_info" >/dev/null; do
        sleep 2
    done

    echo "$name ready."
}

wait_worker "$WORKER0_PORT" "worker0"
wait_worker "$WORKER1_PORT" "worker1"

echo "Launching router..."

nohup "$ROUTER_BIN" launch \
    --worker-urls \
        "http://127.0.0.1:${WORKER0_PORT}" \
        "http://127.0.0.1:${WORKER1_PORT}" \
    --policy round_robin \
    --host "$HOST" \
    --port "$ROUTER_PORT" \
    > "$LOG_DIR/router.log" 2>&1 &

echo $! > "$RUN_DIR/router.pid"

sleep 3

echo
echo "========================================="
echo "Colocated deployment ready"
echo "GPU0 : P + D"
echo "GPU1 : P + D"
echo "Router: http://${HOST}:${ROUTER_PORT}"
echo "Logs  : $LOG_DIR"
echo "========================================="

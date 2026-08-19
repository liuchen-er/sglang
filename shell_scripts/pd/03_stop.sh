#!/usr/bin/env bash

EXP_ROOT=/root/autodl-tmp/sglang_pd_exp
RUN_ROOT="$EXP_ROOT/run"

stop_pid_file() {
    local file=$1

    [ -f "$file" ] || return

    local pid
    pid=$(cat "$file")

    if kill -0 "$pid" 2>/dev/null; then
        echo "Stopping PID $pid ($(basename "$file"))..."
        kill "$pid" 2>/dev/null || true

        # 最多等待 5 秒
        for _ in {1..10}; do
            if ! kill -0 "$pid" 2>/dev/null; then
                break
            fi
            sleep 0.5
        done

        # 仍未退出则提示
        if kill -0 "$pid" 2>/dev/null; then
            echo "WARNING: PID $pid is still alive"
        fi
    fi

    rm -f "$file"
}

find "$RUN_ROOT" -name "*.pid" -type f 2>/dev/null | while read -r file; do
    stop_pid_file "$file"
done

echo
echo "Remaining SGLang processes:"
ps -ef | grep -E "sglang|sglang-router" | grep -v grep || true

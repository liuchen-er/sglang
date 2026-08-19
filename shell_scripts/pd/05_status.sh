#!/usr/bin/env bash

echo "===== GPU ====="
nvidia-smi --query-compute-apps=gpu_uuid,pid,used_memory --format=csv || true

echo
echo "===== Processes ====="
ps -ef | grep -E "sglang serve|sglang-router|smg launch" | grep -v grep || true

echo
echo "===== Ports ====="
ss -lntp | grep -E ":30000|:30001|:8000" || true

echo
echo "===== Disk ====="
df -h /root /root/autodl-tmp

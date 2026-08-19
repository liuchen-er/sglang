#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "$0")" && pwd)
source "$SCRIPT_DIR/00_common.sh"

TS=$(date +%Y%m%d_%H%M%S)
OUT="$META_ROOT/meta_${TS}.txt"

{
    echo "===== TIME ====="
    date

    echo
    echo "===== GIT ====="
    cd "$REPO"
    git branch --show-current
    git rev-parse HEAD
    git status --short

    echo
    echo "===== SGLANG ====="
    sglang version || true

    echo
    echo "===== PYTHON ====="
    which python
    python --version

    echo
    echo "===== PYTORCH / CUDA ====="
    python - <<'PY'
import torch
print("torch:", torch.__version__)
print("cuda:", torch.version.cuda)
print("gpu_count:", torch.cuda.device_count())
for i in range(torch.cuda.device_count()):
    print(i, torch.cuda.get_device_name(i))
PY

    echo
    echo "===== NVIDIA SMI ====="
    nvidia-smi

    echo
    echo "===== TOPOLOGY ====="
    nvidia-smi topo -m

    echo
    echo "===== DISK ====="
    df -h /root /root/autodl-tmp

} | tee "$OUT"

echo
echo "Saved to: $OUT"

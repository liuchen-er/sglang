#!/usr/bin/env bash
set -euo pipefail

source /root/autodl-tmp/venvs/sglang015/bin/activate
source "$(dirname "$0")/env.sh"

CHUNK_SIZE="${CHUNK_SIZE:-4096}"
MEM_FRACTION="${MEM_FRACTION:-0.75}"
ADAPTIVE="${ADAPTIVE:-0}"

ARGS=(
  --model-path "${MODEL_PATH}"
  --host 0.0.0.0
  --port "${PORT}"
  --context-length 32768
  --mem-fraction-static "${MEM_FRACTION}"
  --chunked-prefill-size "${CHUNK_SIZE}"
  --max-prefill-tokens 16384
  --enable-mixed-chunk
  --disable-radix-cache
  --stream-interval 1
)

# 完成源码修改后，ADAPTIVE=1 才会使用这些参数。
if [[ "${ADAPTIVE}" == "1" ]]; then
  ARGS+=(
    --enable-load-aware-chunking
    --adaptive-chunk-min 512
    --adaptive-chunk-small 1024
    --adaptive-chunk-medium 2048
    --adaptive-chunk-max "${CHUNK_SIZE}"
    --adaptive-decode-low-watermark 4
    --adaptive-decode-high-watermark 12
    --adaptive-prefill-queue-high-watermark 8
  )
fi

exec python -m sglang.launch_server "${ARGS[@]}" "$@"
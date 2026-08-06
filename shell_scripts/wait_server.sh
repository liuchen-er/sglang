#!/usr/bin/env bash
set -euo pipefail

source "$(dirname "$0")/env.sh"

for _ in $(seq 1 180); do
  if curl -fsS "${BASE_URL}/health" >/dev/null 2>&1; then
    echo "SGLang server is ready."
    exit 0
  fi

  sleep 2
done

echo "SGLang server was not ready within 360 seconds." >&2
exit 1
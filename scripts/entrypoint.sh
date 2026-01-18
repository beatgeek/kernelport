#!/usr/bin/env sh
set -eu

if [ -n "${MODEL_MANIFEST_PATH:-}" ]; then
  python3 /app/scripts/model_fetch.py "$MODEL_MANIFEST_PATH"
fi

exec /app/kernelportd serve "$@"

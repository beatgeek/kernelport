#!/usr/bin/env sh
set -eu

SERVE_ARGS=""
if [ -n "${MODEL_MANIFEST_PATH:-}" ]; then
  python3 /app/scripts/model_fetch.py "$MODEL_MANIFEST_PATH"
  SERVE_ARGS=$(python3 /app/scripts/manifest_to_serve_args.py "$MODEL_MANIFEST_PATH")
fi

exec /app/kernelportd serve $SERVE_ARGS "$@"

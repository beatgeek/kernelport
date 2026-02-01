#!/usr/bin/env python3
"""
Read a model manifest and print kernelportd serve args (--backend, --model-path or
--helion-model, etc.). Used by entrypoint.sh when MODEL_MANIFEST_PATH is set.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

import yaml


def load_manifest(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def main() -> int:
    if len(sys.argv) != 2:
        print("usage: manifest_to_serve_args.py <manifest.yaml>", file=sys.stderr)
        return 2

    manifest_path = Path(sys.argv[1]).resolve()
    manifest = load_manifest(manifest_path)

    backend = manifest.get("backend")
    if not backend:
        backend = "onnx" if manifest.get("format") == "onnx" else "helion"
    model_id = manifest.get("id", "model")
    cache_dir = manifest.get("cache", {}).get("dir", "/models")

    args = ["--backend", backend]

    if backend == "onnx":
        files = manifest.get("files", ["model.onnx"])
        model_file = files[0] if files else "model.onnx"
        model_path = Path(cache_dir) / model_id / model_file
        args.extend(["--model-path", str(model_path)])
    elif backend == "helion":
        helion_model = manifest.get("helion_model", model_id)
        helion_addr = os.environ.get("HELION_ADDR", "http://127.0.0.1:50061")
        args.extend(["--helion-addr", helion_addr, "--helion-model", helion_model])
    else:
        print(f"unknown backend: {backend}", file=sys.stderr)
        return 1

    device = manifest.get("runtime", {}).get("device", "cpu")
    args.extend(["--device", device])

    print(" ".join(args))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

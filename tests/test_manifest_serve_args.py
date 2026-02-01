#!/usr/bin/env python3
"""
Test manifest_to_serve_args.py output for sample and LuxTTS manifests.
Run from repo root: python3 tests/test_manifest_serve_args.py
Requires PyYAML: pip3 install pyyaml or uv pip install pyyaml
No pytest required.
"""
from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

# manifest_to_serve_args.py imports yaml; fail fast with a clear message
try:
    import yaml  # noqa: F401
except ModuleNotFoundError:
    print(
        "PyYAML is required. Run: pip3 install pyyaml  or  uv pip install pyyaml",
        file=sys.stderr,
    )
    sys.exit(1)


def repo_root() -> Path:
    root = Path(__file__).resolve().parent.parent
    if not (root / "scripts" / "manifest_to_serve_args.py").exists():
        raise RuntimeError(f"repo root not found (expected {root})")
    return root


def run_manifest_to_serve_args(manifest_path: Path) -> str:
    script = repo_root() / "scripts" / "manifest_to_serve_args.py"
    result = subprocess.run(
        [sys.executable, str(script), str(manifest_path)],
        capture_output=True,
        text=True,
        cwd=str(repo_root()),
    )
    if result.returncode != 0:
        raise AssertionError(
            f"manifest_to_serve_args failed: {result.stderr or result.stdout}"
        )
    return result.stdout.strip()


def test_sample_manifest_onnx() -> None:
    root = repo_root()
    manifest = root / "models" / "sample-manifest.yaml"
    if not manifest.exists():
        raise FileNotFoundError(f"manifest not found: {manifest}")
    out = run_manifest_to_serve_args(manifest)
    assert "--backend onnx" in out, f"expected --backend onnx in {out!r}"
    assert "--model-path" in out, f"expected --model-path in {out!r}"
    assert "/models/" in out, f"expected /models/ in {out!r}"


def test_luxtts_manifest_helion() -> None:
    root = repo_root()
    manifest = root / "models" / "luxtts-manifest.yaml"
    if not manifest.exists():
        raise FileNotFoundError(f"manifest not found: {manifest}")
    out = run_manifest_to_serve_args(manifest)
    assert "--backend helion" in out, f"expected --backend helion in {out!r}"
    assert "--helion-model luxtts" in out, f"expected --helion-model luxtts in {out!r}"
    assert "--helion-addr" in out, f"expected --helion-addr in {out!r}"


def main() -> int:
    root = repo_root()
    os.chdir(root)
    try:
        test_sample_manifest_onnx()
        test_luxtts_manifest_helion()
        print("manifest_to_serve_args tests passed")
        return 0
    except Exception as e:
        print(f"FAIL: {e}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())

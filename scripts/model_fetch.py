#!/usr/bin/env python3
import os
import sys
from pathlib import Path
from typing import Optional

import yaml
from huggingface_hub import snapshot_download


def load_manifest(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def resolve_token() -> Optional[str]:
    return os.environ.get("HF_TOKEN") or os.environ.get("HUGGINGFACE_HUB_TOKEN")


def main() -> int:
    if len(sys.argv) != 2:
        print("usage: model_fetch.py <manifest.yaml>", file=sys.stderr)
        return 2

    manifest_path = Path(sys.argv[1]).resolve()
    manifest = load_manifest(manifest_path)

    source = manifest.get("source", {})
    if source.get("kind") != "huggingface":
        raise RuntimeError("manifest source.kind must be 'huggingface'")

    repo = source.get("repo")
    revision = source.get("revision", "main")
    if not repo:
        raise RuntimeError("manifest source.repo is required")

    cache_dir = Path(manifest.get("cache", {}).get("dir", "/models"))
    model_id = manifest.get("id", repo.replace("/", "__"))
    local_dir = cache_dir / model_id
    local_dir.mkdir(parents=True, exist_ok=True)

    files = manifest.get("files")
    allow_patterns = files if isinstance(files, list) else None

    snapshot_download(
        repo_id=repo,
        revision=revision,
        allow_patterns=allow_patterns,
        local_dir=str(local_dir),
        local_dir_use_symlinks=False,
        token=resolve_token(),
    )

    if allow_patterns:
        missing = [name for name in allow_patterns if not (local_dir / name).exists()]
        if missing:
            raise RuntimeError(f"missing expected files: {missing}")

    print(f"model ready at {local_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

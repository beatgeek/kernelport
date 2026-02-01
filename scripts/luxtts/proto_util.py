"""Shared proto compilation for workers that use kernelport inference proto."""
from __future__ import annotations

import os
import sys
import tempfile
from typing import Tuple

from grpc_tools import protoc


def find_repo_root() -> str:
    env_root = os.environ.get("KERNELPORT_REPO_ROOT")
    if env_root:
        return env_root
    here = os.path.abspath(os.path.dirname(__file__))
    for _ in range(5):
        candidate = os.path.join(here, "crates", "kernelport-proto", "src", "inference.proto")
        if os.path.exists(candidate):
            return here
        here = os.path.abspath(os.path.join(here, ".."))
    return os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))


def compile_proto(repo_root: str) -> Tuple[object, object]:
    proto_dir = os.path.join(repo_root, "crates", "kernelport-proto", "src")
    proto_path = os.path.join(proto_dir, "inference.proto")
    tmpdir = tempfile.mkdtemp(prefix="kernelport_proto_")

    args = [
        "grpc_tools.protoc",
        f"-I{proto_dir}",
        f"--python_out={tmpdir}",
        f"--grpc_python_out={tmpdir}",
        proto_path,
    ]
    if protoc.main(args) != 0:
        raise RuntimeError("failed to generate grpc stubs")

    pkg_dir = os.path.join(tmpdir, "kernelport", "v1")
    os.makedirs(pkg_dir, exist_ok=True)
    for path in (
        os.path.join(tmpdir, "kernelport", "__init__.py"),
        os.path.join(tmpdir, "kernelport", "v1", "__init__.py"),
    ):
        if not os.path.exists(path):
            with open(path, "w", encoding="utf-8") as handle:
                handle.write("")

    sys.path.insert(0, tmpdir)
    try:
        from kernelport.v1 import inference_pb2, inference_pb2_grpc
    except ImportError:
        import inference_pb2  # type: ignore
        import inference_pb2_grpc  # type: ignore

    return inference_pb2, inference_pb2_grpc

from __future__ import annotations

import argparse
import os
import sys
import tempfile
from concurrent import futures
from typing import Tuple

import grpc
import numpy as np
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
    return os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..", ".."))


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


def dtype_from_pb(dtype: int, pb) -> np.dtype:
    mapping = {
        pb.F32: np.float32,
        pb.F16: np.float16,
        pb.I64: np.int64,
        pb.I32: np.int32,
        pb.U8: np.uint8,
    }
    if dtype not in mapping:
        raise ValueError(f"unsupported dtype: {dtype}")
    return mapping[dtype]


def pb_from_dtype(dtype: np.dtype, pb) -> int:
    mapping = {
        np.dtype(np.float32): pb.F32,
        np.dtype(np.float16): pb.F16,
        np.dtype(np.int64): pb.I64,
        np.dtype(np.int32): pb.I32,
        np.dtype(np.uint8): pb.U8,
    }
    if np.dtype(dtype) not in mapping:
        raise ValueError(f"unsupported dtype: {dtype}")
    return mapping[np.dtype(dtype)]


def tensor_from_pb(tensor, pb) -> np.ndarray:
    dtype = dtype_from_pb(tensor.dtype, pb)
    shape = list(tensor.shape)
    data = np.frombuffer(tensor.data, dtype=dtype)
    return data.reshape(shape)


def tensor_to_pb(name: str, array: np.ndarray, pb) -> object:
    array = np.ascontiguousarray(array)
    return pb.Tensor(
        name=name,
        dtype=pb_from_dtype(array.dtype, pb),
        shape=list(array.shape),
        data=array.tobytes(),
    )


def softmax(x: np.ndarray, axis: int = 1) -> np.ndarray:
    x_max = np.max(x, axis=axis, keepdims=True)
    exp = np.exp(x - x_max)
    return exp / np.sum(exp, axis=axis, keepdims=True)


class MockHelionService:
    def __init__(self, pb) -> None:
        self.pb = pb

    def Infer(self, request, context):
        if not request.inputs:
            context.abort(grpc.StatusCode.INVALID_ARGUMENT, "no inputs provided")
        if request.model not in ("softmax_two_pass", "softmax_mock"):
            context.abort(grpc.StatusCode.NOT_FOUND, f"unknown model: {request.model}")

        try:
            x = tensor_from_pb(request.inputs[0], self.pb)
            y = softmax(x, axis=1)
            return self.pb.InferResponse(outputs=[tensor_to_pb("y", y, self.pb)])
        except Exception as exc:
            context.abort(grpc.StatusCode.INTERNAL, f"mock inference failed: {exc}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Mock Helion gRPC worker (CPU)")
    parser.add_argument("--addr", default="0.0.0.0:50061")
    args = parser.parse_args()

    repo_root = find_repo_root()
    pb, pb_grpc = compile_proto(repo_root)

    server = grpc.server(thread_pool=futures.ThreadPoolExecutor(max_workers=4))
    pb_grpc.add_InferenceServiceServicer_to_server(MockHelionService(pb), server)

    server.add_insecure_port(args.addr)
    server.start()
    print(f"Mock Helion worker listening on {args.addr}")
    server.wait_for_termination()


if __name__ == "__main__":
    main()

from __future__ import annotations

import argparse
import os
import sys
import tempfile
from concurrent import futures
from typing import Dict, Tuple

import helion
import helion.language as hl
import torch
from grpc_tools import protoc
import grpc


def find_repo_root() -> str:
    env_root = os.environ.get("KERNELPORT_REPO_ROOT")
    if env_root:
        return env_root
    here = os.path.abspath(os.path.dirname(__file__))
    for _ in range(4):
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


@helion.kernel(autotune_effort="quick")
def softmax_two_pass(x: torch.Tensor) -> torch.Tensor:
    m, n = x.size()
    out = torch.empty_like(x)
    block_size_m = hl.register_block_size(m)
    block_size_n = hl.register_block_size(n)
    for tile_m in hl.tile(m, block_size=block_size_m):
        mi = hl.full([tile_m], float("-inf"), dtype=torch.float32)
        di = hl.zeros([tile_m], dtype=torch.float32)
        for tile_n in hl.tile(n, block_size=block_size_n):
            values = x[tile_m, tile_n]
            local_amax = torch.amax(values, dim=1)
            mi_next = torch.maximum(mi, local_amax)
            di = di * torch.exp(mi - mi_next) + torch.exp(values - mi_next[:, None]).sum(dim=1)
            mi = mi_next
        for tile_n in hl.tile(n, block_size=block_size_n):
            values = x[tile_m, tile_n]
            out[tile_m, tile_n] = torch.exp(values - mi[:, None]) / di[:, None]
    return out


def dtype_from_pb(dtype: int, pb) -> torch.dtype:
    mapping = {
        pb.F32: torch.float32,
        pb.F16: torch.float16,
        pb.I64: torch.int64,
        pb.I32: torch.int32,
        pb.U8: torch.uint8,
    }
    if dtype not in mapping:
        raise ValueError(f"unsupported dtype: {dtype}")
    return mapping[dtype]


def pb_from_dtype(dtype: torch.dtype, pb) -> int:
    mapping = {
        torch.float32: pb.F32,
        torch.float16: pb.F16,
        torch.int64: pb.I64,
        torch.int32: pb.I32,
        torch.uint8: pb.U8,
    }
    if dtype not in mapping:
        raise ValueError(f"unsupported dtype: {dtype}")
    return mapping[dtype]


def tensor_from_pb(tensor, pb) -> torch.Tensor:
    dtype = dtype_from_pb(tensor.dtype, pb)
    shape = list(tensor.shape)
    data = torch.frombuffer(memoryview(tensor.data), dtype=dtype)
    return data.reshape(shape)


def tensor_to_pb(name: str, tensor: torch.Tensor, pb) -> object:
    tensor = tensor.detach().contiguous().cpu()
    data = tensor.numpy().tobytes()
    return pb.Tensor(
        name=name,
        dtype=pb_from_dtype(tensor.dtype, pb),
        shape=list(tensor.shape),
        data=data,
    )


class HelionService:
    def __init__(self, kernels: Dict[str, object], device: str, pb) -> None:
        self.kernels = kernels
        self.device = device
        self.pb = pb

    def Infer(self, request, context):
        if not request.inputs:
            context.abort(grpc.StatusCode.INVALID_ARGUMENT, "no inputs provided")
        kernel = self.kernels.get(request.model)
        if kernel is None:
            context.abort(grpc.StatusCode.NOT_FOUND, f"unknown model: {request.model}")

        try:
            x = tensor_from_pb(request.inputs[0], self.pb).to(self.device)
            y = kernel(x)
            return self.pb.InferResponse(outputs=[tensor_to_pb("y", y, self.pb)])
        except Exception as exc:
            context.abort(grpc.StatusCode.INTERNAL, f"helion inference failed: {exc}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Helion gRPC worker")
    parser.add_argument("--addr", default="0.0.0.0:50061")
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    repo_root = find_repo_root()
    pb, pb_grpc = compile_proto(repo_root)

    kernels = {"softmax_two_pass": softmax_two_pass}

    server = grpc.server(thread_pool=futures.ThreadPoolExecutor(max_workers=4))
    pb_grpc.add_InferenceServiceServicer_to_server(
        HelionService(kernels, args.device, pb), server
    )

    server.add_insecure_port(args.addr)
    server.start()
    print(f"Helion worker listening on {args.addr}")
    server.wait_for_termination()


if __name__ == "__main__":
    main()

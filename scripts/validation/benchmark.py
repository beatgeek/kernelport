"""Correctness-gated gRPC benchmark. Run next to the server to exclude WAN latency."""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import time

import grpc
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts/luxtts"))
from proto_util import compile_proto


def fixture(pb, mode, index, shape, prompt=None):
    if mode == "luxtts":
        text = f"KernelPort validation sentence number {index}.".encode()
        return [pb.Tensor(name="text", dtype=pb.U8, shape=[len(text)], data=text),
                pb.Tensor(name="prompt_audio", dtype=pb.U8, shape=[len(prompt)], data=prompt)], None
    dtype = np.dtype("<f2" if mode == "softmax" else "<f4")
    x = np.random.default_rng(2026 + index).normal(size=shape).astype(dtype)
    expected = x
    if mode == "softmax":
        z = x.astype(np.float64)
        z = np.exp(z - z.max(axis=-1, keepdims=True))
        expected = z / z.sum(axis=-1, keepdims=True)
    return [pb.Tensor(name="x", dtype=pb.F16 if mode == "softmax" else pb.F32,
                      shape=shape, data=x.tobytes())], expected


def validate(pb, response, mode, expected):
    outputs = {t.name: t for t in response.outputs}
    if len(outputs) != len(response.outputs):
        raise AssertionError("duplicate output names")
    if mode == "luxtts":
        if set(outputs) != {"audio", "sample_rate"}:
            raise AssertionError("missing LuxTTS output contract")
        audio, rate = outputs["audio"], outputs["sample_rate"]
        if audio.dtype != pb.F32 or len(audio.shape) != 1 or audio.shape[0] < 1:
            raise AssertionError("invalid audio dtype/shape")
        values = np.frombuffer(audio.data, dtype="<f4")
        if len(values) != audio.shape[0] or not np.isfinite(values).all() or not np.any(values):
            raise AssertionError("empty, silent, malformed, or non-finite audio")
        if rate.dtype != pb.I32 or list(rate.shape) != [1] or rate.data != np.array([48000], dtype="<i4").tobytes():
            raise AssertionError("invalid sample rate")
        return
    if set(outputs) != {"y"}:
        raise AssertionError("expected exactly one output named y")
    output = outputs["y"]
    dtype, enum = ("<f2", pb.F16) if mode == "softmax" else ("<f4", pb.F32)
    if output.dtype != enum or list(output.shape) != list(expected.shape):
        raise AssertionError("output dtype/shape mismatch")
    actual = np.frombuffer(output.data, dtype=dtype).reshape(expected.shape)
    if not np.isfinite(actual).all():
        raise AssertionError("non-finite output")
    if mode == "identity":
        np.testing.assert_array_equal(actual, expected)
    else:
        np.testing.assert_allclose(actual, expected, rtol=0.015, atol=0.002)


def exercise(stub, pb, mode, model, shape, prompt, index, timeout):
    inputs, expected = fixture(pb, mode, index, shape, prompt)
    start = time.perf_counter()
    response = stub.Infer(pb.InferRequest(model=model, inputs=inputs), timeout=timeout)
    elapsed_ms = (time.perf_counter() - start) * 1000
    validate(pb, response, mode, expected)
    return elapsed_ms


def benchmark_endpoint(pb, pb_grpc, args, endpoint, model, prompt):
    result = {"endpoint": endpoint, "model": model, "status": "failed", "runs": []}
    try:
        with grpc.insecure_channel(endpoint, options=[("grpc.max_receive_message_length", 64 * 1024 * 1024)]) as channel:
            grpc.channel_ready_future(channel).result(timeout=args.ready_timeout)
            stub = pb_grpc.InferenceServiceStub(channel)
            # First RPC includes any lazy model work/autotuning. With a shared worker,
            # only the first endpoint can incur that cost; this is not boot time.
            result["first_request_ms"] = exercise(stub, pb, args.mode, model, args.shape, prompt, 0, args.cold_timeout)
            for i in range(args.warmup):
                exercise(stub, pb, args.mode, model, args.shape, prompt, i + 1, args.timeout)
            for concurrency in args.concurrency:
                def call(i):
                    try:
                        return exercise(stub, pb, args.mode, model, args.shape, prompt,
                                        1000 + i, args.timeout), None
                    except Exception as exc:
                        return None, str(exc)[:500]
                started = time.perf_counter()
                with ThreadPoolExecutor(max_workers=concurrency) as pool:
                    observations = list(pool.map(call, range(args.requests)))
                seconds = time.perf_counter() - started
                latencies = [lat for lat, err in observations if err is None]
                errors = [err for _, err in observations if err is not None]
                result["runs"].append({
                    "concurrency": concurrency, "requests": args.requests,
                    "successes": len(latencies), "errors": len(errors),
                    "error_examples": errors[:3], "wall_seconds": seconds,
                    "successful_requests_per_second": len(latencies) / seconds,
                    "latency_ms": dict(zip(("p50", "p95", "p99"),
                                           np.percentile(latencies, [50, 95, 99]).tolist())) if latencies else None,
                })
            result["status"] = "passed" if all(r["errors"] == 0 for r in result["runs"]) else "failed"
    except Exception as exc:
        result["error"] = str(exc)[:1000]
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--endpoint", default="127.0.0.1:50051")
    parser.add_argument("--baseline", help="direct sidecar gRPC endpoint")
    parser.add_argument("--mode", choices=["softmax", "identity", "luxtts"], default="softmax")
    parser.add_argument("--baseline-model", default="softmax_two_pass")
    parser.add_argument("--shape", type=int, nargs="+", default=[4, 128])
    parser.add_argument("--prompt-audio", type=Path)
    parser.add_argument("--concurrency", type=int, nargs="+", default=[1, 4, 8])
    parser.add_argument("--requests", type=int, default=100)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--timeout", type=float, default=30)
    parser.add_argument("--cold-timeout", type=float, default=900)
    parser.add_argument("--ready-timeout", type=float, default=600)
    parser.add_argument("--output", type=Path, default=Path("results/benchmark.json"))
    args = parser.parse_args()
    if min(args.requests, *args.concurrency, *args.shape, args.timeout, args.cold_timeout, args.ready_timeout) <= 0 or args.warmup < 0:
        parser.error("counts, shapes, and timeouts must be positive; warmup may be zero")
    if args.mode == "softmax" and len(args.shape) != 2:
        parser.error("softmax requires a two-dimensional shape")
    if args.mode == "luxtts" and args.prompt_audio is None:
        parser.error("LuxTTS requires --prompt-audio with an authorized reference recording")
    prompt = args.prompt_audio.read_bytes() if args.prompt_audio else None
    if args.mode == "luxtts" and not prompt:
        parser.error("prompt audio is empty")
    pb, pb_grpc = compile_proto(str(ROOT))
    report = {"schema_version": 1, "mode": args.mode, "shape": args.shape,
              "seed": 2026, "warmup": args.warmup, "endpoints": {},
              "measurement": "closed-loop client RPC latency; throughput includes client validation",
              "first_request_note": "baseline runs first; proxy shares the warmed worker; neither measures instance boot",
              "prompt_sha256": hashlib.sha256(prompt).hexdigest() if prompt else None}
    if args.baseline:
        report["endpoints"]["direct_worker"] = benchmark_endpoint(pb, pb_grpc, args, args.baseline, args.baseline_model, prompt)
    report["endpoints"]["kernelport"] = benchmark_endpoint(pb, pb_grpc, args, args.endpoint, "demo", prompt)
    report["status"] = "passed" if all(r["status"] == "passed" for r in report["endpoints"].values()) else "failed"
    try:
        report["gpu_snapshot"] = subprocess.check_output([
            "nvidia-smi", "--query-gpu=name,uuid,driver_version,memory.total,memory.used",
            "--format=csv"], text=True, timeout=10)
    except (OSError, subprocess.SubprocessError):
        report["gpu_snapshot"] = None
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(f"Validation {report['status']}: {args.output}")
    return 0 if report["status"] == "passed" else 1


if __name__ == "__main__":
    sys.exit(main())

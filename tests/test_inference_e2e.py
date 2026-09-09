"""Real Rust server + deterministic CPU sidecar; no cloud/GPU needed."""
from concurrent.futures import ThreadPoolExecutor
import importlib.util
import json
import sys
import os
from pathlib import Path
import socket
import subprocess
import tempfile
import unittest

import grpc
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("benchmark", ROOT / "scripts/validation/benchmark.py")
b = importlib.util.module_from_spec(spec)
spec.loader.exec_module(b)
pb, pb_grpc = b.compile_proto(str(ROOT))


class Sidecar:
    def Infer(self, req, context):
        values = {t.name: t for t in req.inputs}
        if "fail" in values:
            context.abort(grpc.StatusCode.INTERNAL, "intentional backend failure")
        if "text" in values:
            if "prompt_audio" not in values:
                context.abort(grpc.StatusCode.INVALID_ARGUMENT, "lost named input")
            return pb.InferResponse(outputs=[
                pb.Tensor(name="audio", dtype=pb.F32, shape=[2], data=np.array([0.25, -0.25], dtype="<f4").tobytes()),
                pb.Tensor(name="sample_rate", dtype=pb.I32, shape=[1], data=np.array([48000], dtype="<i4").tobytes())])
        x = values["x"]
        dtype = "<f2" if x.dtype == pb.F16 else "<f4"
        array = np.frombuffer(x.data, dtype=dtype).reshape(x.shape).astype(np.float64)
        exp = np.exp(array - array.max(axis=-1, keepdims=True))
        y = (exp / exp.sum(axis=-1, keepdims=True)).astype(dtype)
        return pb.InferResponse(outputs=[pb.Tensor(name="y", dtype=x.dtype, shape=x.shape, data=y.tobytes())])


class InferenceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.binary = os.environ.get("KERNELPORT_BIN", str(ROOT / "target/debug/kernelportd"))
        cls.worker = grpc.server(ThreadPoolExecutor(max_workers=8))
        pb_grpc.add_InferenceServiceServicer_to_server(Sidecar(), cls.worker)
        worker_port = cls.worker.add_insecure_port("127.0.0.1:0")
        cls.worker.start()
        cls.worker_endpoint = f"127.0.0.1:{worker_port}"
        with socket.socket() as sock:
            sock.bind(("127.0.0.1", 0))
            port = sock.getsockname()[1]
        cls.logs = tempfile.TemporaryFile()
        cls.process = subprocess.Popen([cls.binary, "serve", "--backend", "helion",
            "--helion-addr", f"http://127.0.0.1:{worker_port}",
            "--grpc-addr", f"127.0.0.1:{port}"], stdout=cls.logs, stderr=cls.logs)
        cls.endpoint = f"127.0.0.1:{port}"
        cls.channel = grpc.insecure_channel(cls.endpoint)
        try:
            grpc.channel_ready_future(cls.channel).result(timeout=15)
        except Exception:
            cls.process.terminate()
            cls.process.wait(timeout=10)
            cls.logs.seek(0)
            raise RuntimeError(cls.logs.read().decode())
        cls.stub = pb_grpc.InferenceServiceStub(cls.channel)

    @classmethod
    def tearDownClass(cls):
        cls.channel.close()
        cls.process.terminate()
        cls.process.wait(timeout=10)
        cls.worker.stop(0).wait()
        cls.logs.close()

    def test_concurrent_distinct_requests_receive_their_own_outputs(self):
        def call(i):
            return b.exercise(self.stub, pb, "softmax", "demo", [4, 128], None, i, 10)
        with ThreadPoolExecutor(max_workers=16) as pool:
            self.assertEqual(len(list(pool.map(call, range(64)))), 64)

    def test_luxtts_named_inputs_and_outputs_survive_proxy(self):
        b.exercise(self.stub, pb, "luxtts", "demo", [], b"fixture", 1, 10)

    def test_backend_errors_are_rpc_failures(self):
        with self.assertRaises(grpc.RpcError) as caught:
            self.stub.Infer(pb.InferRequest(model="demo", inputs=[pb.Tensor(name="fail", dtype=pb.U8, shape=[1], data=b"x")]), timeout=10)
        self.assertEqual(caught.exception.code(), grpc.StatusCode.INTERNAL)

    def test_unknown_model_is_rejected(self):
        with self.assertRaises(grpc.RpcError) as caught:
            self.stub.Infer(pb.InferRequest(model="missing"), timeout=10)
        self.assertEqual(caught.exception.code(), grpc.StatusCode.NOT_FOUND)

    def test_missing_model_fails_startup(self):
        proc = subprocess.run([self.binary, "serve", "--model-path", "/missing/model.onnx"], capture_output=True, timeout=15)
        self.assertNotEqual(proc.returncode, 0)
        self.assertIn(b"failed to load ONNX model", proc.stderr)

    def test_cli_report_compares_baseline_and_proxy(self):
        with tempfile.TemporaryDirectory() as tmp:
            output = Path(tmp) / "benchmark.json"
            run = subprocess.run([sys.executable, str(ROOT / "scripts/validation/benchmark.py"),
                "--endpoint", self.endpoint, "--baseline", self.worker_endpoint,
                "--requests", "12", "--warmup", "1", "--concurrency", "1", "4",
                "--output", str(output)], capture_output=True, text=True, timeout=30)
            self.assertEqual(run.returncode, 0, run.stderr)
            report = json.loads(output.read_text())
            self.assertEqual(report["status"], "passed")
            self.assertEqual(set(report["endpoints"]), {"direct_worker", "kernelport"})
            for endpoint in report["endpoints"].values():
                self.assertEqual(len(endpoint["runs"]), 2)
                self.assertTrue(all(r["successes"] == 12 for r in endpoint["runs"]))

    def test_cli_unavailable_endpoint_writes_failed_report(self):
        with tempfile.TemporaryDirectory() as tmp:
            output = Path(tmp) / "benchmark.json"
            run = subprocess.run([sys.executable, str(ROOT / "scripts/validation/benchmark.py"),
                "--endpoint", "127.0.0.1:1", "--ready-timeout", "0.1",
                "--output", str(output)], capture_output=True, text=True, timeout=15)
            self.assertEqual(run.returncode, 1, run.stderr)
            self.assertEqual(json.loads(output.read_text())["status"], "failed")

    def test_validator_rejects_empty_and_wrong_outputs(self):
        _, expected = b.fixture(pb, "softmax", 123, [4, 128])
        with self.assertRaises(AssertionError):
            b.validate(pb, pb.InferResponse(), "softmax", expected)
        wrong = pb.InferResponse(outputs=[pb.Tensor(name="y", dtype=pb.F16, shape=[4, 128], data=np.zeros((4,128), dtype="<f2").tobytes())])
        with self.assertRaises(AssertionError):
            b.validate(pb, wrong, "softmax", expected)


if __name__ == "__main__":
    unittest.main()

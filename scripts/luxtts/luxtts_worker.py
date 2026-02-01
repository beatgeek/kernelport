"""
LuxTTS gRPC worker: implements kernelport.v1.InferenceService for LuxTTS.

Input tensors (LuxTTS contract): text (U8), prompt_audio (U8), optional rms, t_shift,
num_steps, speed, return_smooth, ref_duration. Output: audio (F32), sample_rate (I32).
"""
from __future__ import annotations

import argparse
import os
import struct
import sys
import tempfile
from concurrent import futures
from typing import Any, Dict, Optional

import grpc
import numpy as np

# Ensure script dir is on path for proto_util
_script_dir = os.path.dirname(os.path.abspath(__file__))
if _script_dir not in sys.path:
    sys.path.insert(0, _script_dir)
from proto_util import compile_proto, find_repo_root

# LuxTTS repo is expected at LUXTTS_REPO or /app/LuxTTS in Docker
_luxtts_repo = os.environ.get("LUXTTS_REPO", "/app/LuxTTS")
if os.path.isdir(_luxtts_repo):
    sys.path.insert(0, _luxtts_repo)

# After path is set, import LuxTTS
try:
    from zipvoice.luxvoice import LuxTTS
except ImportError as e:
    LuxTTS = None  # type: ignore
    _import_error = e
else:
    _import_error = None

# DType enum values from inference.proto
DTYPE_F32 = 1
DTYPE_I32 = 4
DTYPE_U8 = 5


def _inputs_by_name(request: Any) -> Dict[str, Any]:
    """Build a dict of input name -> tensor (name, dtype, shape, data)."""
    out = {}
    for t in request.inputs:
        out[t.name] = {"dtype": t.dtype, "shape": list(t.shape), "data": t.data}
    return out


def _scalar_f32(data: bytes) -> float:
    if len(data) < 4:
        return 0.0
    return struct.unpack("<f", data[:4])[0]


def _scalar_i32(data: bytes) -> int:
    if len(data) < 4:
        return 0
    return struct.unpack("<i", data[:4])[0]


def _tensor_pb(pb_module: Any, name: str, dtype: int, shape: list, data: bytes) -> Any:
    return pb_module.Tensor(name=name, dtype=dtype, shape=shape, data=data)


class LuxTTSService:
    def __init__(self, model_name: str, device: str, pb_module: Any) -> None:
        if LuxTTS is None:
            raise RuntimeError(f"LuxTTS not available: {_import_error}")
        self.pb = pb_module
        self.device = device
        self.model_name = model_name
        token = os.environ.get("HUGGINGFACE_HUB_TOKEN") or os.environ.get("HF_TOKEN")
        self.lux_tts = LuxTTS("YatharthS/LuxTTS", device=device, token=token)

    def Infer(self, request: Any, context: grpc.ServicerContext) -> Any:
        if request.model != self.model_name:
            context.abort(grpc.StatusCode.NOT_FOUND, f"unknown model: {request.model}")

        inputs = _inputs_by_name(request)
        if "text" not in inputs or "prompt_audio" not in inputs:
            context.abort(
                grpc.StatusCode.INVALID_ARGUMENT,
                "inputs must include 'text' (U8) and 'prompt_audio' (U8)",
            )

        try:
            text = inputs["text"]["data"].decode("utf-8")
        except Exception as e:
            context.abort(grpc.StatusCode.INVALID_ARGUMENT, f"text must be UTF-8: {e}")

        prompt_audio_bytes = inputs["prompt_audio"]["data"]
        rms = _scalar_f32(inputs["rms"]["data"]) if "rms" in inputs else 0.01
        t_shift = _scalar_f32(inputs["t_shift"]["data"]) if "t_shift" in inputs else 0.9
        num_steps = _scalar_i32(inputs["num_steps"]["data"]) if "num_steps" in inputs else 4
        speed = _scalar_f32(inputs["speed"]["data"]) if "speed" in inputs else 1.0
        return_smooth = bool(_scalar_i32(inputs["return_smooth"]["data"])) if "return_smooth" in inputs else False
        ref_duration = _scalar_i32(inputs["ref_duration"]["data"]) if "ref_duration" in inputs else 5

        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
            f.write(prompt_audio_bytes)
            prompt_path = f.name
        try:
            encoded_prompt = self.lux_tts.encode_prompt(
                prompt_path, duration=ref_duration, rms=rms
            )
            final_wav = self.lux_tts.generate_speech(
                text,
                encoded_prompt,
                num_steps=num_steps,
                t_shift=t_shift,
                speed=speed,
                return_smooth=return_smooth,
            )
        finally:
            try:
                os.unlink(prompt_path)
            except OSError:
                pass

        # LuxTTS returns tensor; ensure numpy float32 1D
        if hasattr(final_wav, "numpy"):
            audio_np = final_wav.numpy().squeeze().astype(np.float32)
        else:
            audio_np = np.asarray(final_wav, dtype=np.float32).squeeze()
        if audio_np.ndim != 1:
            audio_np = audio_np.ravel()
        audio_bytes = audio_np.tobytes()
        sample_rate = 48000
        sr_bytes = np.array([sample_rate], dtype=np.int32).tobytes()

        return self.pb.InferResponse(
            outputs=[
                _tensor_pb(self.pb, "audio", DTYPE_F32, [len(audio_np)], audio_bytes),
                _tensor_pb(self.pb, "sample_rate", DTYPE_I32, [1], sr_bytes),
            ],
            queued_us=0,
            batched_us=0,
            backend_us=0,
        )


def main() -> None:
    parser = argparse.ArgumentParser(description="LuxTTS gRPC worker")
    parser.add_argument("--addr", default="0.0.0.0:50061")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--model-name", default="luxtts")
    args = parser.parse_args()

    repo_root = find_repo_root()
    pb_mod, pb_grpc_mod = compile_proto(repo_root)

    service = LuxTTSService(args.model_name, args.device, pb_mod)
    server = grpc.server(thread_pool=futures.ThreadPoolExecutor(max_workers=4))
    pb_grpc_mod.add_InferenceServiceServicer_to_server(service, server)
    server.add_insecure_port(args.addr)
    server.start()
    print(f"LuxTTS worker listening on {args.addr}")
    server.wait_for_termination()


if __name__ == "__main__":
    main()

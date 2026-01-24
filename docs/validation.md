# Validation

This document covers quick CPU/GPU validation and a minimal gRPC test.

## 1) CPU: run the sample model

```bash
docker run --rm -p 50051:50051 \
  -v "$PWD/models/sample:/models/sample:ro" \
  kernelport:cpu \
  --device cpu \
  --model-path /models/sample/model.onnx
```

## 2) GPU: run with CUDA + ORT dylib

```bash
docker run --rm --gpus all -p 50051:50051 \
  -e ORT_DYLIB_PATH=/opt/ort/libonnxruntime.so \
  -v /path/to/ort/libonnxruntime.so:/opt/ort/libonnxruntime.so:ro \
  -v "$PWD/models/sample:/models/sample:ro" \
  kernelport:gpu \
  --device cuda:0 \
  --model-path /models/sample/model.onnx
```

## 3) Minimal gRPC test

This uses `grpcurl` and assumes the server is on `localhost:50051`.

```bash
grpcurl -plaintext -d '{
  "model": "demo",
  "inputs": [
    { "name": "x", "dtype": "F32", "shape": [1, 1, 2, 2], "data": "AACAPwAAAEAAAEBAAACAQA==" }
  ]
}' localhost:50051 kernelport.v1.InferenceService/Infer
```

Expected response (identity model):

```json
{
  "outputs": [
    { "name": "y", "dtype": "F32", "shape": ["1","1","2","2"], "data": "AACAPwAAAEAAAEBAAACAQA==" }
  ],
  "queuedUs": "0",
  "batchedUs": "0",
  "backendUs": "..."
}
```

Notes:
- `data` is raw bytes, base64-encoded (`[1.0,2.0,3.0,4.0]` = `AACAPwAAAEAAAEBAAACAQA==`).
- Input/output names must match the model (sample identity model uses `x` -> `y`).
- `dtype` is an enum; grpcurl accepts the symbolic name (`F32`).

## 4) Helion softmax (experimental)

Prereqs (Python, uv):
```bash
uv venv .venv
source .venv/bin/activate
uv pip install "torch==2.9.*" --index-url https://download.pytorch.org/whl/cu126
uv pip install helion grpcio grpcio-tools numpy
```

Start the Helion worker:
```bash
python scripts/helion/helion_worker.py --addr 0.0.0.0:50061 --device cuda
```

Start kernelportd with the Helion backend:
```bash
cargo run -p kernelport-server -- serve \
  --backend helion \
  --device cuda:0 \
  --helion-addr http://127.0.0.1:50061 \
  --helion-model softmax_two_pass
```

Generate a base64 payload for a small F16 input:
```bash
python - <<'PY'
import base64
import torch
x = torch.randn(4, 8, dtype=torch.float16)
print(base64.b64encode(x.numpy().tobytes()).decode())
PY
```

Then call the server (paste the base64 string into `data`):
```bash
grpcurl -plaintext -d '{
  "model": "demo",
  "inputs": [
    { "name": "x", "dtype": "F16", "shape": [4, 8], "data": "<BASE64>" }
  ]
}' localhost:50051 kernelport.v1.InferenceService/Infer
```

## 5) Mock Helion (CPU, Docker)

Start the CPU mock sidecar + kernelport:
```bash
docker compose -f docker-compose.mock.yml up --build
```

Generate a base64 payload:
```bash
python - <<'PY'
import base64
import numpy as np
x = np.random.randn(4, 8).astype(np.float32)
print(base64.b64encode(x.tobytes()).decode())
PY
```

Then call the server (paste the base64 string into `data`):
```bash
grpcurl -plaintext -d '{
  "model": "demo",
  "inputs": [
    { "name": "x", "dtype": "F32", "shape": [4, 8], "data": "<BASE64>" }
  ]
}' localhost:50051 kernelport.v1.InferenceService/Infer
```

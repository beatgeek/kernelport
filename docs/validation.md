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

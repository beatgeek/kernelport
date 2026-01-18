# KernelPort

**KernelPort** is a high-performance inference server that lets you run models
from TensorFlow, PyTorch, ONNX, and TensorRT **without giving up their native GPU kernels**.

Bring your kernels with you. Serve them from one runtime.

---

## Why KernelPort?

Most inference servers force you into a single runtime or a lowest-common-denominator
format. KernelPort takes a different approach:

- **Native execution**: TensorFlow runs with TensorFlow kernels. PyTorch runs with PyTorch kernels.
- **GPU-first design**: Built for CUDA streams, batching, and memory reuse.
- **One server, many backends**: ONNX Runtime, TensorRT, LibTorch, TensorFlow.
- **Production-grade**: Dynamic batching, scheduling, backpressure, observability.

KernelPort is written in **Rust** for safety, performance, and predictable latency.

---

## What KernelPort Is (and Is Not)

**KernelPort is:**
- An inference *orchestrator* with pluggable execution backends
- A way to consolidate heterogeneous model stacks into one serving plane
- A system designed for low-latency, high-throughput GPU inference

**KernelPort is not:**
- A training framework
- A model compiler
- A replacement for TensorFlow or PyTorch

---

## Core Concepts

### Native Backend Execution
KernelPort does not reimplement ML frameworks. Instead, it embeds and orchestrates
their production runtimes:

| Backend | Execution |
|------|----------|
| PyTorch | LibTorch (TorchScript / compiled artifacts) |
| TensorFlow | TensorFlow C API (SavedModel) |
| ONNX | ONNX Runtime (CUDA / TensorRT EP) |
| TensorRT | Native TensorRT engines |

This ensures you get:
- Vendor-optimized kernels
- Framework correctness
- No silent performance regressions

---

### Dynamic Batching
Requests are dynamically batched per model and shape:
- Configurable max batch size
- Tight batching windows (ms-level)
- Shape-aware grouping

This improves GPU utilization without sacrificing latency SLAs.

---

### GPU-Aware Scheduling
KernelPort schedules work across:
- GPUs (or MIG slices)
- CUDA streams
- Backend-specific execution contexts

The scheduler can be tuned for:
- Throughput
- Latency
- Fairness across models or tenants

---

## Architecture (High Level)

![KernelPort Architecture](docs/kernelport-hl-2026-01-04-005213.svg)

![KernelPort GPU Worker](docs/kernelport-gpu-2026-01-04-005556.svg)

---

## Local Development

### CPU (macOS or Linux)

- Install Rust components: `rustup component add rustfmt clippy`
- Run checks: `make check`
- Run the server:
  - `cargo run -p kernelport-server -- serve --device cpu`
- ORT identity test (CPU): `cargo test -p kernelport-backend-ort --test identity`

### CUDA (Linux + NVIDIA)

KernelPort supports CUDA via ONNX Runtime. For GPU inference:

- Provide a CUDA-enabled ONNX Runtime shared library (build from source or use a CUDA-enabled package).
- Set `ORT_DYLIB_PATH` to the CUDA-enabled `libonnxruntime.so`.
- Build with the CUDA feature and select a device:
  - `cargo run -p kernelport-server --features ort-cuda -- serve --device cuda:0`

---

## Containers (CPU and GPU)

KernelPort ships with separate Dockerfiles for CPU and GPU runtime environments:

- `Dockerfile.cpu` builds a CPU-only image intended for local dev or CPU deployments.
- `Dockerfile.gpu` builds a GPU-ready image that expects CUDA + TensorRT on the host.
  - Base image: `nvidia/cuda:12.2.0-runtime-ubuntu22.04`

The GPU image is designed for "bring your own kernel" by letting you mount a
CUDA-enabled ONNX Runtime shared library at runtime:

- Mount your `libonnxruntime.so` and set `ORT_DYLIB_PATH`.
- Run with `--gpus` and choose `--device cuda:N`.

### Recommended NVIDIA stack

- Driver: 535+ (or newer)
- CUDA: 12.2
- cuDNN: 8.9
- TensorRT: 8.6

### Build

```bash
docker build -f Dockerfile.cpu -t kernelport:cpu .
docker build -f Dockerfile.gpu -t kernelport:gpu .
```

### Run (CPU)

```bash
docker run --rm -p 8080:8080 kernelport:cpu --device cpu
```

### Run (CPU + HF pull)

```bash
docker run --rm -p 8080:8080 \
  -e MODEL_MANIFEST_PATH=/app/models/manifest.yaml \
  -e HF_TOKEN=your_hf_token \
  -v /path/to/manifest.yaml:/app/models/manifest.yaml:ro \
  -v /path/to/model-cache:/models \
  kernelport:cpu --device cpu --model-path /models/bert-base-uncased/model.onnx
```

### Run (GPU)

```bash
docker run --rm --gpus all \
  -e ORT_DYLIB_PATH=/opt/ort/libonnxruntime.so \
  -v /path/to/ort/libonnxruntime.so:/opt/ort/libonnxruntime.so:ro \
  -p 8080:8080 \
  kernelport:gpu --device cuda:0 --model-path /models/bert-base-uncased/model.onnx
```

### Run (GPU + HF pull)

```bash
docker run --rm --gpus all \
  -e ORT_DYLIB_PATH=/opt/ort/libonnxruntime.so \
  -e MODEL_MANIFEST_PATH=/app/models/manifest.yaml \
  -e HF_TOKEN=your_hf_token \
  -v /path/to/ort/libonnxruntime.so:/opt/ort/libonnxruntime.so:ro \
  -v /path/to/manifest.yaml:/app/models/manifest.yaml:ro \
  -v /path/to/model-cache:/models \
  -p 8080:8080 \
  kernelport:gpu --device cuda:0 --model-path /models/bert-base-uncased/model.onnx
```

See `docs/model-ingestion.md` for the HuggingFace model ingestion plan and
`docs/model-manifest.md` for the proposed manifest schema.

Validation steps are in `docs/validation.md`.

## Pre-commit (local)

Install pre-commit and enable the hook:

```bash
pipx install pre-commit
pre-commit install
```

Run on demand:

```bash
pre-commit run --all-files
```

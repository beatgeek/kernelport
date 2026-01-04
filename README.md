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


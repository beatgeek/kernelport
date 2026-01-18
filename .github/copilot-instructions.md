# Copilot Instructions for KernelPort

## Repository Summary
- KernelPort is a Rust-based inference server/orchestrator that serves TensorFlow, PyTorch (TorchScript), ONNX Runtime, and TensorRT models with native kernels.
- Primary runtime is Rust; Dockerfiles provide CPU/GPU container images. Small helper scripts are in `scripts/` (Python + shell).
- Repo size is moderate (Rust workspace with multiple crates). Main targets: server (`kernelportd`), ONNX Runtime backend, and gRPC API.

## High-Level Project Info
- Languages: Rust (core/runtime/server), shell/Python (container entrypoint + model fetch).
- Build system: Cargo workspace (`Cargo.toml` at repo root).
- gRPC stack: tonic 0.14 + prost 0.14; protobufs in `crates/kernelport-proto/src/inference.proto`.
- Containers: `Dockerfile.cpu` (Debian bookworm runtime), `Dockerfile.gpu` (CUDA runtime on Ubuntu 22.04).

## Build / Validate (validated commands)
Always prefer these sequences and avoid ad-hoc commands unless needed.

Bootstrap (local dev):
- Install Rust stable and components: `rustup component add rustfmt clippy`
- Optional but useful: `pipx install pre-commit` then `pre-commit install`

Format (validated):
- `cargo fmt --all`
- Pre-commit hook mirrors this (`.pre-commit-config.yaml`).

Lint (validated):
- `cargo clippy --all-targets --all-features -- -D warnings`

Test (validated):
- `cargo test --all`
- Targeted ORT test (validated): `cargo test -p kernelport-backend-ort --test identity`

Build (validated):
- `cargo build --all`
- If a build runs without network access, Cargo may fail to download crates; ensure network access is available or dependencies are cached.

Run (validated via docs):
- Local server (CPU): `cargo run -p kernelport-server -- serve --device cpu`
- CUDA local dev (Linux + NVIDIA): set `ORT_DYLIB_PATH` to a CUDA-enabled `libonnxruntime.so` and run with `--features ort-cuda --device cuda:N`.
- Validation via containers is documented in `docs/validation.md`.

Containers (validated builds):
- `docker build -f Dockerfile.cpu -t kernelport:cpu .`
- `docker build -f Dockerfile.gpu -t kernelport:gpu .`
- GPU runtime requires NVIDIA drivers + `--gpus` and a mounted CUDA-enabled `libonnxruntime.so`.

Known pitfalls and workarounds:
- gRPC reflection is enabled; use `grpcurl` without `-protoset` after starting the server.
- The sample ONNX model expects input name `x` with shape `[1,1,2,2]` and outputs `y`; see `docs/validation.md`.
- `clippy` enforces `-D warnings`; do not leave warnings in `kernelport-backend-ort`.

## CI / Checks (must match locally)
`.github/workflows/ci.yml` runs:
- `cargo fmt --all -- --check`
- `cargo clippy --all-targets --all-features -- -D warnings`
- `cargo test --all`
- `cargo build --all`
- Extra CPU ORT identity test: `cargo test -p kernelport-backend-ort --test identity`

## Project Layout & Architecture
Root layout (files):
- `Cargo.toml`, `Cargo.lock` (workspace), `Makefile` (fmt/clippy/test/build/check targets)
- `Dockerfile.cpu`, `Dockerfile.gpu`, `.pre-commit-config.yaml`, `.dockerignore`
- `README.md` (concepts, dev + container usage, pre-commit)
- `docs/` (model ingestion, manifest, validation)
- `models/` (sample ONNX bundle + example manifest)
- `scripts/` (entrypoint + HF model fetcher)
- `.github/workflows/ci.yml`

Key Rust crates:
- `crates/kernelport-server/`:
  - `src/main.rs` (server entry; batcher/scheduler/worker wiring; reflection)
  - `src/grpc.rs` (gRPC Infer handler, DType mapping)
  - `src/cli.rs` (CLI flags: `--device`, `--model-path`, `--grpc-addr`, `--log`)
  - `src/registry.rs` (model loading via ORT, output name capture)
- `crates/kernelport-backend-ort/`:
  - ONNX Runtime backend; CPU and optional CUDA feature.
- `crates/kernelport-proto/`:
  - `src/inference.proto` (gRPC API, `Infer` RPC, `DType` enum)
  - `build.rs` (generates descriptor set for reflection)
- `crates/kernelport-core/` and `crates/kernelport-runtime/` (tensor types, batching, scheduler/worker).

Docs and validated validation flow:
- `docs/validation.md` (CPU/GPU run + grpcurl example).
- `docs/model-ingestion.md` and `docs/model-manifest.md` (HF model ingestion plan).

## Read This First (reduce exploration)
- Assume `kernelportd` is the server binary.
- Trust these instructions and only search if information here is missing or incorrect.

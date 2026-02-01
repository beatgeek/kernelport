# Helion Workers

- `helion_worker.py`: CUDA Helion worker (real kernel, needs GPU).
- `mock/helion_worker_mock.py`: CPU-only mock worker for local testing.

The mock worker mirrors the gRPC interface but uses NumPy softmax and runs
without CUDA or Helion installed.

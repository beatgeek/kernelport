# HuggingFace Model Ingestion (Plan)

KernelPort will support two model delivery modes so users can trade off
performance vs. flexibility.

## Modes

1) Baked into image (maximum performance, reproducible)
   - Artifacts are copied into the container at build time.
   - Images are per-model (or per-bundle).

2) Pulled at startup (flexible, cacheable)
   - Container pulls from HuggingFace into a mounted cache volume.
   - Runtime loads from the cache directory.

## Formats (initial)

- ONNX
- TorchScript
- TensorRT (requires CUDA + TensorRT toolchain)

## Proposed Flow

1) Provide a manifest (see `docs/model-manifest.md`).
2) An entrypoint or sidecar fetches artifacts into `/models`.
3) KernelPort loads the artifacts from the manifest path.

## Loading from the cache

The server currently accepts a direct model path:

```bash
kernelport-server serve --model-path /models/<id>/model.onnx
```

## HuggingFace Authentication

Set `HF_TOKEN` (or `HUGGINGFACE_HUB_TOKEN`) when pulling from private repos.

## Container Integration

- Bake mode: copy artifacts into the image under `/models`.
- Pull mode: mount a volume at `/models` and populate it at startup.
- GPU images should mount a CUDA-enabled `libonnxruntime.so` via `ORT_DYLIB_PATH`
  when using ONNX Runtime with CUDA.

## Examples

- Sample bundle: `models/sample/model.onnx`
- Sample manifest: `models/sample-manifest.yaml`

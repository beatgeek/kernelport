# Model Manifest (Proposed)

This document defines a proposed manifest used to fetch, cache, and load models
from HuggingFace or other sources. It is not yet implemented in the runtime.

## Goals

- Keep model source info declarative.
- Support ONNX, TorchScript, and TensorRT artifacts.
- Allow a "fetch at startup" flow without forcing new images per model.

## Schema (YAML)

```yaml
id: bert-base-uncased
source:
  kind: huggingface
  repo: bert-base-uncased
  revision: main
format: onnx # onnx | torchscript | tensorrt
files:
  - model.onnx
cache:
  dir: /models
runtime:
  device: cuda:0
```

## Notes

- `files` lists the expected artifact filenames once fetched or generated.
- `cache.dir` is where the container stores artifacts when pulling at startup.
- The runtime may later accept a `conversion` block for ONNX or TensorRT builds.
- For private repos, set `HF_TOKEN` (or `HUGGINGFACE_HUB_TOKEN`) in the container.
- See `models/sample-manifest.yaml` for a concrete example.

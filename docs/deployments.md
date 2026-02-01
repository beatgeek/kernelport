# Deployments

This document describes how to run each KernelPort deployment (model + backend + optional worker).

## Deployment list

| Name           | Manifest                    | Backend | Worker image           | How to run |
|----------------|-----------------------------|---------|------------------------|------------|
| identity-onnx  | models/sample-manifest.yaml | onnx    | —                      | `MODEL_MANIFEST_PATH=... model_fetch` then `serve --backend onnx --model-path /models/<id>/model.onnx` |
| helion-demo    | —                           | helion  | kernelport-helion:gpu  | `docker compose up` (default) |
| luxtts         | models/luxtts-manifest.yaml | helion  | kernelport-luxtts:gpu  | `docker compose -f docker-compose.yml -f docker-compose.luxtts.yml up` or Lambda deploy |

The authoritative mapping is in [deployments/deployments.yaml](../deployments/deployments.yaml).

## Running the LuxTTS deployment

1. Set `HUGGINGFACE_HUB_TOKEN` (or `HF_TOKEN`) so the LuxTTS worker can pull the model from Hugging Face.
2. Build and run with the LuxTTS override:
   ```bash
   docker compose -f docker-compose.yml -f docker-compose.luxtts.yml up --build
   ```
3. Inference endpoint: gRPC at `localhost:50051`, method `kernelport.v1.InferenceService/Infer`. See [LuxTTS tensor contract](#luxtts-tensor-contract) for input/output tensors.

## Lambda Cloud deploy

The GitHub Actions workflow [.github/workflows/deploy-lambda.yml](../.github/workflows/deploy-lambda.yml) can create a Lambda Cloud GPU instance, build and push images to GHCR, and run the LuxTTS stack (or another deployment) via cloud-init.

Required repository secrets:

- **LAMBDA_CLOUD_API_KEY** — Lambda Cloud API key for launching instances.
- **HUGGINGFACE_HUB_TOKEN** — Hugging Face token for pulling models (e.g. LuxTTS); passed to the LuxTTS worker at runtime.

Trigger: `workflow_dispatch` (manual) or push to a branch you configure. See the workflow file for inputs (e.g. instance type).

## LuxTTS tensor contract

Inputs (all optional except `text` and `prompt_audio`; omitted use LuxTTS defaults):

| Name            | Dtype | Shape | Description |
|-----------------|-------|-------|-------------|
| text            | U8    | [N]   | UTF-8 bytes of the text to synthesize. |
| prompt_audio    | U8    | [M]   | Raw reference audio file bytes (WAV or MP3). |
| rms             | F32   | [1]   | Loudness (default 0.01). |
| t_shift         | F32   | [1]   | Sampling param (default 0.9). |
| num_steps       | I32   | [1]   | Sampling steps (default 4). |
| speed           | F32   | [1]   | Playback speed (default 1.0). |
| return_smooth   | I32   | [1]   | 0 = false, 1 = true (default 0). |
| ref_duration    | I32   | [1]   | Reference duration in seconds (default 5). |

Outputs:

| Name         | Dtype | Shape | Description |
|--------------|-------|-------|-------------|
| audio         | F32   | [S]   | Mono waveform at 48 kHz. |
| sample_rate   | I32   | [1]   | 48000. |

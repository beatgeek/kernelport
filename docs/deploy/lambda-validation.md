# Lambda validation: provision → infer → measure → clean up

Run **Actions → Deploy to Lambda Cloud → Run workflow**, selecting the
`filesystem-support` branch while PR #3 is open. Choose `softmax` for the first
GPU validation. The workflow builds the selected images, deploys them by digest,
executes real inference, compares the direct sidecar with KernelPort, and saves
results. It does not merge code or launch on pushes.

## One-time configuration

Repository Actions secrets:

| Secret | Purpose |
|---|---|
| `LAMBDA_CLOUD_API_KEY` | Provision and clean up your Lambda resources |
| `LAMBDA_SSH_PRIVATE_KEY` | Private key matching the public key registered in Lambda under the `ssh_key_name` input |
| `HUGGINGFACE_HUB_TOKEN` | Only required for LuxTTS |
| `LUXTTS_PROMPT_WAV_BASE64` | Only required for LuxTTS: base64 of a small reference WAV you are authorized to use; keep within GitHub's secret size limit |

The default `GITHUB_TOKEN` handles GHCR push/pull. No additional registry token
is required. Use an **x86_64** Lambda image with Docker, NVIDIA Container Toolkit,
Python 3 with venv support, and an `ubuntu` user able to run Docker. SSH from the
Actions runner must be permitted. The script checks these prerequisites and
fails if the GPU or attached filesystem is unavailable. It does not install or
reconfigure host drivers. GH200/ARM hosts are not supported by these images.

SSH uses trust on first use for the new instance IP, then pins that host key for
the remainder of the run. gRPC binds to loopback; it does not require public
50051/50061 access. Tokens travel over SSH stdin, are not baked into images,
and are excluded from artifacts. Remote credentials are removed on exit.

## Inputs and lifecycle

- `instance_type_name`, `region_name`, `ssh_key_name`: choose available Lambda
  capacity and your registered key.
- `filesystem_id`: optional existing filesystem, looked up in the account and
  checked against the selected region. Reused filesystems are always retained.
  Without it, the workflow creates a new filesystem and attaches it by name.
  Lambda filesystems grow with usage: the old `filesystem_size_gb` input was
  removed because it was not a supported quota control.
- `validation_mode`: `softmax` (default) or `luxtts`.
- `retain_instance`: false by default. Successful runs terminate the instance
  and delete storage created by this run. True retains the host and its new
  filesystem for inspection, although test containers stop after collecting
  logs. Failures attempt cleanup regardless of this flag.

Resource IDs are saved immediately after each successful create/launch response
and uploaded before validation. A timeout with an ambiguous launch response is
not retried: inspect the Lambda console for the run name shown in
`resources.json`. This avoids accidentally creating duplicate instances.

Cleanup confirms termination, waits for filesystem detachment, deletes only
owned storage, and confirms absence. HTTP errors and unconfirmed cleanup fail
the workflow. A reused filesystem never prevents termination of a new instance.

**Cancelled runs or runner loss still require manual teardown.** Download the
resource-ID artifact, or identify the instance/filesystem by the run name in the
Lambda console. Use **Teardown Lambda Cloud Resources** once that workflow is
available on the default branch; filesystem deletion is opt-in. Before merge,
you can run the checked-out helper locally:

```bash
export LAMBDA_CLOUD_API_KEY='...'   # prefer loading from your secret manager
export INSTANCE_ID='...'
export FILESYSTEM_ID='...'
export DELETE_FILESYSTEM=false     # true only for storage you intend to delete
python3 scripts/lambda/resources.py teardown
```

## What passes the gate

Softmax uses deterministic F16 tensors with a different seed for each request.
Every response must have the right name, dtype, shape, finite values, and agree
with a float64 reference within the declared tolerance. Concurrency 1, 4, and 8
are tested with 100 requests each; first-request and warm-up work are separate.
LuxTTS checks both named outputs, nonempty finite nonsilent F32 audio, and the
48 kHz sample rate, using 10 requests at concurrency 1. This is an audio contract
test, not a speech-quality evaluation or deterministic waveform comparison.

The direct-worker RPC runs first, so it may include lazy initialization and
Helion autotuning. KernelPort then uses the same warmed worker. `first_request_ms`
is explicitly **not instance boot time**, and the two first-request numbers are
not a cold-start comparison. Warm RPC p50/p95/p99 exclude validation time;
closed-loop throughput includes fixture generation and client validation.

Results artifact contents:

- `resources.json`: commit, resource IDs, and confirmed cleanup outcomes.
- `host-results.tar.gz`: `benchmark.json`, image digests, installed worker
  package versions, container logs, GPU/driver details, and one-second GPU
  utilization/memory samples. It includes an exit code even when validation fails.

Compare warm direct-worker and KernelPort metrics at each concurrency. No fixed
performance threshold is asserted yet; collect a baseline on the same GPU,
image digests, shape, and request counts before setting a regression budget.
One-second GPU samples are observations, not exact peak-memory measurements.

## CPU regression and standalone checks

CI runs a real Rust server against a deterministic CPU sidecar. Tests cover
concurrent distinct requests, LuxTTS input/output names, backend errors, unknown
models, failed model loading, and rejection of invalid validation outputs. The
existing ORT CPU identity test covers ONNX wiring.

The benchmark client is also usable locally:

```bash
uv venv .venv
uv pip install --python .venv/bin/python -r scripts/validation/requirements.txt
.venv/bin/python scripts/validation/benchmark.py \
  --endpoint 127.0.0.1:50051 --baseline 127.0.0.1:50061
```

For an ORT identity server, use `--mode identity --shape 1 1 2 2` with the tracked
`models/identity.onnx` fixture (check its model IO if substituting a different
model). ORT GPU validation remains a separate bring-your-own CUDA-enabled
`libonnxruntime.so` path; the automated GPU gate runs Helion or LuxTTS. The
orchestrator image loads ORT dynamically only when using the ONNX backend,
so Helion/LuxTTS do not require an ORT shared library.

## Current runtime boundary

The server executes each request independently. Actual tensor stacking/splitting
is not implemented; the prior first-output fan-out was incorrect for concurrent
requests and is removed. Model/backend failures now fail the RPC, and startup
fails if model loading fails. Named sidecar inputs and outputs survive the
adapter. Scheduling optimization should follow these correctness gates and
baseline measurements.

API reference: https://docs.lambda.ai/public-cloud/cloud-api/

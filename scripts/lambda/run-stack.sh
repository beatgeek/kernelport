#!/usr/bin/env bash
# Runs on the Lambda host after source and image digests have been copied over SSH.
set -euo pipefail
cd "$(dirname "$0")/../.."
mkdir -p results
worker_id=""
server_id=""
gpu_pid=""
finish() {
  code=$?
  trap - EXIT
  if [[ -n "$gpu_pid" ]]; then kill "$gpu_pid" 2>/dev/null || true; fi
  if [[ -n "$server_id" ]]; then docker logs "$server_id" > results/kernelport.log 2>&1 || true; docker rm -f "$server_id" >/dev/null || true; fi
  if [[ -n "$worker_id" ]]; then docker logs "$worker_id" > results/worker.log 2>&1 || true; docker rm -f "$worker_id" >/dev/null || true; fi
  printf '%s\n' "$code" > results/exit-code.txt
  exit "$code"
}
trap finish EXIT

command -v docker >/dev/null
nvidia-smi > results/nvidia-smi.txt
python3 -m venv .validation-venv
.validation-venv/bin/pip install -r scripts/validation/requirements.txt > results/client-install.log 2>&1
kernelport_image=$(python3 -c 'import json; print(json.load(open("images.json"))["kernelport"])')
worker_image=$(python3 -c 'import json; print(json.load(open("images.json"))["worker"])')
mode=$(python3 -c 'import json; print(json.load(open("images.json"))["mode"])')
cp images.json results/images.json
for image in "$kernelport_image" "$worker_image"; do
  [[ "$image" =~ ^ghcr.io/[a-z0-9_./-]+@sha256:[a-f0-9]{64}$ ]] || { echo 'Expected digest-pinned GHCR image'; exit 1; }
  docker pull "$image"
done
docker image inspect --format '{{json .RepoDigests}}' "$kernelport_image" "$worker_image" > results/image-digests.jsonl

# This directory must be inside the attached filesystem; fail rather than silently
# putting the model/autotune cache on the ephemeral root volume.
mapfile -t mounts < <(findmnt -rn -t nfs,nfs4 -o TARGET)
[[ "${#mounts[@]}" == 1 ]] || { echo 'Expected exactly one attached NFS filesystem'; exit 1; }
cache_root="${mounts[0]}"
mountpoint -q "$cache_root"
mkdir -p "$cache_root/kernelport-cache"
worker_args=(--gpus all --network host -v "$cache_root/kernelport-cache:/cache" -e XDG_CACHE_HOME=/cache -e HF_HOME=/cache/huggingface)
benchmark_args=(--mode "$mode" --baseline 127.0.0.1:50061 --output results/benchmark.json)
if [[ "$mode" == luxtts ]]; then
  [[ -s prompt.wav ]] || { echo 'LuxTTS requires a reference recording'; exit 1; }
  worker_args+=(--env-file worker.env)
  model=luxtts
  benchmark_args+=(--baseline-model luxtts --prompt-audio prompt.wav --requests 10 --concurrency 1)
else
  model=softmax_two_pass
fi
worker_id=$(docker run -d "${worker_args[@]}" "$worker_image" python /app/scripts/"$( [[ "$mode" == luxtts ]] && echo luxtts/luxtts_worker.py || echo helion/helion_worker.py )" --addr 127.0.0.1:50061 --device cuda)
# Wait for the worker listener. The benchmark records first-inference/autotune
# separately from the warm comparison.
.validation-venv/bin/python - "$model" <<'PY'
import sys, grpc
with grpc.insecure_channel('127.0.0.1:50061') as channel:
    grpc.channel_ready_future(channel).result(timeout=900)
PY
server_id=$(docker run -d --network host "$kernelport_image" --backend helion --device cuda:0 --grpc-addr 127.0.0.1:50051 --helion-addr http://127.0.0.1:50061 --helion-model "$model")
nvidia-smi --query-gpu=timestamp,uuid,utilization.gpu,memory.used --format=csv -l 1 > results/gpu-samples.csv &
gpu_pid=$!
.validation-venv/bin/python scripts/validation/benchmark.py "${benchmark_args[@]}"
docker exec "$worker_id" python -c 'import importlib.metadata as m; print("\n".join(sorted(d.metadata.get("Name", "unknown") + "==" + d.version for d in m.distributions())))' > results/worker-packages.txt

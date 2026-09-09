#!/usr/bin/env bash
# No credentials in command arguments, source archive, images.json, or artifacts.
set -euo pipefail
cd "$(dirname "$0")/../.."
ip=$(python3 -c 'import json; print(json.load(open("results/resources.json"))["instance_ip"])')
python3 - "$ip" <<'PY'
import ipaddress, sys
ipaddress.ip_address(sys.argv[1])
PY
mkdir -p "$RUNNER_TEMP/kernelport-ssh"
chmod 700 "$RUNNER_TEMP/kernelport-ssh"
key="$RUNNER_TEMP/kernelport-ssh/key"
known="$RUNNER_TEMP/kernelport-ssh/known_hosts"
printf '%s\n' "$LAMBDA_SSH_PRIVATE_KEY" > "$key"
chmod 600 "$key"
unset LAMBDA_SSH_PRIVATE_KEY
remote_dir="kernelport-${GITHUB_RUN_ID}-${GITHUB_RUN_ATTEMPT}"
[[ "$remote_dir" =~ ^kernelport-[0-9]+-[0-9]+$ ]]
# TOFU for this freshly provisioned IP, then pin that host key for this run.
ssh_args=(-i "$key" -o BatchMode=yes -o IdentitiesOnly=yes -o ConnectTimeout=10 -o ServerAliveInterval=15 -o ServerAliveCountMax=4 -o UserKnownHostsFile="$known" -o StrictHostKeyChecking=accept-new)
connected=false
for attempt in $(seq 1 60); do
  if ssh "${ssh_args[@]}" "ubuntu@$ip" true; then connected=true; break; fi
  sleep 10
done
[[ "$connected" == true ]] || { echo 'SSH readiness timed out'; exit 1; }
ssh_args=("${ssh_args[@]/StrictHostKeyChecking=accept-new/StrictHostKeyChecking=yes}")
ssh "${ssh_args[@]}" "ubuntu@$ip" "mkdir -p '$remote_dir' && chmod 700 '$remote_dir'"
git archive "$GITHUB_SHA" | ssh "${ssh_args[@]}" "ubuntu@$ip" "tar -x -C '$remote_dir'"
scp -i "$key" -o BatchMode=yes -o UserKnownHostsFile="$known" -o StrictHostKeyChecking=yes results/resources.json images.json "ubuntu@$ip:$remote_dir/"
printf '%s' "$GHCR_TOKEN" | ssh "${ssh_args[@]}" "ubuntu@$ip" "docker login ghcr.io -u '$GHCR_USER' --password-stdin >/dev/null"
unset GHCR_TOKEN
collect() {
  code=$?
  trap - EXIT
  ssh "${ssh_args[@]}" "ubuntu@$ip" "tar -cz -C '$remote_dir' results" > results/host-results.tar.gz || true
  ssh "${ssh_args[@]}" "ubuntu@$ip" "docker logout ghcr.io >/dev/null 2>&1; rm -f '$remote_dir/worker.env' '$remote_dir/prompt.wav'" || true
  rm -f "$key"
  exit "$code"
}
trap collect EXIT
if [[ "$VALIDATION_MODE" == luxtts ]]; then
  printf 'HUGGINGFACE_HUB_TOKEN=%s\n' "$HUGGINGFACE_HUB_TOKEN" | ssh "${ssh_args[@]}" "ubuntu@$ip" "umask 077; cat > '$remote_dir/worker.env'"
  printf '%s' "$LUXTTS_PROMPT_WAV_BASE64" | base64 --decode | ssh "${ssh_args[@]}" "ubuntu@$ip" "umask 077; cat > '$remote_dir/prompt.wav'"
fi
unset HUGGINGFACE_HUB_TOKEN LUXTTS_PROMPT_WAV_BASE64
ssh "${ssh_args[@]}" "ubuntu@$ip" "cd '$remote_dir' && bash scripts/lambda/run-stack.sh"

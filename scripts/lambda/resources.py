"""Lambda lifecycle with durable resource IDs and independently owned cleanup.

API contract: https://docs.lambda.ai/public-cloud/cloud-api/
Only GETs and idempotent cleanup operations may be retried. Never retry launches.
"""
from __future__ import annotations

import argparse
import base64
import json
import os
from pathlib import Path
import re
import time
from urllib import error, request


class APIError(RuntimeError):
    def __init__(self, status, message):
        super().__init__(f"Lambda HTTP {status}: {message}")
        self.status = status


class API:
    def __init__(self):
        self.base = "https://cloud.lambdalabs.com/api/v1"
        token = os.environ["LAMBDA_CLOUD_API_KEY"]
        self.auth = "Basic " + base64.b64encode(f"{token}:".encode()).decode()

    def call(self, method, path, body=None):
        req = request.Request(
            self.base + path,
            data=json.dumps(body).encode() if body is not None else None,
            headers={"Authorization": self.auth, "Content-Type": "application/json"},
            method=method,
        )
        try:
            with request.urlopen(req, timeout=30) as response:
                raw = response.read()
        except error.HTTPError as exc:
            # Do not echo request bodies or credentials in diagnostics.
            raise APIError(exc.code, f"{method} {path} failed") from exc
        data = json.loads(raw) if raw else {}
        if data.get("error") or data.get("errors"):
            raise APIError(200, f"{method} {path} returned an error envelope")
        return data.get("data")


def identifier(value):
    value = value.strip()
    # IDs are opaque. Do not invent an fs_ prefix requirement.
    if not re.fullmatch(r"[A-Za-z0-9_-]+", value):
        raise ValueError("invalid resource ID")
    return value


def save(path, state):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(state, indent=2) + "\n")
    tmp.replace(path)


def filesystem(api, state, state_path, region, existing, name):
    if existing.strip():
        fs_id = identifier(existing)
        matches = [f for f in api.call("GET", "/filesystems") if f["id"] == fs_id]
        if len(matches) != 1:
            raise ValueError("filesystem ID not found in this account")
        fs = matches[0]
        if fs["region"]["name"] != region:
            raise ValueError("filesystem and instance regions must match")
        owned = False
    else:
        # Lambda filesystems grow with usage; size_gb is not a quota input.
        fs = api.call("POST", "/filesystems", {"name": name, "region": region})
        owned = True
    state.update(filesystem_id=identifier(fs["id"]), filesystem_name=fs.get("name", name),
                 filesystem_created=owned)
    save(state_path, state)


def launch(api, state, state_path, region, instance_type, ssh_key, name):
    result = api.call("POST", "/instance-operations/launch", {
        "region_name": region, "instance_type_name": instance_type,
        "ssh_key_names": [ssh_key], "file_system_names": [state["filesystem_name"]],
        "quantity": 1, "name": name,
    })
    ids = result["instance_ids"]
    if len(ids) != 1:
        raise ValueError("expected exactly one instance; inspect Lambda console")
    state["instance_id"] = identifier(ids[0])
    save(state_path, state)


def poll_active(api, state, state_path, attempts=60, delay=10):
    for _ in range(attempts):
        info = api.call("GET", f"/instances/{identifier(state['instance_id'])}")
        if info["status"] == "active" and info.get("ip"):
            state.update(instance_ip=info["ip"], instance=info)
            save(state_path, state)
            return
        if info["status"] in ("terminated", "unhealthy"):
            raise RuntimeError(f"instance entered {info['status']}")
        time.sleep(delay)
    raise TimeoutError("instance did not become active")


def terminate(api, instance_id, attempts=60, delay=10):
    path = f"/instances/{identifier(instance_id)}"
    try:
        info = api.call("GET", path)
    except APIError as exc:
        if exc.status == 404:
            return
        raise
    if info["status"] == "terminated":
        return
    if info["status"] != "terminating":
        api.call("POST", "/instance-operations/terminate", {"instance_ids": [instance_id]})
    for _ in range(attempts):
        try:
            if api.call("GET", path)["status"] == "terminated":
                return
        except APIError as exc:
            if exc.status == 404:
                return
            raise
        time.sleep(delay)
    raise TimeoutError("termination not confirmed; filesystem retained")


def delete_filesystem(api, fs_id, attempts=30, delay=10):
    fs_id = identifier(fs_id)
    for _ in range(attempts):
        filesystems = api.call("GET", "/filesystems")
        fs = next((f for f in filesystems if f["id"] == fs_id), None)
        if fs is None:
            return
        # Missing attachment metadata is not evidence that deletion is safe.
        if fs.get("is_in_use") is False:
            try:
                api.call("DELETE", f"/filesystems/{fs_id}")
            except APIError as exc:
                if exc.status == 404:
                    return
                if exc.status not in (409, 429, 500, 502, 503, 504):
                    raise
            # Confirm absence on a subsequent list, even after a 2xx response.
        time.sleep(delay)
    raise TimeoutError("filesystem deletion not confirmed; inspect Lambda console")


def cleanup(api, state, state_path, attempts=60, delay=10):
    errors = []
    terminated = not state.get("instance_id")
    if state.get("instance_id"):
        try:
            terminate(api, state["instance_id"], attempts, delay)
            state["instance_terminated"] = True
            terminated = True
        except Exception as exc:
            errors.append(str(exc))
    # Reused filesystems never suppress instance termination, and are never deleted.
    if state.get("filesystem_created") and state.get("filesystem_id") and terminated:
        try:
            delete_filesystem(api, state["filesystem_id"], attempts, delay)
            state["filesystem_deleted"] = True
        except Exception as exc:
            errors.append(str(exc))
    state["cleanup_errors"] = errors
    save(state_path, state)
    if errors:
        raise RuntimeError("; ".join(errors))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("action", choices=["provision", "cleanup", "teardown"])
    parser.add_argument("--state", default="results/resources.json")
    args = parser.parse_args()
    api = API()
    state = json.loads(Path(args.state).read_text()) if Path(args.state).exists() else {}
    if args.action == "provision":
        if state.get("instance_id") or state.get("filesystem_id"):
            raise ValueError("state already contains resources; use a fresh run directory")
        run_name = f"kernelport-{os.environ['GITHUB_RUN_ID']}-{os.environ['GITHUB_RUN_ATTEMPT']}"
        state.update(commit=os.environ["GITHUB_SHA"], run_name=run_name)
        save(args.state, state)
        region = os.environ["REGION_NAME"]
        filesystem(api, state, args.state, region, os.getenv("FILESYSTEM_ID", ""), run_name)
        launch(api, state, args.state, region, os.environ["INSTANCE_TYPE_NAME"],
               os.environ["SSH_KEY_NAME"], run_name)
        poll_active(api, state, args.state)
    else:
        if args.action == "teardown":
            state = {"instance_id": identifier(os.environ["INSTANCE_ID"]),
                     "filesystem_id": os.getenv("FILESYSTEM_ID", "").strip(),
                     "filesystem_created": os.getenv("DELETE_FILESYSTEM") == "true"}
            if state["filesystem_created"] and not state["filesystem_id"]:
                raise ValueError("filesystem ID required when deletion is selected")
        if state.get("filesystem_created") and state.get("filesystem_id"):
            state["filesystem_id"] = identifier(state["filesystem_id"])
        save(args.state, state)
        cleanup(api, state, args.state)


if __name__ == "__main__":
    main()

import importlib.util
import io
import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
from urllib.error import HTTPError

spec = importlib.util.spec_from_file_location("resources", Path(__file__).resolve().parents[1] / "scripts/lambda/resources.py")
r = importlib.util.module_from_spec(spec)
spec.loader.exec_module(r)


class FakeAPI:
    def __init__(self, replies):
        self.replies = iter(replies)
        self.calls = []

    def call(self, method, path, body=None):
        self.calls.append((method, path, body))
        reply = next(self.replies)
        if isinstance(reply, Exception):
            raise reply
        return reply


class ResourceTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.path = Path(self.tmp.name) / "state.json"

    def test_reused_filesystem_does_not_suppress_instance_cleanup(self):
        api = FakeAPI([{"status": "active"}, {}, {"status": "terminated"}])
        state = {"instance_id": "abc123", "filesystem_id": "reused", "filesystem_created": False}
        r.cleanup(api, state, self.path, attempts=2, delay=0)
        self.assertTrue(state["instance_terminated"])
        self.assertFalse(any(m == "DELETE" for m, _, _ in api.calls))

    def test_waits_for_termination_and_detachment_then_confirms_deletion(self):
        api = FakeAPI([{"status": "active"}, {}, {"status": "terminating"},
                       {"status": "terminated"}, [{"id": "fs_a", "is_in_use": True}],
                       [{"id": "fs_a", "is_in_use": False}], {}, []])
        state = {"instance_id": "instance", "filesystem_id": "fs_a", "filesystem_created": True}
        r.cleanup(api, state, self.path, attempts=4, delay=0)
        self.assertTrue(json.loads(self.path.read_text())["filesystem_deleted"])

    def test_termination_timeout_retains_filesystem_and_records_failure(self):
        api = FakeAPI([{"status": "terminating"}, {"status": "terminating"}])
        state = {"instance_id": "instance", "filesystem_id": "fs_a", "filesystem_created": True}
        with self.assertRaises(RuntimeError):
            r.cleanup(api, state, self.path, attempts=1, delay=0)
        self.assertTrue(state["cleanup_errors"])
        self.assertFalse(any(m == "DELETE" for m, _, _ in api.calls))

    def test_already_deleted_resources_are_idempotent(self):
        api = FakeAPI([r.APIError(404, "missing"), []])
        state = {"instance_id": "instance", "filesystem_id": "fs_a", "filesystem_created": True}
        r.cleanup(api, state, self.path, attempts=1, delay=0)
        self.assertTrue(state["filesystem_deleted"])

    def test_http_failure_is_not_success_or_retried(self):
        with patch.dict(os.environ, {"LAMBDA_CLOUD_API_KEY": "test"}):
            api = r.API()
        error = HTTPError("https://example.invalid", 403, "Forbidden", {}, io.BytesIO(b"secret"))
        with patch.object(r.request, "urlopen", side_effect=error) as call:
            with self.assertRaises(r.APIError) as caught:
                api.call("POST", "/instance-operations/launch", {})
            self.assertEqual(call.call_count, 1)
            self.assertNotIn("secret", str(caught.exception))

    def test_delete_error_does_not_claim_success(self):
        api = FakeAPI([[{"id": "fs_a", "is_in_use": False}], r.APIError(403, "denied")])
        state = {"filesystem_id": "fs_a", "filesystem_created": True}
        with self.assertRaises(RuntimeError):
            r.cleanup(api, state, self.path, attempts=1, delay=0)
        self.assertNotIn("filesystem_deleted", state)

    def test_region_mismatch_fails_before_launch(self):
        api = FakeAPI([[{"id": "abc", "region": {"name": "wrong"}}]])
        with self.assertRaises(ValueError):
            r.filesystem(api, {}, self.path, "us-west-1", " abc ", "test")

    def test_create_and_launch_payloads_and_durable_ids(self):
        api = FakeAPI([{"id": "012345abcdef", "name": "test"}, {"instance_ids": ["instance"]}])
        state = {}
        r.filesystem(api, state, self.path, "us-west-1", "", "test")
        self.assertEqual(json.loads(self.path.read_text())["filesystem_id"], "012345abcdef")
        r.launch(api, state, self.path, "us-west-1", "gpu", "ssh-key", "test")
        self.assertEqual(api.calls[0][2], {"name": "test", "region": "us-west-1"})
        self.assertEqual(api.calls[1][2]["file_system_names"], ["test"])
        self.assertEqual(json.loads(self.path.read_text())["instance_id"], "instance")

    def test_malformed_ids_fail_closed(self):
        for value in ("", "../foo", "fs_a\nfs_b", "$(cmd)"):
            with self.assertRaises(ValueError):
                r.identifier(value)
        self.assertEqual(r.identifier("  abc-def123  "), "abc-def123")


if __name__ == "__main__":
    unittest.main()

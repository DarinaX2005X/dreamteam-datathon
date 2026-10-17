from __future__ import annotations

import hashlib
import importlib.util
import json
import os
import sys
import tempfile
import threading
import unittest
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
SPEC = importlib.util.spec_from_file_location("fetch_models", ROOT / "scripts" / "fetch_models.py")
fetch_models = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
sys.modules[SPEC.name] = fetch_models
SPEC.loader.exec_module(fetch_models)


class RangeHandler(BaseHTTPRequestHandler):
    payload = b""
    wrong_range = False
    requests_seen: list[str | None] = []

    def do_GET(self):
        value = self.headers.get("Range")
        type(self).requests_seen.append(value)
        if value is None or not value.startswith("bytes="):
            self.send_response(400)
            self.end_headers()
            return
        start_text, end_text = value[6:].split("-", 1)
        start, end = int(start_text), int(end_text)
        body = type(self).payload[start : end + 1]
        reported_start = start + 1 if type(self).wrong_range else start
        self.send_response(206)
        self.send_header("Content-Length", str(len(body)))
        self.send_header(
            "Content-Range", f"bytes {reported_start}-{end}/{len(type(self).payload)}"
        )
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, format, *args):
        return


class FetchModelsTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        RangeHandler.payload = (b"range-safe-model-data-" * 8192) + b"end"
        RangeHandler.wrong_range = False
        RangeHandler.requests_seen = []
        self.server = ThreadingHTTPServer(("127.0.0.1", 0), RangeHandler)
        self.thread = threading.Thread(target=self.server.serve_forever, daemon=True)
        self.thread.start()
        self.url = f"http://127.0.0.1:{self.server.server_port}/model.bin"

    def tearDown(self):
        self.server.shutdown()
        self.server.server_close()
        self.thread.join()
        self.temporary.cleanup()

    def pin(self, *, digest=None, path="fixture/model.bin"):
        payload = RangeHandler.payload
        return fetch_models.FilePin(
            "fixture",
            path,
            self.url,
            digest or hashlib.sha256(payload).hexdigest(),
            len(payload),
        )

    def test_resumes_existing_partial_range_and_publishes_atomically(self):
        pin = self.pin()
        target = self.root / pin.path
        target.parent.mkdir(parents=True)
        part_dir = target.parent / f".{target.name}.{pin.sha256[:16]}.parts"
        part_dir.mkdir()
        resumed = 12_345
        (part_dir / "000.part").write_bytes(RangeHandler.payload[:resumed])

        result = fetch_models.download_file(
            pin, self.root, workers=1, timeout=5, retries=1, allow_insecure_http=True
        )

        self.assertEqual(result, "downloaded")
        self.assertEqual(target.read_bytes(), RangeHandler.payload)
        self.assertIn(f"bytes={resumed}-{len(RangeHandler.payload) - 1}", RangeHandler.requests_seen)

    def test_wrong_content_range_keeps_existing_target(self):
        pin = self.pin()
        target = self.root / pin.path
        target.parent.mkdir(parents=True)
        target.write_bytes(b"existing-unverified-file")
        RangeHandler.wrong_range = True

        with self.assertRaises(fetch_models.FetchError):
            fetch_models.download_file(
                pin, self.root, workers=1, timeout=5, retries=0, allow_insecure_http=True
            )

        self.assertEqual(target.read_bytes(), b"existing-unverified-file")

    def test_hash_mismatch_does_not_replace_existing_target(self):
        pin = self.pin(digest="0" * 64)
        target = self.root / pin.path
        target.parent.mkdir(parents=True)
        target.write_bytes(b"keep-me")

        with self.assertRaises(fetch_models.FetchError):
            fetch_models.download_file(
                pin, self.root, workers=2, timeout=5, retries=0, allow_insecure_http=True
            )

        self.assertEqual(target.read_bytes(), b"keep-me")

    def test_valid_existing_file_avoids_network(self):
        pin = self.pin()
        target = self.root / pin.path
        target.parent.mkdir(parents=True)
        target.write_bytes(RangeHandler.payload)
        self.assertEqual(fetch_models.download_file(pin, self.root), "verified")
        self.assertEqual(RangeHandler.requests_seen, [])

    def test_rejects_traversal_and_symlink_target(self):
        with self.assertRaises(fetch_models.FetchError):
            fetch_models.safe_target(self.root, "../escape.bin")
        real = self.root / "real"
        real.mkdir()
        link = self.root / "linked"
        try:
            os.symlink(real, link, target_is_directory=True)
        except OSError:
            self.skipTest("symlinks require additional privileges on this platform")
        with self.assertRaises(fetch_models.FetchError):
            fetch_models.safe_target(self.root, "linked/model.bin")

    def test_manifest_rejects_http_rejected_model_and_outside_local_dir(self):
        base = {
            "schema_version": 1,
            "models": [
                {
                    "id": "bad",
                    "local_dir": "safe",
                    "status": "rejected",
                    "files": [],
                }
            ],
        }
        manifest = self.root / "manifest.json"
        manifest.write_text(json.dumps(base), encoding="utf-8")
        with self.assertRaises(fetch_models.FetchError):
            fetch_models.load_pins(manifest, ["bad"])

        base["models"][0].pop("status")
        base["models"][0]["files"] = [
            {
                "path": "other/file.bin",
                "url": self.url,
                "sha256": "0" * 64,
                "size_bytes": 1,
            }
        ]
        manifest.write_text(json.dumps(base), encoding="utf-8")
        with self.assertRaises(fetch_models.FetchError):
            fetch_models.load_pins(manifest, ["bad"])


if __name__ == "__main__":
    unittest.main()

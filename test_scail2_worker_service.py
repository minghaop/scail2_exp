from __future__ import annotations

import io
import json
import tempfile
import unittest
from email.message import Message
from pathlib import Path
from unittest import mock

import scail2_worker_service as worker


def valid_submission_message() -> dict[str, object]:
    paths = {
        "reference_image": "reference.png",
        "reference_mask": "reference_mask.png",
        "driving_video": "driving.mp4",
        "driving_mask": "driving_mask.mp4",
    }
    return {
        "handle": "test-handle",
        "workflow": worker.WORKFLOW,
        "params": {**paths, "prompt": "A person walking naturally"},
        "s3": {
            "relative_path_fields": list(worker.PATH_PARAM_FIELDS),
            "downloads": [
                {
                    "key": field_name,
                    "local_file": local_file,
                    "url": f"https://example.invalid/{local_file}",
                }
                for field_name, local_file in paths.items()
            ],
            "uploads": ["https://example.invalid/output.mp4"],
        },
    }


class FakeResponse(io.BytesIO):
    def __init__(self, payload: bytes) -> None:
        super().__init__(payload)
        self.headers = Message()
        self.headers["Content-Type"] = "application/octet-stream"
        self.headers["X-SCAIL2-Prompt-SHA256"] = "prompt-hash"
        self.headers["X-SCAIL2-Cache-Hit"] = "true"

    def __enter__(self) -> "FakeResponse":
        return self

    def __exit__(self, *args: object) -> None:
        self.close()


class SubmissionValidationTest(unittest.TestCase):
    def test_accepts_prompt_and_four_media_downloads(self) -> None:
        submission = worker.validate_submission(valid_submission_message())
        self.assertEqual(submission.params["prompt"], "A person walking naturally")
        self.assertEqual(len(submission.downloads), 4)
        self.assertNotIn("t5_cache", worker.PATH_PARAM_FIELDS)

    def test_rejects_missing_prompt(self) -> None:
        message = valid_submission_message()
        params = dict(message["params"])
        params.pop("prompt")
        message["params"] = params
        with self.assertRaisesRegex(worker.SubmissionError, "params.prompt is required"):
            worker.validate_submission(message)

    def test_rejects_empty_prompt(self) -> None:
        message = valid_submission_message()
        params = dict(message["params"])
        params["prompt"] = "   "
        message["params"] = params
        with self.assertRaisesRegex(
            worker.SubmissionError,
            "params.prompt must be a non-empty string",
        ):
            worker.validate_submission(message)


class T5PrecacheRequestTest(unittest.TestCase):
    def test_posts_prompt_and_writes_cache_artifact(self) -> None:
        artifact = b"test-safetensors"
        captured_request = None

        def open_request(request: object, *, timeout: float) -> FakeResponse:
            nonlocal captured_request
            captured_request = request
            self.assertEqual(timeout, worker.T5_PRECACHE_TIMEOUT_SECONDS)
            return FakeResponse(artifact)

        with tempfile.TemporaryDirectory() as directory:
            with mock.patch.object(worker.urllib.request, "urlopen", open_request):
                output = worker.request_t5_cache(Path(directory), "test prompt")
            self.assertEqual(output.name, worker.T5_CACHE_FILENAME)
            self.assertEqual(output.read_bytes(), artifact)

        self.assertIsNotNone(captured_request)
        self.assertEqual(captured_request.full_url, worker.T5_PRECACHE_URL)
        self.assertEqual(captured_request.get_method(), "POST")
        self.assertEqual(
            json.loads(captured_request.data.decode("utf-8")),
            {"prompt": "test prompt"},
        )


if __name__ == "__main__":
    unittest.main()

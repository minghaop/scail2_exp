"""Dispatcher-compatible WebSocket worker for the SCAIL-2 inference SDK.

The module is intentionally self-contained.  Rank 0 hosts the WebSocket server
and owns a single-job Backend; rank 1 only participates in the distributed SDK
runtime.  Inputs and the output MP4 live under ``/dev/shm/<rank-0-pid>``.
"""

import argparse
import asyncio
import json
import logging
import os
import shutil
import threading
import time
import traceback
import urllib.error
import urllib.request
import uuid
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path, PurePosixPath
from typing import Any


# Deployment and model configuration is deliberately kept in code.  torchrun's
# RANK/WORLD_SIZE environment variables are still consumed by the SDK itself.
LISTEN_HOST = "0.0.0.0"
DEFAULT_PORT = 3000
WORKFLOW = "scail2_video"

CHECKPOINT_DIR = Path("/models")
DIT_CHECKPOINT = Path(
    "/models/derived/"
    "SCAIL-2-lightx2v-r128-dpo-alpha1-full-bf16.safetensors"
)
PROFILE_NAME = "scail2-512p-bf16-v1"
EXPECTED_WORLD_SIZE = 2
OUTPUT_AUDIO_MODE = "driving"

DOWNLOAD_TIMEOUT_SECONDS = 300.0
UPLOAD_TIMEOUT_SECONDS = 900.0
INFERENCE_TIMEOUT_SECONDS = 7200.0
STARTUP_TIMEOUT_SECONDS = 1800.0
SHUTDOWN_TIMEOUT_SECONDS = 600.0
COMPUTING_EVENT_INTERVAL_SECONDS = 30.0

CONTROL_POLL_SECONDS = 1.0
CONTROL_TIMEOUT_SECONDS = 120.0

PATH_PARAM_FIELDS = (
    "reference_image",
    "reference_mask",
    "driving_video",
    "driving_mask",
)
REQUIRED_PARAM_FIELDS = (*PATH_PARAM_FIELDS, "prompt")

def unix_milliseconds() -> int:
    return int(time.time() * 1000)


def json_safe(value: Any) -> Any:
    """Convert SDK result metadata to values accepted by json.dumps."""

    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, Mapping):
        return {str(key): json_safe(item) for key, item in value.items()}
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        return [json_safe(item) for item in value]
    return str(value)


class SubmissionError(ValueError):
    """The Dispatcher request cannot be accepted as an inference task."""


class TransferError(RuntimeError):
    """An input download or output upload failed."""


class WorkerBusyError(RuntimeError):
    pass


class WorkerStoppingError(RuntimeError):
    pass


@dataclass(frozen=True, slots=True)
class DownloadSpec:
    key: str
    local_file: PurePosixPath
    url: str


@dataclass(frozen=True, slots=True)
class ValidatedSubmission:
    handle: str
    params: dict[str, Any]
    downloads: tuple[DownloadSpec, ...]
    upload_url: str


def _required_nonempty_string(value: Any, name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise SubmissionError(f"{name} must be a non-empty string")
    return value


def _relative_path(value: Any, name: str) -> PurePosixPath:
    raw = _required_nonempty_string(value, name)
    path = PurePosixPath(raw)
    if path.is_absolute():
        raise SubmissionError(f"{name} must be relative")
    if path == PurePosixPath(".") or ".." in path.parts:
        raise SubmissionError(f"{name} contains an invalid path")
    return path


def validate_submission(message: Mapping[str, Any]) -> ValidatedSubmission:
    handle = _required_nonempty_string(message.get("handle"), "handle")
    workflow = _required_nonempty_string(message.get("workflow"), "workflow")
    if workflow != WORKFLOW:
        raise SubmissionError(f"Unsupported workflow: {workflow}")

    params_value = message.get("params", {})
    if not isinstance(params_value, dict):
        raise SubmissionError("params must be an object")
    params = dict(params_value)
    for field_name in REQUIRED_PARAM_FIELDS:
        if field_name not in params:
            raise SubmissionError(f"params.{field_name} is required")
    prompt = _required_nonempty_string(params["prompt"], "params.prompt").strip()
    if len(prompt) > 16384:
        raise SubmissionError("params.prompt is too long")
    params["prompt"] = prompt
    for field_name in PATH_PARAM_FIELDS:
        _relative_path(params[field_name], f"params.{field_name}")

    s3 = message.get("s3")
    if not isinstance(s3, dict):
        raise SubmissionError("s3 must be an object")

    relative_fields = s3.get("relative_path_fields")
    if not isinstance(relative_fields, list) or any(
        not isinstance(item, str) for item in relative_fields
    ):
        raise SubmissionError("s3.relative_path_fields must be a string array")
    if set(relative_fields) != set(PATH_PARAM_FIELDS) or len(relative_fields) != len(
        PATH_PARAM_FIELDS
    ):
        expected = ", ".join(PATH_PARAM_FIELDS)
        raise SubmissionError(
            f"s3.relative_path_fields must contain exactly: {expected}"
        )

    downloads_value = s3.get("downloads")
    if not isinstance(downloads_value, list) or len(downloads_value) != len(
        PATH_PARAM_FIELDS
    ):
        raise SubmissionError("s3.downloads must contain exactly four inputs")
    downloads: list[DownloadSpec] = []
    destinations: set[PurePosixPath] = set()
    for index, item in enumerate(downloads_value):
        if not isinstance(item, dict):
            raise SubmissionError(f"s3.downloads[{index}] must be an object")
        local_file = _relative_path(
            item.get("local_file"), f"s3.downloads[{index}].local_file"
        )
        if local_file in destinations:
            raise SubmissionError(f"Duplicate download destination: {local_file}")
        destinations.add(local_file)
        downloads.append(
            DownloadSpec(
                key=str(item.get("key", "")),
                local_file=local_file,
                url=_required_nonempty_string(
                    item.get("url"), f"s3.downloads[{index}].url"
                ),
            )
        )

    for field_name in PATH_PARAM_FIELDS:
        parameter_path = _relative_path(params[field_name], f"params.{field_name}")
        if parameter_path not in destinations:
            raise SubmissionError(
                f"params.{field_name} does not reference a downloaded local_file"
            )

    uploads = s3.get("uploads")
    if not isinstance(uploads, list) or len(uploads) != 1:
        raise SubmissionError("s3.uploads must contain exactly one URL")
    upload_url = _required_nonempty_string(uploads[0], "s3.uploads[0]")

    return ValidatedSubmission(
        handle=handle,
        params=params,
        downloads=tuple(downloads),
        upload_url=upload_url,
    )


def clear_work_directory(work_dir: Path) -> None:
    shutil.rmtree(work_dir, ignore_errors=True)
    work_dir.mkdir(parents=True, exist_ok=True)


def _transfer_failure(kind: str, local_name: str, error: BaseException) -> TransferError:
    if isinstance(error, urllib.error.HTTPError):
        detail = f"HTTP {error.code}"
    elif isinstance(error, urllib.error.URLError):
        detail = f"request failed ({error.reason})"
    else:
        detail = type(error).__name__
    return TransferError(f"Failed to {kind} {local_name}: {detail}")


def download_inputs(work_dir: Path, downloads: Sequence[DownloadSpec]) -> None:
    for download in downloads:
        destination = work_dir.joinpath(*download.local_file.parts)
        destination.parent.mkdir(parents=True, exist_ok=True)
        try:
            with urllib.request.urlopen(
                download.url, timeout=DOWNLOAD_TIMEOUT_SECONDS
            ) as response:
                data = response.read()
            if not data:
                raise TransferError(f"Downloaded input is empty: {download.local_file}")
            destination.write_bytes(data)
        except TransferError:
            raise
        except Exception as error:
            raise _transfer_failure("download", str(download.local_file), error) from error


def upload_output(output_path: Path, upload_url: str) -> None:
    if not output_path.is_file() or output_path.stat().st_size == 0:
        raise TransferError(f"Inference output is missing or empty: {output_path.name}")
    try:
        request = urllib.request.Request(
            upload_url,
            data=output_path.read_bytes(),
            headers={"Content-Type": "video/mp4"},
            method="PUT",
        )
        with urllib.request.urlopen(
            request, timeout=UPLOAD_TIMEOUT_SECONDS
        ) as response:
            response.read()
    except Exception as error:
        raise _transfer_failure("upload", output_path.name, error) from error


@dataclass(slots=True)
class TaskRecord:
    job_id: str
    loop: asyncio.AbstractEventLoop
    completion: asyncio.Event
    queued_ms: int = field(default_factory=unix_milliseconds)
    submitted_monotonic: float = field(default_factory=time.monotonic)
    compute_start_ms: int | None = None
    compute_end_ms: int | None = None
    sdk_result: Any = None
    error_type: str | None = None
    error: str | None = None
    status: str = "queued"

    def notify_completion(self) -> None:
        try:
            self.loop.call_soon_threadsafe(self.completion.set)
        except RuntimeError:
            # The WebSocket loop is already gone; the runtime must still be
            # allowed to finish and shut down normally.
            pass


class DispatcherJobBackend:
    """A blocking, single-job SDK Backend with no waiting queue.

    ``reserve`` covers input preparation, inference and result upload.  This is
    necessary because all tasks reuse the same PID-specific work directory.
    """

    def __init__(self) -> None:
        self._condition = threading.Condition()
        self._stopping = False
        self._busy_job_id: str | None = None
        self._pending_job: Any = None
        self._active_runtime_job_id: str | None = None
        self._records: dict[str, TaskRecord] = {}

    def reserve(self, job_id: str) -> None:
        with self._condition:
            if self._stopping:
                raise WorkerStoppingError("The worker is stopping")
            if self._busy_job_id is not None:
                raise WorkerBusyError("Worker is busy")
            self._busy_job_id = job_id

    def cancel_reservation(self, job_id: str) -> None:
        with self._condition:
            if self._busy_job_id == job_id and job_id not in self._records:
                self._busy_job_id = None

    def submit_reserved(self, job: Any, record: TaskRecord) -> None:
        with self._condition:
            if self._stopping:
                raise WorkerStoppingError("The worker is stopping")
            if self._busy_job_id != record.job_id:
                raise RuntimeError("Task submission does not own the worker reservation")
            if self._pending_job is not None or self._active_runtime_job_id is not None:
                raise RuntimeError("Single-job Backend received concurrent work")
            self._records[record.job_id] = record
            self._pending_job = job
            self._condition.notify_all()

    def acquire(self) -> Any | None:
        with self._condition:
            while self._pending_job is None and not self._stopping:
                self._condition.wait()
            if self._pending_job is None:
                return None
            job = self._pending_job
            self._pending_job = None
            self._active_runtime_job_id = job.job_id
            return job

    def mark_running(self, job: Any) -> None:
        with self._condition:
            record = self._records[job.job_id]
            record.status = "running"
            record.compute_start_ms = unix_milliseconds()

    def mark_success(self, job: Any, result: Any) -> None:
        with self._condition:
            record = self._records[job.job_id]
            record.status = "sdk_succeeded"
            record.compute_end_ms = unix_milliseconds()
            record.sdk_result = result
            self._active_runtime_job_id = None
            record.notify_completion()

    def mark_failed(
        self,
        job: Any,
        error: BaseException,
        traceback_text: str,
    ) -> None:
        logging.error(
            "SCAIL2_JOB_FAILED job_id=%s error_type=%s error=%s\n%s",
            job.job_id,
            type(error).__name__,
            error,
            traceback_text,
        )
        with self._condition:
            record = self._records[job.job_id]
            record.status = "sdk_failed"
            record.compute_end_ms = unix_milliseconds()
            record.error_type = type(error).__name__
            record.error = str(error)
            self._active_runtime_job_id = None
            record.notify_completion()

    def finish_external_processing(self, job_id: str) -> None:
        with self._condition:
            self._records.pop(job_id, None)
            if self._busy_job_id == job_id:
                self._busy_job_id = None

    def request_stop(self) -> None:
        with self._condition:
            if self._stopping:
                return
            self._stopping = True
            if self._pending_job is not None:
                job = self._pending_job
                self._pending_job = None
                record = self._records[job.job_id]
                record.status = "sdk_failed"
                record.compute_end_ms = unix_milliseconds()
                record.error_type = "WorkerStoppingError"
                record.error = "The worker stopped before inference started"
                record.notify_completion()
            self._condition.notify_all()


class DispatcherWorkerService:
    def __init__(
        self,
        backend: DispatcherJobBackend,
        inference_job_class: type[Any],
        *,
        sdk_version: str,
    ) -> None:
        self._backend = backend
        self._inference_job_class = inference_job_class
        self._sdk_version = sdk_version
        self._client_id = uuid.uuid4().hex
        self._work_dir = Path("/dev/shm") / str(os.getpid())
        self._background_tasks: set[asyncio.Task[Any]] = set()

    def _track_background(self, coroutine: Any) -> None:
        task = asyncio.create_task(coroutine)
        self._background_tasks.add(task)
        task.add_done_callback(self._background_tasks.discard)

    async def handle_connection(
        self,
        websocket: Any,
        disconnect_error: type[BaseException],
    ) -> None:
        await websocket.accept()
        connected = False
        try:
            while True:
                text = await websocket.receive_text()
                try:
                    message = json.loads(text)
                except json.JSONDecodeError:
                    await websocket.send_json(
                        {"type": "error", "message": "Invalid JSON message"}
                    )
                    continue
                if not isinstance(message, dict):
                    await websocket.send_json(
                        {"type": "error", "message": "Message must be a JSON object"}
                    )
                    continue

                action = message.get("action")
                if action == "connect":
                    await websocket.send_json(
                        {
                            "type": "return",
                            "action": "connect",
                            "clientId": self._client_id,
                            "workflows": [WORKFLOW],
                        }
                    )
                    connected = True
                elif action == "submitTask":
                    if not connected:
                        await websocket.send_json(
                            {
                                "type": "error",
                                "message": "connect must complete before submitTask",
                            }
                        )
                        continue
                    await self._handle_submission(websocket, message)
                else:
                    await websocket.send_json(
                        {"type": "error", "message": "Unsupported action"}
                    )
        except disconnect_error:
            return
        except asyncio.CancelledError:
            raise
        except Exception:
            logging.exception("WEBSOCKET_CONNECTION_FAILED")

    async def _send_rejection(
        self,
        websocket: Any,
        *,
        handle: str,
        job_id: str,
        cause: str,
    ) -> None:
        await websocket.send_json(
            {
                "type": "return",
                "action": "submitTask",
                "handle": handle,
                "id": job_id,
                "state": "failed",
                "cause": cause,
            }
        )

    async def _handle_submission(
        self,
        websocket: Any,
        message: Mapping[str, Any],
    ) -> None:
        job_id = uuid.uuid4().hex
        raw_handle = message.get("handle")
        if not isinstance(raw_handle, str) or not raw_handle:
            await websocket.send_json(
                {"type": "error", "message": "submitTask requires a non-empty handle"}
            )
            return

        try:
            submission = validate_submission(message)
        except SubmissionError as error:
            await self._send_rejection(
                websocket,
                handle=raw_handle,
                job_id=job_id,
                cause=str(error),
            )
            return

        try:
            self._backend.reserve(job_id)
        except (WorkerBusyError, WorkerStoppingError) as error:
            await self._send_rejection(
                websocket,
                handle=submission.handle,
                job_id=job_id,
                cause=str(error),
            )
            return

        record: TaskRecord | None = None
        submitted = False
        try:
            await asyncio.to_thread(clear_work_directory, self._work_dir)
            await asyncio.to_thread(
                download_inputs, self._work_dir, submission.downloads
            )
            job = self._build_job(job_id, submission)
            job.validate(check_paths=True)
            loop = asyncio.get_running_loop()
            record = TaskRecord(
                job_id=job_id,
                loop=loop,
                completion=asyncio.Event(),
            )
            self._backend.submit_reserved(job, record)
            submitted = True
        except asyncio.CancelledError:
            self._backend.cancel_reservation(job_id)
            raise
        except Exception as error:
            self._backend.cancel_reservation(job_id)
            logging.warning(
                "TASK_REJECTED job_id=%s error_type=%s error=%s",
                job_id,
                type(error).__name__,
                error,
            )
            await self._send_rejection(
                websocket,
                handle=submission.handle,
                job_id=job_id,
                cause=f"{type(error).__name__}: {error}",
            )
            return

        assert record is not None and submitted
        try:
            # This acknowledgement is deliberately sent before inspecting or
            # forwarding any state already produced by the runtime thread.
            await websocket.send_json(
                {
                    "type": "return",
                    "action": "submitTask",
                    "handle": submission.handle,
                    "id": job_id,
                    "verb": f"/SCAIL2/{WORKFLOW}",
                }
            )
        except asyncio.CancelledError:
            self._track_background(self._finish_disconnected_task(record))
            raise
        except Exception:
            self._track_background(self._finish_disconnected_task(record))
            raise

        try:
            await self._wait_for_sdk(record, websocket)
            await self._complete_task(websocket, record, submission.upload_url)
        except asyncio.CancelledError:
            if not record.completion.is_set():
                self._track_background(self._finish_disconnected_task(record))
            else:
                self._backend.finish_external_processing(record.job_id)
            raise
        except Exception:
            if not record.completion.is_set():
                self._track_background(self._finish_disconnected_task(record))
            else:
                self._backend.finish_external_processing(record.job_id)
            raise

    def _build_job(
        self,
        job_id: str,
        submission: ValidatedSubmission,
    ) -> Any:
        params = submission.params

        def local_path(field_name: str) -> Path:
            relative = _relative_path(params[field_name], f"params.{field_name}")
            return self._work_dir.joinpath(*relative.parts)

        reference_mask = local_path("reference_mask")
        if reference_mask.suffix.lower() != ".png":
            raise SubmissionError("reference_mask must be a lossless PNG file")

        return self._inference_job_class(
            job_id=job_id,
            reference_image=local_path("reference_image"),
            reference_mask=reference_mask,
            driving_video=local_path("driving_video"),
            driving_mask=local_path("driving_mask"),
            prompt=params["prompt"],
            output_path=self._work_dir / f"{job_id}.mp4",
            overwrite=False,
            metadata={"source": "dispatcher", "handle": submission.handle},
        )

    async def _wait_for_sdk(self, record: TaskRecord, websocket: Any | None) -> None:
        if websocket is not None and not record.completion.is_set():
            await self._send_computing(websocket, record)

        while not record.completion.is_set():
            elapsed = time.monotonic() - record.submitted_monotonic
            remaining = INFERENCE_TIMEOUT_SECONDS - elapsed
            if remaining <= 0:
                await self._terminate_timed_out_worker(websocket, record)
            try:
                await asyncio.wait_for(
                    record.completion.wait(),
                    timeout=min(COMPUTING_EVENT_INTERVAL_SECONDS, remaining),
                )
            except asyncio.TimeoutError:
                if websocket is not None:
                    await self._send_computing(websocket, record)

    async def _send_computing(self, websocket: Any, record: TaskRecord) -> None:
        step = "inference" if record.status == "running" else "queued"
        await websocket.send_json(
            {
                "type": "event",
                "data": {
                    "event": "computing",
                    "timestamp": unix_milliseconds(),
                    "task_id": record.job_id,
                    "state": "running",
                    "step": step,
                },
            }
        )

    async def _complete_task(
        self,
        websocket: Any,
        record: TaskRecord,
        upload_url: str,
    ) -> None:
        try:
            if record.status != "sdk_succeeded":
                await self._send_completed_failed(
                    websocket,
                    record,
                    f"{record.error_type or 'InferenceError'}: "
                    f"{record.error or 'Inference failed'}",
                )
                return

            result = record.sdk_result
            sdk_status = getattr(result, "status", None)
            if sdk_status not in {"success", "skipped"}:
                await self._send_completed_failed(
                    websocket,
                    record,
                    f"Unexpected SDK result status: {sdk_status}",
                )
                return

            output_path = Path(result.output_path)
            try:
                await asyncio.to_thread(upload_output, output_path, upload_url)
            except Exception as error:
                await self._send_completed_failed(
                    websocket,
                    record,
                    f"{type(error).__name__}: {error}",
                )
                return

            completed_ms = unix_milliseconds()
            result_payload = (
                result.to_dict() if callable(getattr(result, "to_dict", None)) else {}
            )
            await websocket.send_json(
                {
                    "type": "event",
                    "data": {
                        "event": "completed",
                        "timestamp": completed_ms,
                        "task_id": record.job_id,
                        "state": "succeed",
                        "outputs": {
                            "save": [output_path.name],
                            "result": json_safe(result_payload),
                        },
                        "timestamps": self._timestamps(record, completed_ms),
                        "additional_info": {
                            "sdk_version": self._sdk_version,
                            "sdk_status": sdk_status,
                        },
                    },
                }
            )
        finally:
            self._backend.finish_external_processing(record.job_id)

    async def _send_completed_failed(
        self,
        websocket: Any,
        record: TaskRecord,
        cause: str,
    ) -> None:
        completed_ms = unix_milliseconds()
        await websocket.send_json(
            {
                "type": "event",
                "data": {
                    "event": "completed",
                    "timestamp": completed_ms,
                    "task_id": record.job_id,
                    "state": "failed",
                    "outputs": None,
                    "cause": cause,
                    "timestamps": self._timestamps(record, completed_ms),
                    "additional_info": {},
                },
            }
        )

    @staticmethod
    def _timestamps(record: TaskRecord, completed_ms: int) -> dict[str, int]:
        timestamps = {"queued": record.queued_ms, "completed": completed_ms}
        if record.compute_start_ms is not None:
            timestamps["compute_start"] = record.compute_start_ms
        if record.compute_end_ms is not None:
            timestamps["compute_end"] = record.compute_end_ms
        return timestamps

    async def _finish_disconnected_task(self, record: TaskRecord) -> None:
        try:
            await self._wait_for_sdk(record, None)
        finally:
            self._backend.finish_external_processing(record.job_id)

    async def _terminate_timed_out_worker(
        self,
        websocket: Any | None,
        record: TaskRecord,
    ) -> None:
        logging.critical(
            "SCAIL2_INFERENCE_TIMEOUT job_id=%s timeout_seconds=%s",
            record.job_id,
            INFERENCE_TIMEOUT_SECONDS,
        )
        if websocket is not None:
            try:
                await self._send_completed_failed(
                    websocket,
                    record,
                    f"Inference exceeded {INFERENCE_TIMEOUT_SECONDS:g} seconds",
                )
            except Exception:
                pass
        # A timed-out distributed CUDA/NCCL call cannot be safely cancelled or
        # reused.  Exiting rank 0 makes torchrun stop rank 1; Podman then restarts
        # the complete worker.  os._exit is intentional because a hung runtime
        # thread would prevent a normal Python shutdown.
        os._exit(1)


def create_websocket_app(service: DispatcherWorkerService) -> Any:
    try:
        from fastapi import FastAPI, WebSocket, WebSocketDisconnect
    except ImportError as error:
        raise RuntimeError(
            "The worker service requires fastapi and uvicorn[standard]"
        ) from error

    app = FastAPI(docs_url=None, redoc_url=None, openapi_url=None)

    @app.websocket("/")
    async def dispatcher_socket(websocket: WebSocket) -> None:
        await service.handle_connection(websocket, WebSocketDisconnect)

    return app


def positive_port(value: str) -> int:
    port = int(value)
    if not 1 <= port <= 65535:
        raise argparse.ArgumentTypeError("port must be between 1 and 65535")
    return port


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--port", type=positive_port, default=DEFAULT_PORT)
    # Some torchrun versions pass this argument while newer SDK code generally
    # reads LOCAL_RANK from the environment.  Accept both spellings harmlessly.
    parser.add_argument(
        "--local-rank",
        "--local_rank",
        type=int,
        default=None,
        help=argparse.SUPPRESS,
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(threadName)s %(message)s",
    )

    try:
        import uvicorn
        from scail2_inference import (
            EngineConfig,
            InferenceJob,
            ProductionProfile,
            Scail2DistributedRuntime,
            Scail2InferenceEngine,
            __version__,
        )
    except ImportError as error:
        raise RuntimeError(
            "Install scail2-inference, fastapi and uvicorn[standard] in the container"
        ) from error

    profile = ProductionProfile.from_name(PROFILE_NAME)
    config = EngineConfig(
        checkpoint_dir=CHECKPOINT_DIR,
        scail_checkpoint=DIT_CHECKPOINT,
        profile=profile,
        expected_world_size=EXPECTED_WORLD_SIZE,
        t5_fsdp=True,
        dit_fsdp=True,
        offload_model=False,
        output_audio_mode=OUTPUT_AUDIO_MODE,
    )
    runtime = Scail2DistributedRuntime(
        Scail2InferenceEngine(config),
        control_poll_seconds=CONTROL_POLL_SECONDS,
        control_timeout_seconds=CONTROL_TIMEOUT_SECONDS,
    )

    if not runtime.engine.is_primary:
        runtime.run(None)
        if runtime.failure is not None:
            raise RuntimeError("SCAIL-2 non-primary runtime failed") from runtime.failure
        return

    backend = DispatcherJobBackend()

    def run_runtime() -> None:
        try:
            runtime.run(backend)
        except BaseException:
            logging.error("SCAIL2_RUNTIME_FAILED\n%s", traceback.format_exc())

    runtime_thread = threading.Thread(
        target=run_runtime,
        name="scail2-runtime",
        daemon=False,
    )
    runtime_thread.start()

    if not runtime.wait_until_ready(STARTUP_TIMEOUT_SECONDS):
        backend.request_stop()
        runtime_thread.join(timeout=30.0)
        if runtime.failure is not None:
            raise RuntimeError("SCAIL-2 runtime failed during startup") from runtime.failure
        raise RuntimeError(
            f"SCAIL-2 runtime did not become ready within "
            f"{STARTUP_TIMEOUT_SECONDS:g} seconds"
        )

    service = DispatcherWorkerService(
        backend,
        InferenceJob,
        sdk_version=__version__,
    )
    app = create_websocket_app(service)
    server = uvicorn.Server(
        uvicorn.Config(
            app,
            host=LISTEN_HOST,
            port=args.port,
            workers=1,
            log_level="info",
        )
    )

    def monitor_runtime() -> None:
        runtime.wait_until_stopped()
        server.should_exit = True

    threading.Thread(
        target=monitor_runtime,
        name="scail2-runtime-monitor",
        daemon=True,
    ).start()

    print(
        f"SCAIL2_DISPATCHER_WORKER_READY rank=0 host={LISTEN_HOST} "
        f"port={args.port} workflow={WORKFLOW}",
        flush=True,
    )
    try:
        server.run()
    finally:
        backend.request_stop()
        runtime_thread.join(timeout=SHUTDOWN_TIMEOUT_SECONDS)

    if runtime_thread.is_alive():
        logging.critical(
            "SCAIL-2 runtime did not stop within %s seconds",
            SHUTDOWN_TIMEOUT_SECONDS,
        )
        os._exit(1)
    if runtime.failure is not None:
        raise RuntimeError("SCAIL-2 runtime stopped after a fatal error") from runtime.failure


if __name__ == "__main__":
    main()
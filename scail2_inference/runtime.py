"""Persistent worker loop with a service-owned serial job backend."""

from __future__ import annotations

import sys
import threading
import time
import traceback
from datetime import timedelta
from queue import Empty, Queue
from typing import Any, Protocol, runtime_checkable

from .contracts import EngineState, InferenceJob, InferenceResult
from .engine import Scail2InferenceEngine


@runtime_checkable
class JobBackend(Protocol):
    """Queue/storage adapter implemented by the deployment service on rank 0."""

    def acquire(self) -> InferenceJob | None:
        """Block until a job is available; return None for graceful shutdown."""

    def mark_running(self, job: InferenceJob) -> None: ...

    def mark_success(self, job: InferenceJob, result: InferenceResult) -> None: ...

    def mark_failed(
        self,
        job: InferenceJob,
        error: BaseException,
        traceback_text: str,
    ) -> None: ...


class Scail2DistributedRuntime:
    """Keep one engine resident and dispatch serial jobs to every rank.

    Model collectives use the engine's default NCCL process group. Runtime
    Distributed commands use a separate CPU/Gloo group, while rank 0 acquires
    jobs in a daemon thread. This lets every rank exchange an idle heartbeat
    without leaving a GPU occupied by an unmatched NCCL collective.
    """

    def __init__(
        self,
        engine: Scail2InferenceEngine,
        *,
        control_poll_seconds: float = 1.0,
        control_timeout_seconds: float = 120.0,
    ):
        if control_poll_seconds <= 0:
            raise ValueError("control_poll_seconds must be positive")
        if control_timeout_seconds <= control_poll_seconds:
            raise ValueError(
                "control_timeout_seconds must exceed control_poll_seconds"
            )
        self.engine = engine
        self.control_poll_seconds = control_poll_seconds
        self.control_timeout_seconds = control_timeout_seconds
        self._control_group: Any | None = None
        self._ready_event = threading.Event()
        self._stopped_event = threading.Event()
        self._failure: BaseException | None = None

    @property
    def ready(self) -> bool:
        return self._ready_event.is_set() and self.engine.state in {
            EngineState.READY,
            EngineState.BUSY,
        }

    @property
    def stopped(self) -> bool:
        return self._stopped_event.is_set()

    @property
    def failure(self) -> BaseException | None:
        return self._failure

    def wait_until_ready(self, timeout: float | None = None) -> bool:
        deadline = None if timeout is None else time.monotonic() + timeout
        while not self._stopped_event.is_set():
            wait_seconds = 0.1
            if deadline is not None:
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    return False
                wait_seconds = min(wait_seconds, remaining)
            if self._ready_event.wait(wait_seconds):
                return self.ready
        return False

    def wait_until_stopped(self, timeout: float | None = None) -> bool:
        return self._stopped_event.wait(timeout)

    def _emit_control_event(self, status: str) -> None:
        print(
            " ".join(
                [
                    "SCAIL2_CONTROL",
                    f"rank={self.engine.rank}",
                    "backend=gloo" if self.engine.distributed else "backend=local",
                    f"status={status}",
                    f"heartbeat_seconds={self.control_poll_seconds:g}",
                ]
            ),
            file=sys.stderr,
            flush=True,
        )

    def _initialize_control_group(self) -> None:
        if not self.engine.distributed:
            self._emit_control_event("ready")
            return
        import torch.distributed as dist

        if not dist.is_gloo_available():
            raise RuntimeError("PyTorch Gloo backend is unavailable")
        self._control_group = dist.new_group(
            ranks=list(range(self.engine.world_size)),
            backend="gloo",
            timeout=timedelta(seconds=self.control_timeout_seconds),
        )
        self._emit_control_event("ready")

    def _destroy_control_group(self) -> None:
        if self._control_group is None:
            return
        import torch.distributed as dist

        dist.destroy_process_group(self._control_group)
        self._control_group = None

    @staticmethod
    def _start_backend_acquire(
        backend: JobBackend,
    ) -> Queue[tuple[str, InferenceJob | BaseException | None]]:
        result_queue: Queue[
            tuple[str, InferenceJob | BaseException | None]
        ] = Queue(maxsize=1)

        def acquire_once() -> None:
            try:
                result_queue.put(("job", backend.acquire()))
            except BaseException as error:
                result_queue.put(("error", error))

        threading.Thread(
            target=acquire_once,
            name="scail2-backend-acquire",
            daemon=True,
        ).start()
        return result_queue

    def _primary_command(
        self,
        backend: JobBackend,
        acquisition: Queue[tuple[str, InferenceJob | BaseException | None]],
    ) -> tuple[dict[str, Any], InferenceJob | None]:
        try:
            kind, value = acquisition.get(timeout=self.control_poll_seconds)
        except Empty:
            return {"type": "idle"}, None
        if kind == "error":
            assert isinstance(value, BaseException)
            return (
                {
                    "type": "backend_error",
                    "error_type": type(value).__name__,
                    "error": str(value),
                },
                None,
            )
        if value is None:
            return {"type": "stop"}, None
        if not isinstance(value, InferenceJob):
            return (
                {
                    "type": "backend_error",
                    "error_type": "TypeError",
                    "error": "JobBackend.acquire() returned an invalid value",
                },
                None,
            )
        try:
            backend.mark_running(value)
        except BaseException as error:
            return (
                {
                    "type": "backend_error",
                    "error_type": type(error).__name__,
                    "error": str(error),
                },
                None,
            )
        return {"type": "infer", "job": value.to_dict()}, value

    def _broadcast_command(self, command: dict[str, Any] | None) -> dict[str, Any]:
        if not self.engine.distributed:
            if command is None:
                raise RuntimeError("Primary runtime command is missing")
            return command
        import torch
        import torch.distributed as dist

        if self._control_group is None:
            raise RuntimeError("Distributed control group is not initialized")

        values = [command if self.engine.is_primary else None]
        dist.broadcast_object_list(
            values,
            src=0,
            group=self._control_group,
            device=torch.device("cpu"),
        )
        if not isinstance(values[0], dict):
            raise RuntimeError("Primary rank broadcast an invalid runtime command")
        return values[0]

    def run(self, backend: JobBackend | None = None) -> None:
        """Run until rank 0's backend requests a graceful stop.

        The backend exists only on rank 0. Non-primary ranks wait for broadcast
        commands and never connect to the external queue or result store.
        """
        if self.engine.is_primary:
            if backend is None or not isinstance(backend, JobBackend):
                raise TypeError("Rank 0 requires a JobBackend implementation")
        elif backend is not None:
            raise ValueError("Only rank 0 may own a JobBackend")

        try:
            self.engine.load()
            self.engine.warmup()
            self._initialize_control_group()
            self._ready_event.set()
            acquisition = (
                self._start_backend_acquire(backend)
                if self.engine.is_primary
                else None
            )
            while True:
                command = None
                job = None
                if self.engine.is_primary:
                    assert backend is not None
                    assert acquisition is not None
                    command, job = self._primary_command(
                        backend,
                        acquisition,
                    )
                command = self._broadcast_command(command)
                command_type = command.get("type")
                if command_type == "idle":
                    continue
                if command_type == "stop":
                    break
                if command_type == "backend_error":
                    raise RuntimeError(
                        "Rank-0 JobBackend failed: "
                        f"{command.get('error_type')}: {command.get('error')}"
                    )
                if command_type != "infer" or not isinstance(
                    command.get("job"), dict
                ):
                    raise RuntimeError(f"Unsupported runtime command: {command!r}")
                job = InferenceJob.from_dict(command["job"])
                try:
                    result = self.engine.infer(job)
                except Exception as error:
                    callback_command = None
                    if self.engine.is_primary:
                        assert backend is not None
                        try:
                            backend.mark_failed(job, error, traceback.format_exc())
                            callback_command = {"type": "backend_ack"}
                        except BaseException as backend_error:
                            callback_command = {
                                "type": "backend_error",
                                "error_type": type(backend_error).__name__,
                                "error": str(backend_error),
                            }
                    callback_command = self._broadcast_command(callback_command)
                    if callback_command.get("type") == "backend_error":
                        raise RuntimeError(
                            "Rank-0 JobBackend failed: "
                            f"{callback_command.get('error_type')}: "
                            f"{callback_command.get('error')}"
                        ) from error
                    if self.engine.state == EngineState.READY:
                        # Input and output contract failures leave every rank
                        # synchronized and the resident model reusable.
                        if self.engine.is_primary:
                            assert backend is not None
                            acquisition = self._start_backend_acquire(backend)
                        continue
                    # Distributed inference failures may leave a collective in
                    # an unknown state. Fail the worker and let the supervisor
                    # restart it instead of risking a corrupt next result.
                    raise
                callback_command = None
                if self.engine.is_primary:
                    assert backend is not None
                    try:
                        backend.mark_success(job, result)
                        callback_command = {"type": "backend_ack"}
                    except BaseException as error:
                        callback_command = {
                            "type": "backend_error",
                            "error_type": type(error).__name__,
                            "error": str(error),
                        }
                callback_command = self._broadcast_command(callback_command)
                if callback_command.get("type") == "backend_error":
                    raise RuntimeError(
                        "Rank-0 JobBackend failed: "
                        f"{callback_command.get('error_type')}: "
                        f"{callback_command.get('error')}"
                    )
                if self.engine.is_primary:
                    assert backend is not None
                    acquisition = self._start_backend_acquire(backend)
        except BaseException as error:
            self._failure = error
            raise
        finally:
            self._ready_event.clear()
            try:
                self._destroy_control_group()
            finally:
                try:
                    self.engine.close()
                finally:
                    self._stopped_event.set()

"""Persistent single-GPU worker loop with a service-owned serial job backend."""

from __future__ import annotations

import threading
import traceback
from typing import Protocol, runtime_checkable

from .contracts import EngineState, InferenceJob, InferenceResult
from .engine import Scail2InferenceEngine


@runtime_checkable
class JobBackend(Protocol):
    """Queue/storage adapter implemented by the deployment service."""

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


class Scail2Runtime:
    """Keep one engine resident and execute backend jobs serially."""

    def __init__(self, engine: Scail2InferenceEngine):
        self.engine = engine
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
        return self._ready_event.wait(timeout) and self.ready

    def wait_until_stopped(self, timeout: float | None = None) -> bool:
        return self._stopped_event.wait(timeout)

    def run(self, backend: JobBackend) -> None:
        if not isinstance(backend, JobBackend):
            raise TypeError("A JobBackend implementation is required")
        try:
            self.engine.load()
            self.engine.warmup()
            self._ready_event.set()
            while True:
                job = backend.acquire()
                if job is None:
                    break
                if not isinstance(job, InferenceJob):
                    raise TypeError("JobBackend.acquire() returned an invalid value")
                backend.mark_running(job)
                try:
                    result = self.engine.infer(job)
                except Exception as error:
                    backend.mark_failed(job, error, traceback.format_exc())
                    if self.engine.state == EngineState.READY:
                        continue
                    raise
                backend.mark_success(job, result)
        except BaseException as error:
            self._failure = error
            raise
        finally:
            self._ready_event.clear()
            try:
                self.engine.close()
            finally:
                self._stopped_event.set()

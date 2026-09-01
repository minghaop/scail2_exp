"""Persistent, service-facing SCAIL-2 inference engine."""

from __future__ import annotations

import ctypes
import gc
import logging
import os
import threading
import time
from datetime import datetime, timezone
from fractions import Fraction
from pathlib import Path
from types import SimpleNamespace
from typing import Any

from .contracts import (
    EngineConfig,
    EngineState,
    InferenceJob,
    InferenceResult,
)
from .errors import (
    EngineStateError,
    EnvironmentValidationError,
    InputValidationError,
    OutputValidationError,
)
from .media import (
    atomic_publish_output,
    checkpoint_provenance,
    mux_driving_audio,
    output_validation_error,
    probe_audio,
    probe_video,
)


class Scail2InferenceEngine:
    """Load SCAIL-2 once and execute prepared jobs serially.

    One process owns one visible GPU. The engine remains resident across serial
    calls until ``close()`` or process termination.
    """

    def __init__(self, config: EngineConfig):
        config.validate(check_paths=False)
        self.config = config
        self.state = EngineState.CREATED
        self.pipeline: Any | None = None
        self.device_id = 0
        self._checkpoint_identity: dict[str, Any] | None = None
        self._cfg: Any | None = None
        self._lock = threading.Lock()

    def _require_state(self, *allowed: EngineState) -> None:
        if self.state not in allowed:
            allowed_text = ", ".join(state.value for state in allowed)
            raise EngineStateError(
                f"Engine state is {self.state.value}; expected one of {allowed_text}"
            )

    def _emit_initialization_event(
        self,
        stage: str,
        status: str,
        *,
        elapsed_seconds: float | None = None,
    ) -> None:
        """Emit one startup event through the service logger."""
        fields = [
            "SCAIL2_INIT",
            f"device={self.device_id}",
            f"stage={stage}",
            f"status={status}",
        ]
        if elapsed_seconds is not None:
            fields.append(f"elapsed_seconds={elapsed_seconds:.3f}")
        logging.info("%s", " ".join(fields))

    def _emit_ready_event(self) -> None:
        logging.info(
            "%s",
            " ".join(
                [
                    "SCAIL2_WORKER_READY",
                    f"device={self.device_id}",
                    f"profile={self.config.profile.name}",
                    f"resident_dtype={self.config.profile.resident_dtype}",
                ]
            ),
        )

    def _configure_process(self) -> None:
        import torch
        if not torch.cuda.is_available():
            raise EnvironmentValidationError("CUDA is unavailable")
        visible_gpus = torch.cuda.device_count()
        if visible_gpus != 1:
            raise EnvironmentValidationError(
                "The single-GPU worker requires exactly one visible CUDA device; "
                f"found {visible_gpus}"
            )
        torch.cuda.set_device(self.device_id)

    def load(self) -> None:
        """Initialize CUDA and load all model components exactly once."""
        self._require_state(EngineState.CREATED)
        self.state = EngineState.LOADING
        load_started = time.monotonic()
        self._emit_initialization_event("engine_load", "start")
        try:
            self.config.validate(check_paths=True)
            process_started = time.monotonic()
            self._emit_initialization_event("cuda_process", "start")
            self._configure_process()
            self._emit_initialization_event(
                "cuda_process",
                "complete",
                elapsed_seconds=time.monotonic() - process_started,
            )
            pipeline_started = time.monotonic()
            self._emit_initialization_event("pipeline_load", "start")
            self.pipeline, self._cfg, self._checkpoint_identity = (
                self._load_model_components()
            )
            self._emit_initialization_event(
                "pipeline_load",
                "complete",
                elapsed_seconds=time.monotonic() - pipeline_started,
            )
            self.state = EngineState.WARMING_UP
            self._emit_initialization_event(
                "engine_load",
                "complete",
                elapsed_seconds=time.monotonic() - load_started,
            )
        except Exception:
            self.state = EngineState.ERROR
            self._emit_initialization_event(
                "engine_load",
                "failed",
                elapsed_seconds=time.monotonic() - load_started,
            )
            raise

    def _load_model_components(self) -> tuple[Any, Any, dict[str, Any]]:
        """Load the production pipeline; split out for CPU lifecycle tests."""
        import wan
        from generate import _init_logging
        from wan.configs import SCAIL_CONFIGS

        _init_logging()
        model_name = self.config.profile.model.upper()
        if model_name not in SCAIL_CONFIGS:
            raise EnvironmentValidationError(
                f"Unknown model {model_name!r}; choices: {sorted(SCAIL_CONFIGS)}"
            )
        cfg = SCAIL_CONFIGS[model_name]
        scail_checkpoint = self.config.scail_checkpoint.resolve(strict=True)
        identity = checkpoint_provenance(scail_checkpoint)
        pipeline = wan.SCAIL2Pipeline(
            config=cfg,
            checkpoint_dir=str(self.config.checkpoint_dir.resolve(strict=True)),
            scail_safetensors_path=str(scail_checkpoint),
            scail_config_path=str(
                self.config.resolved_scail_config_path().resolve(strict=True)
            ),
            device_id=self.device_id,
            lora_path=None,
            lora_alpha=None,
            dit_resident_dtype=self.config.profile.resident_dtype,
            dit_meta_load=True,
            keep_dit_cpu_state_dict=True,
            vae_dit_offload_blocks=7,
            offload_vae_during_dit=True,
        )
        return pipeline, cfg, identity

    def warmup(self) -> None:
        """Synchronize the loaded worker before advertising readiness.

        A deployment may run a golden warmup job before calling this method.
        The SDK itself does not create an unrequested output artifact.
        """
        self._require_state(EngineState.WARMING_UP)
        warmup_started = time.monotonic()
        self._emit_initialization_event("device_synchronize", "start")
        self._synchronize_device()
        self._emit_initialization_event(
            "device_synchronize",
            "complete",
            elapsed_seconds=time.monotonic() - warmup_started,
        )
        if hasattr(self.pipeline, "assert_ready_residency"):
            self.pipeline.assert_ready_residency()
        self.state = EngineState.READY
        self._emit_initialization_event(
            "warmup",
            "complete",
            elapsed_seconds=time.monotonic() - warmup_started,
        )
        self._emit_ready_event()

    def _synchronize_device(self) -> None:
        import torch

        torch.cuda.synchronize(self.device_id)

    @staticmethod
    def _trim_process_heap() -> None:
        """Return freed large CPU work buffers to the OS on glibc systems."""
        try:
            trimmed = int(ctypes.CDLL(None).malloc_trim(0))
            logging.info("SCAIL2_CPU_HEAP action=trim result=%d", trimmed)
        except (AttributeError, OSError, TypeError, ValueError) as error:
            logging.warning("SCAIL2_CPU_HEAP action=trim status=unavailable error=%s", error)

    def _normalize_job(
        self, job: InferenceJob
    ) -> tuple[InferenceJob, dict[str, int | float | str]]:
        job.validate(check_paths=True)
        if job.reference_mask.suffix.lower() != ".png":
            raise InputValidationError(
                f"Job {job.job_id} reference mask must be lossless PNG"
            )
        if job.output_path.suffix.lower() != ".mp4":
            raise InputValidationError(
                f"Job {job.job_id} output path must use the .mp4 suffix"
            )
        try:
            pose_info = probe_video(job.driving_video)
            mask_info = probe_video(job.driving_mask)
            if self.config.output_audio_mode == "driving":
                probe_audio(job.driving_video)
        except Exception as error:
            raise InputValidationError(
                f"Job {job.job_id} video inspection failed: {error}"
            ) from error
        for field_name in ("width", "height", "frames", "fps_fraction"):
            if pose_info[field_name] != mask_info[field_name]:
                raise InputValidationError(
                    f"Job {job.job_id} driving video/mask {field_name} mismatch: "
                    f"{pose_info[field_name]} vs {mask_info[field_name]}"
                )
        frames = (
            int(pose_info["frames"])
            if job.expected_output_frames is None
            else job.expected_output_frames
        )
        if frames != int(pose_info["frames"]):
            raise InputValidationError(
                f"Job {job.job_id} expects {frames} frames but the driving video "
                f"contains {pose_info['frames']}"
            )
        fps_fraction = job.output_fps_fraction or str(pose_info["fps_fraction"])
        rate = Fraction(fps_fraction)
        duration = (
            float(pose_info["duration"])
            if job.expected_output_duration is None
            else job.expected_output_duration
        )
        cfr_duration = frames / float(rate)
        if abs(duration - cfr_duration) > 0.5 / float(rate):
            raise InputValidationError(
                f"Job {job.job_id} cannot preserve {frames} frames, FPS "
                f"{fps_fraction}, and duration {duration:.6f}s as CFR"
            )
        normalized = InferenceJob(
            job_id=job.job_id,
            reference_image=job.reference_image.resolve(strict=True),
            reference_mask=job.reference_mask.resolve(strict=True),
            driving_video=job.driving_video.resolve(strict=True),
            driving_mask=job.driving_mask.resolve(strict=True),
            t5_cache_path=job.t5_cache_path.resolve(strict=True),
            output_path=job.output_path.resolve(),
            output_fps_fraction=fps_fraction,
            expected_output_frames=frames,
            expected_output_duration=duration,
            seed=job.seed,
            overwrite=job.overwrite,
            metadata=job.metadata,
        )
        return normalized, pose_info

    def _run_generation(
        self,
        job: InferenceJob,
        *,
        temp_output: Path,
        seed: int,
    ) -> None:
        """Call the legacy, byte-verified generation adapter."""
        from generate import generate_video

        from .conditioning import expected_t5_metadata, load_t5_cache

        conditioning_started = time.monotonic()
        t5_checkpoint = self.config.checkpoint_dir / self._cfg.t5_checkpoint
        expected = expected_t5_metadata(
            profile=self.config.profile.name,
            text_len=self._cfg.text_len,
            t5_checkpoint=t5_checkpoint,
        )
        try:
            conditioning = load_t5_cache(job.t5_cache_path, expected)
        except Exception as error:
            raise InputValidationError(
                f"Job {job.job_id} T5 cache validation failed: {error}"
            ) from error
        logging.info(
            "SCAIL2_T5_CACHE job_id=%s status=loaded path=%s "
            "elapsed_seconds=%.3f text_shape=%s negative_shape=%s",
            job.job_id,
            job.t5_cache_path,
            time.monotonic() - conditioning_started,
            tuple(conditioning["text_context"].shape),
            tuple(conditioning["negative_context"].shape),
        )

        profile = self.config.profile
        args = SimpleNamespace(
            target_h=profile.target_height,
            target_w=profile.target_width,
            sample_shift=profile.sample_shift,
            sample_solver=profile.sample_solver,
            segment_len=profile.segment_len,
            segment_overlap=profile.segment_overlap,
            sample_steps=profile.sample_steps,
            sample_guide_scale=profile.guide_scale,
            base_seed=seed,
            save_file=str(temp_output),
            save_dir=str(job.output_path.parent),
        )
        generate_video(
            self.pipeline,
            str(job.reference_image),
            str(job.reference_mask),
            str(job.driving_video),
            str(job.driving_mask),
            args,
            self.device_id,
            self._cfg,
            False,
            {},
            output_fps=float(Fraction(job.output_fps_fraction or "")),
            output_fps_fraction=job.output_fps_fraction,
            conditioning=conditioning,
        )

    def infer(self, job: InferenceJob) -> InferenceResult:
        """Generate and strictly publish one video without unloading the model."""
        self._require_state(EngineState.READY)
        if not self._lock.acquire(blocking=False):
            raise EngineStateError("This engine already has an active inference job")
        self.state = EngineState.BUSY
        import torch

        started_at = datetime.now(timezone.utc).isoformat()
        temp_output: Path | None = None
        generation_output: Path | None = None
        audio_output: Path | None = None
        residency_mutated = False
        try:
            normalized_job, _ = self._normalize_job(job)
            profile = self.config.profile
            seed = profile.seed if normalized_job.seed is None else normalized_job.seed
            output_path = normalized_job.output_path
            expected_width = profile.target_width
            expected_height = profile.target_height
            try:
                output_path.parent.mkdir(parents=True, exist_ok=True)
                existing_error = output_validation_error(
                    output_path,
                    expected_width=expected_width,
                    expected_height=expected_height,
                    expected_fps_fraction=normalized_job.output_fps_fraction or "",
                    expected_frames=normalized_job.expected_output_frames or 0,
                    expected_duration=normalized_job.expected_output_duration or 0.0,
                    require_audio=self.config.output_audio_mode == "driving",
                )
                if output_path.exists() and existing_error is None and not normalized_job.overwrite:
                    decision = ("skip", None)
                elif output_path.exists() and existing_error is not None and not normalized_job.overwrite:
                    decision = ("error", existing_error)
                else:
                    decision = ("run", None)
            except Exception as error:
                decision = ("fatal", f"{type(error).__name__}: {error}")
            action, message = decision
            if action == "error":
                raise OutputValidationError(
                    f"Job {normalized_job.job_id} has an invalid existing output: {message}"
                )
            if action == "fatal":
                raise RuntimeError(
                    f"Job {normalized_job.job_id} output preparation failed: {message}"
                )
            if action == "run":
                if hasattr(self.pipeline, "assert_ready_residency"):
                    self.pipeline.assert_ready_residency()
                residency_mutated = True
                temp_output = output_path.with_name(
                    f".{output_path.stem}.inprogress-{os.getpid()}.mp4"
                )
                if temp_output.exists():
                    temp_output.unlink()
                generation_output = temp_output
                self._run_generation(
                    normalized_job,
                    temp_output=temp_output,
                    seed=seed,
                )
                validation_error = output_validation_error(
                    temp_output,
                    expected_width=expected_width,
                    expected_height=expected_height,
                    expected_fps_fraction=normalized_job.output_fps_fraction or "",
                    expected_frames=normalized_job.expected_output_frames or 0,
                    expected_duration=normalized_job.expected_output_duration or 0.0,
                )
                if validation_error is not None:
                    raise OutputValidationError(
                        f"Job {normalized_job.job_id} output validation failed: "
                        f"{validation_error}"
                    )
                if self.config.output_audio_mode == "driving":
                    audio_output = output_path.with_name(
                        f".{output_path.stem}.audio-{os.getpid()}.mp4"
                    )
                    postprocess_started = time.monotonic()
                    logging.info(
                        "SCAIL2_POSTPROCESS job_id=%s stage=audio_mux status=start source=%s",
                        normalized_job.job_id,
                        normalized_job.driving_video,
                    )
                    mux_driving_audio(
                        temp_output,
                        normalized_job.driving_video,
                        audio_output,
                        frames=normalized_job.expected_output_frames or 0,
                        fps_fraction=normalized_job.output_fps_fraction or "",
                    )
                    audio_error = output_validation_error(
                        audio_output,
                        expected_width=expected_width,
                        expected_height=expected_height,
                        expected_fps_fraction=normalized_job.output_fps_fraction or "",
                        expected_frames=normalized_job.expected_output_frames or 0,
                        expected_duration=normalized_job.expected_output_duration or 0.0,
                        require_audio=True,
                    )
                    if audio_error is not None:
                        raise OutputValidationError(audio_error)
                    logging.info(
                        "SCAIL2_POSTPROCESS job_id=%s stage=audio_mux status=complete elapsed_seconds=%.3f",
                        normalized_job.job_id,
                        time.monotonic() - postprocess_started,
                    )
                    temp_output = audio_output
                atomic_publish_output(
                    temp_output,
                    output_path,
                    overwrite=normalized_job.overwrite,
                )
                if hasattr(self.pipeline, "restore_ready_residency"):
                    self.pipeline.restore_ready_residency(
                        reason=f"job_{normalized_job.job_id}_complete"
                    )
                if hasattr(self.pipeline, "assert_ready_residency"):
                    self.pipeline.assert_ready_residency()
                residency_mutated = False

            output_info = probe_video(output_path)
            checkpoint = self._checkpoint_identity or {}
            result = InferenceResult(
                job_id=normalized_job.job_id,
                status="skipped" if action == "skip" else "success",
                output_path=output_path,
                profile=profile.name,
                seed=seed,
                frames=int(output_info["frames"]),
                fps_fraction=str(output_info["fps_fraction"]),
                duration=float(output_info["duration"]),
                width=int(output_info["width"]),
                height=int(output_info["height"]),
                started_at=started_at,
                finished_at=datetime.now(timezone.utc).isoformat(),
                checkpoint=str(checkpoint.get("path", "")),
                checkpoint_bytes=int(checkpoint.get("bytes", 0)),
                metadata=normalized_job.metadata,
            )
            self.state = EngineState.READY
            return result
        except (InputValidationError, OutputValidationError, FileExistsError):
            # Contract failures do not invalidate resident model parameters.
            if residency_mutated:
                try:
                    if hasattr(self.pipeline, "restore_ready_residency"):
                        self.pipeline.restore_ready_residency(
                            reason="contract_failure_recovery"
                        )
                    if hasattr(self.pipeline, "assert_ready_residency"):
                        self.pipeline.assert_ready_residency()
                except Exception as recovery_error:
                    self.state = EngineState.ERROR
                    raise EngineStateError(
                        "Could not restore READY model residency after a contract "
                        "failure"
                    ) from recovery_error
            self.state = EngineState.READY
            raise
        except Exception:
            # CUDA/model failures invalidate the resident worker.
            self.state = EngineState.ERROR
            raise
        finally:
            for candidate in (generation_output, audio_output, temp_output):
                if candidate is not None and candidate.exists():
                    try:
                        candidate.unlink()
                    except OSError as error:
                        logging.warning(
                            "Could not remove temporary output %s: %s",
                            candidate,
                            error,
                        )
            gc.collect()
            self._trim_process_heap()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            self._lock.release()

    def health(self) -> dict[str, Any]:
        return {
            "state": self.state.value,
            "ready": self.state == EngineState.READY,
            "model_loaded": self.pipeline is not None,
            "cuda_device": self.device_id,
            "profile": self.config.profile.name,
            "resident_dtype": self.config.profile.resident_dtype,
            "output_audio_mode": self.config.output_audio_mode,
            "checkpoint": None
            if self._checkpoint_identity is None
            else self._checkpoint_identity.get("path"),
        }

    def close(self) -> None:
        """Release the model only when the persistent worker is stopping."""
        if self.state == EngineState.CLOSED:
            return
        if self.state == EngineState.BUSY:
            raise EngineStateError("Cannot close the engine while inference is active")
        import torch

        self.pipeline = None
        self._cfg = None
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        self.state = EngineState.CLOSED

    def __enter__(self) -> "Scail2InferenceEngine":
        self.load()
        self.warmup()
        return self

    def __exit__(self, exc_type, exc, traceback) -> None:
        self.close()

"""Persistent, service-facing SCAIL-2 inference engine."""

from __future__ import annotations

import gc
import logging
import os
import sys
import threading
import time
from datetime import datetime, timedelta, timezone
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

    Every rank in a torchrun worker must construct the same engine and call
    ``load()``, ``warmup()``, and ``infer()`` in the same order. Only rank 0
    encodes and publishes the result; all ranks retain model parameters until
    ``close()`` or process termination.
    """

    def __init__(self, config: EngineConfig):
        config.validate(check_paths=False)
        self.config = config
        self.state = EngineState.CREATED
        self.pipeline: Any | None = None
        self.rank = int(os.getenv("RANK", "0"))
        self.local_rank = int(os.getenv("LOCAL_RANK", "0"))
        self.world_size = int(os.getenv("WORLD_SIZE", "1"))
        self._owns_process_group = False
        self._checkpoint_identity: dict[str, Any] | None = None
        self._cfg: Any | None = None
        self._lock = threading.Lock()

    @property
    def is_primary(self) -> bool:
        return self.rank == 0

    @property
    def distributed(self) -> bool:
        return self.world_size > 1

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
        """Emit one rank-aware startup event without polluting JSONL stdout."""
        fields = [
            "SCAIL2_INIT",
            f"rank={self.rank}",
            f"local_rank={self.local_rank}",
            f"world_size={self.world_size}",
            f"stage={stage}",
            f"status={status}",
        ]
        if elapsed_seconds is not None:
            fields.append(f"elapsed_seconds={elapsed_seconds:.3f}")
        print(" ".join(fields), file=sys.stderr, flush=True)

    def _emit_ready_event(self) -> None:
        if not self.is_primary:
            return
        print(
            " ".join(
                [
                    "SCAIL2_WORKER_READY",
                    f"rank={self.rank}",
                    f"world_size={self.world_size}",
                    f"profile={self.config.profile.name}",
                    f"resident_dtype={self.config.profile.resident_dtype}",
                ]
            ),
            file=sys.stderr,
            flush=True,
        )

    def _configure_process(self) -> None:
        import torch
        import torch.distributed as dist

        expected = self.config.expected_world_size
        if expected is not None and self.world_size != expected:
            raise EnvironmentValidationError(
                f"WORLD_SIZE is {self.world_size}, expected {expected}"
            )
        if not torch.cuda.is_available():
            raise EnvironmentValidationError("CUDA is unavailable")
        visible_gpus = torch.cuda.device_count()
        if visible_gpus < self.world_size:
            raise EnvironmentValidationError(
                f"Only {visible_gpus} CUDA device(s) are visible for "
                f"WORLD_SIZE={self.world_size}"
            )
        if self.local_rank < 0 or self.local_rank >= visible_gpus:
            raise EnvironmentValidationError(
                f"LOCAL_RANK={self.local_rank} is outside {visible_gpus} visible GPUs"
            )
        torch.cuda.set_device(self.local_rank)
        if self.distributed:
            if not dist.is_initialized():
                if not self.config.initialize_process_group:
                    raise EnvironmentValidationError(
                        "Distributed process group is not initialized"
                    )
                process_group_kwargs = {}
                timeout_text = os.getenv("SCAIL2_PROCESS_GROUP_TIMEOUT_SECONDS")
                if timeout_text:
                    timeout_seconds = float(timeout_text)
                    if timeout_seconds <= 0:
                        raise EnvironmentValidationError(
                            "SCAIL2_PROCESS_GROUP_TIMEOUT_SECONDS must be positive"
                        )
                    process_group_kwargs["timeout"] = timedelta(
                        seconds=timeout_seconds
                    )
                dist.init_process_group(
                    backend=self.config.distributed_backend,
                    init_method="env://",
                    rank=self.rank,
                    world_size=self.world_size,
                    device_id=torch.device("cuda", self.local_rank),
                    **process_group_kwargs,
                )
                self._owns_process_group = True
            if dist.get_rank() != self.rank or dist.get_world_size() != self.world_size:
                raise EnvironmentValidationError(
                    "Environment rank/world size does not match the process group"
                )
        elif self.config.t5_fsdp or self.config.dit_fsdp:
            raise EnvironmentValidationError(
                "FSDP is enabled but the worker is not distributed"
            )

    def load(self) -> None:
        """Initialize CUDA/FSDP and load all model components exactly once."""
        self._require_state(EngineState.CREATED)
        self.state = EngineState.LOADING
        load_started = time.monotonic()
        self._emit_initialization_event("engine_load", "start")
        try:
            self.config.validate(check_paths=True)
            process_started = time.monotonic()
            self._emit_initialization_event("process_group", "start")
            self._configure_process()
            self._emit_initialization_event(
                "process_group",
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
            barrier_started = time.monotonic()
            self._emit_initialization_event("post_load_barrier", "start")
            self._barrier()
            self._emit_initialization_event(
                "post_load_barrier",
                "complete",
                elapsed_seconds=time.monotonic() - barrier_started,
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

        _init_logging(self.rank)
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
            device_id=self.local_rank,
            rank=self.rank,
            t5_fsdp=self.config.t5_fsdp,
            dit_fsdp=self.config.dit_fsdp,
            cast_dit_forward_inputs=self.config.cast_dit_forward_inputs,
            use_usp=False,
            t5_cpu=self.config.t5_cpu,
            lora_path=None,
            lora_alpha=None,
            dit_resident_dtype=self.config.profile.resident_dtype,
            dit_meta_load=self.config.dit_meta_load,
            init_on_cpu=self.config.dit_init_on_cpu,
            keep_dit_cpu_state_dict=self.config.keep_dit_cpu_state_dict,
            vae_dit_offload_blocks=self.config.vae_dit_offload_blocks,
            offload_vae_during_dit=self.config.offload_vae_during_dit,
            t5_meta_load=self.config.t5_meta_load,
            precomputed_conditioning=self.config.precomputed_conditioning,
            online_clip_conditioning=self.config.online_clip_conditioning,
        )
        return pipeline, cfg, identity

    def _barrier(self) -> None:
        if self.distributed:
            import torch.distributed as dist

            dist.barrier()

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
        barrier_started = time.monotonic()
        self._emit_initialization_event("ready_barrier", "start")
        self._barrier()
        self._emit_initialization_event(
            "ready_barrier",
            "complete",
            elapsed_seconds=time.monotonic() - barrier_started,
        )
        self.state = EngineState.READY
        self._emit_initialization_event(
            "warmup",
            "complete",
            elapsed_seconds=time.monotonic() - warmup_started,
        )
        self._emit_ready_event()

    def _synchronize_device(self) -> None:
        import torch

        torch.cuda.synchronize(self.local_rank)

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
            prompt=job.prompt.strip(),
            output_path=job.output_path.resolve(),
            conditioning_path=(
                None
                if job.conditioning_path is None
                else job.conditioning_path.resolve(strict=True)
            ),
            output_fps_fraction=fps_fraction,
            expected_output_frames=frames,
            expected_output_duration=duration,
            seed=job.seed,
            overwrite=job.overwrite,
            metadata=job.metadata,
        )
        return normalized, pose_info

    def _broadcast_object(self, value: Any) -> Any:
        if not self.distributed:
            return value
        import torch.distributed as dist

        container = [value if self.is_primary else None]
        dist.broadcast_object_list(container, src=0)
        return container[0]

    def _run_generation(
        self,
        job: InferenceJob,
        *,
        temp_output: Path,
        seed: int,
        diagnostic_memory_probe: bool = False,
        diagnostic_memory_probe_steps: int = 1,
        diagnostic_segment_limit: int | None = None,
    ) -> None:
        """Call the legacy, byte-verified generation adapter."""
        from generate import generate_video

        conditioning = None
        if self.config.precomputed_conditioning:
            if job.conditioning_path is None:
                raise InputValidationError(
                    f"Job {job.job_id} requires a conditioning artifact"
                )
            from .conditioning import expected_metadata, load_conditioning

            conditioning_started = time.monotonic()
            t5_checkpoint = self.config.checkpoint_dir / self._cfg.t5_checkpoint
            clip_checkpoint = self.config.checkpoint_dir / self._cfg.clip_checkpoint
            expected = expected_metadata(
                prompt=job.prompt,
                negative_prompt="",
                reference_image=job.reference_image,
                target_width=self.config.profile.target_width,
                target_height=self.config.profile.target_height,
                t5_checkpoint=t5_checkpoint,
                clip_checkpoint=clip_checkpoint,
            )
            try:
                conditioning = load_conditioning(job.conditioning_path, expected)
            except Exception as error:
                raise InputValidationError(
                    f"Job {job.job_id} conditioning validation failed: {error}"
                ) from error
            logging.info(
                "SCAIL2_CONDITIONING job_id=%s status=loaded path=%s "
                "elapsed_seconds=%.3f text_shape=%s negative_shape=%s "
                "clip_shape=%s",
                job.job_id,
                job.conditioning_path,
                time.monotonic() - conditioning_started,
                tuple(conditioning["text_context"].shape),
                tuple(conditioning["negative_context"].shape),
                tuple(conditioning["clip_context"].shape),
            )
        elif job.conditioning_path is not None:
            raise InputValidationError(
                f"Job {job.job_id} supplied conditioning_path, but the engine "
                "was not configured for precomputed conditioning"
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
            offload_model=False,
            save_file=str(temp_output),
            save_dir=str(job.output_path.parent),
            ring_size=1,
            prompt=job.prompt,
            diagnostic_memory_probe=diagnostic_memory_probe,
            diagnostic_memory_probe_steps=diagnostic_memory_probe_steps,
            diagnostic_segment_limit=diagnostic_segment_limit,
        )
        generate_video(
            self.pipeline,
            job.prompt,
            str(job.reference_image),
            str(job.reference_mask),
            str(job.driving_video),
            str(job.driving_mask),
            args,
            self.local_rank,
            self.rank,
            self._cfg,
            None,
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
        import torch.distributed as dist

        started_at = datetime.now(timezone.utc).isoformat()
        temp_output: Path | None = None
        generation_output: Path | None = None
        audio_output: Path | None = None
        try:
            normalized_job, _ = self._normalize_job(job)
            profile = self.config.profile
            seed = profile.seed if normalized_job.seed is None else normalized_job.seed
            output_path = normalized_job.output_path
            expected_width = profile.target_width
            expected_height = profile.target_height
            decision: tuple[str, str | None] | None = None
            if self.is_primary:
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
                    if (
                        output_path.exists()
                        and existing_error is None
                        and not normalized_job.overwrite
                    ):
                        decision = ("skip", None)
                    elif (
                        output_path.exists()
                        and existing_error is not None
                        and not normalized_job.overwrite
                    ):
                        decision = ("error", existing_error)
                    else:
                        decision = ("run", None)
                except Exception as error:
                    decision = (
                        "fatal",
                        f"{type(error).__name__}: {error}",
                    )
            decision = self._broadcast_object(decision)
            if decision is None:
                raise RuntimeError("Primary rank did not provide an inference decision")
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
                if self.is_primary:
                    temp_output = output_path.with_name(
                        f".{output_path.stem}.inprogress-{os.getpid()}.mp4"
                    )
                    if temp_output.exists():
                        temp_output.unlink()
                temp_text = self._broadcast_object(
                    str(temp_output) if self.is_primary else None
                )
                temp_output = Path(temp_text)
                generation_output = temp_output
                self._barrier()
                self._run_generation(
                    normalized_job,
                    temp_output=temp_output,
                    seed=seed,
                )
                self._barrier()

                validation_error = None
                if self.is_primary:
                    validation_error = output_validation_error(
                        temp_output,
                        expected_width=expected_width,
                        expected_height=expected_height,
                        expected_fps_fraction=normalized_job.output_fps_fraction or "",
                        expected_frames=normalized_job.expected_output_frames or 0,
                        expected_duration=normalized_job.expected_output_duration or 0.0,
                    )
                validation_error = self._broadcast_object(validation_error)
                if validation_error is not None:
                    raise OutputValidationError(
                        f"Job {normalized_job.job_id} output validation failed: "
                        f"{validation_error}"
                    )
                if self.config.output_audio_mode == "driving":
                    postprocess_message: dict[str, str | None] | None = None
                    if self.is_primary:
                        audio_output = output_path.with_name(
                            f".{output_path.stem}.audio-{os.getpid()}.mp4"
                        )
                        postprocess_started = time.monotonic()
                        logging.info(
                            "SCAIL2_POSTPROCESS job_id=%s stage=audio_mux "
                            "status=start source=%s",
                            normalized_job.job_id,
                            normalized_job.driving_video,
                        )
                        try:
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
                                expected_fps_fraction=(
                                    normalized_job.output_fps_fraction or ""
                                ),
                                expected_frames=(
                                    normalized_job.expected_output_frames or 0
                                ),
                                expected_duration=(
                                    normalized_job.expected_output_duration or 0.0
                                ),
                                require_audio=True,
                            )
                            if audio_error is not None:
                                raise OutputValidationError(audio_error)
                            logging.info(
                                "SCAIL2_POSTPROCESS job_id=%s stage=audio_mux "
                                "status=complete elapsed_seconds=%.3f",
                                normalized_job.job_id,
                                time.monotonic() - postprocess_started,
                            )
                            postprocess_message = {
                                "path": str(audio_output),
                                "error": None,
                            }
                        except Exception as error:
                            postprocess_message = {
                                "path": None,
                                "error": f"{type(error).__name__}: {error}",
                            }
                    postprocess_message = self._broadcast_object(postprocess_message)
                    if not isinstance(postprocess_message, dict):
                        raise OutputValidationError(
                            f"Job {normalized_job.job_id} audio postprocessing "
                            "returned no result"
                        )
                    if postprocess_message.get("error") is not None:
                        raise OutputValidationError(
                            f"Job {normalized_job.job_id} audio postprocessing failed: "
                            f"{postprocess_message['error']}"
                        )
                    postprocessed_path = postprocess_message.get("path")
                    if not postprocessed_path:
                        raise OutputValidationError(
                            f"Job {normalized_job.job_id} audio postprocessing "
                            "returned no output path"
                        )
                    temp_output = Path(postprocessed_path)
                publish_error = None
                if self.is_primary:
                    try:
                        atomic_publish_output(
                            temp_output,
                            output_path,
                            overwrite=normalized_job.overwrite,
                        )
                    except Exception as error:
                        publish_error = f"{type(error).__name__}: {error}"
                publish_error = self._broadcast_object(publish_error)
                if publish_error is not None:
                    raise OutputValidationError(
                        f"Job {normalized_job.job_id} output publication failed: "
                        f"{publish_error}"
                    )
                self._barrier()

            result_message = None
            if self.is_primary:
                try:
                    output_info = probe_video(output_path)
                    checkpoint = self._checkpoint_identity or {}
                    result_message = {
                        "result": InferenceResult(
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
                        ).to_dict()
                    }
                except Exception as error:
                    result_message = {
                        "error": f"{type(error).__name__}: {error}"
                    }
            result_message = self._broadcast_object(result_message)
            if not isinstance(result_message, dict):
                raise RuntimeError("Primary rank did not return an inference result")
            if result_message.get("error") is not None:
                raise OutputValidationError(
                    f"Job {normalized_job.job_id} final result inspection failed: "
                    f"{result_message['error']}"
                )
            result_payload = result_message.get("result")
            if not isinstance(result_payload, dict):
                raise RuntimeError("Primary rank did not return an inference result")
            self.state = EngineState.READY
            return InferenceResult.from_dict(result_payload)
        except (InputValidationError, OutputValidationError, FileExistsError):
            # Contract failures do not invalidate resident model parameters.
            self.state = EngineState.READY
            raise
        except Exception:
            # CUDA/NCCL/model failures may leave collectives in an unknown
            # state. The runtime treats ERROR as fatal and restarts the worker.
            self.state = EngineState.ERROR
            raise
        finally:
            if self.is_primary:
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
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            self._lock.release()

    def health(self) -> dict[str, Any]:
        return {
            "state": self.state.value,
            "ready": self.state == EngineState.READY,
            "model_loaded": self.pipeline is not None,
            "rank": self.rank,
            "local_rank": self.local_rank,
            "world_size": self.world_size,
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
        import torch.distributed as dist

        self.pipeline = None
        self._cfg = None
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        if self._owns_process_group and dist.is_initialized():
            dist.destroy_process_group()
        self.state = EngineState.CLOSED

    def __enter__(self) -> "Scail2InferenceEngine":
        self.load()
        self.warmup()
        return self

    def __exit__(self, exc_type, exc, traceback) -> None:
        self.close()

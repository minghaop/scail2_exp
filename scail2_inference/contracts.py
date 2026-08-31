"""Versioned configuration and job/result contracts for SCAIL-2 inference."""

from __future__ import annotations

import json
import math
from dataclasses import asdict, dataclass, field
from enum import Enum
from fractions import Fraction
from importlib import resources
from pathlib import Path
from typing import Any, Mapping

from .errors import EnvironmentValidationError, InputValidationError


class EngineState(str, Enum):
    CREATED = "created"
    LOADING = "loading"
    WARMING_UP = "warming_up"
    READY = "ready"
    BUSY = "busy"
    ERROR = "error"
    CLOSED = "closed"


@dataclass(frozen=True)
class ProductionProfile:
    """Algorithm settings that form part of the output compatibility contract."""

    name: str = "scail2-512p-bf16-v1"
    model: str = "SCAIL-14B"
    resident_dtype: str = "bf16"
    target_width: int = 512
    target_height: int = 896
    sample_solver: str = "euler"
    sample_steps: int = 6
    sample_shift: float = 5.0
    guide_scale: float = 1.0
    seed: int = 42
    segment_len: int = 81
    segment_overlap: int = 1

    def validate(self) -> None:
        if not self.name.strip():
            raise EnvironmentValidationError("Profile name cannot be empty")
        if self.resident_dtype not in {"fp32", "bf16"}:
            raise EnvironmentValidationError(
                f"Unsupported resident dtype: {self.resident_dtype!r}"
            )
        if self.target_width <= 0 or self.target_height <= 0:
            raise EnvironmentValidationError("Target dimensions must be positive")
        if self.target_width % 32 or self.target_height % 32:
            raise EnvironmentValidationError(
                "Target width and height must be divisible by 32"
            )
        if self.sample_solver not in {"unipc", "dpm++", "euler"}:
            raise EnvironmentValidationError(
                f"Unsupported sample solver: {self.sample_solver!r}"
            )
        if self.sample_steps <= 0:
            raise EnvironmentValidationError("sample_steps must be positive")
        if not math.isfinite(self.sample_shift):
            raise EnvironmentValidationError("sample_shift must be finite")
        if not math.isfinite(self.guide_scale) or self.guide_scale <= 0:
            raise EnvironmentValidationError("guide_scale must be finite and positive")
        if self.seed < 0:
            raise EnvironmentValidationError("seed must be nonnegative")
        if (self.segment_len - 1) % 4:
            raise EnvironmentValidationError("segment_len must equal 4*n+1")
        if not 0 < self.segment_overlap < self.segment_len:
            raise EnvironmentValidationError(
                "segment_overlap must be in (0, segment_len)"
            )
        if (self.segment_overlap - 1) % 4:
            raise EnvironmentValidationError("segment_overlap must equal 4*n+1")

    @classmethod
    def from_name(cls, name: str) -> "ProductionProfile":
        filename = f"{name}.json"
        try:
            profile_text = (
                resources.files("scail2_inference.profiles")
                .joinpath(filename)
                .read_text(encoding="utf-8")
            )
        except FileNotFoundError as exc:
            raise EnvironmentValidationError(
                f"Unknown packaged production profile: {name!r}"
            ) from exc
        profile = cls(**json.loads(profile_text))
        profile.validate()
        return profile

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class EngineConfig:
    """Installation and process settings for one persistent model instance."""

    checkpoint_dir: Path
    scail_checkpoint: Path
    scail_config_path: Path | None = None
    profile: ProductionProfile = field(default_factory=ProductionProfile)
    output_audio_mode: str = "none"

    def validate(self, *, check_paths: bool = True) -> None:
        self.profile.validate()
        if self.output_audio_mode not in {"none", "driving"}:
            raise EnvironmentValidationError(
                "output_audio_mode must be either 'none' or 'driving'"
            )
        if check_paths:
            for label, path, require_directory in (
                ("checkpoint directory", self.checkpoint_dir, True),
                ("SCAIL checkpoint", self.scail_checkpoint, False),
                ("SCAIL config", self.resolved_scail_config_path(), False),
            ):
                candidate = Path(path)
                valid = candidate.is_dir() if require_directory else candidate.is_file()
                if not valid:
                    raise EnvironmentValidationError(
                        f"Invalid {label}: {candidate}"
                    )

    def resolved_scail_config_path(self) -> Path:
        if self.scail_config_path is not None:
            return Path(self.scail_config_path)
        model_name = self.profile.model.upper()
        filename_by_model = {
            "SCAIL-14B": "config-14b.json",
            "SCAIL-1.3B": "config-1.3b.json",
        }
        try:
            filename = filename_by_model[model_name]
        except KeyError as exc:
            raise EnvironmentValidationError(
                f"No packaged architecture config for model {model_name!r}"
            ) from exc
        resource = resources.files("scail2_inference.model_configs").joinpath(
            filename
        )
        path = Path(str(resource))
        if not path.is_file():
            raise EnvironmentValidationError(
                f"Packaged architecture config is unavailable: {filename}"
            )
        return path

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["checkpoint_dir"] = str(self.checkpoint_dir)
        payload["scail_checkpoint"] = str(self.scail_checkpoint)
        payload["scail_config_path"] = (
            None if self.scail_config_path is None else str(self.scail_config_path)
        )
        payload["profile"] = self.profile.to_dict()
        return payload


@dataclass(frozen=True)
class InferenceJob:
    """One fully prepared local-file animation request."""

    job_id: str
    reference_image: Path
    reference_mask: Path
    driving_video: Path
    driving_mask: Path
    t5_cache_path: Path
    output_path: Path
    output_fps_fraction: str | None = None
    expected_output_frames: int | None = None
    expected_output_duration: float | None = None
    seed: int | None = None
    overwrite: bool = False
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def validate(self, *, check_paths: bool = True) -> None:
        if not self.job_id.strip():
            raise InputValidationError("job_id cannot be empty")
        if self.seed is not None and self.seed < 0:
            raise InputValidationError(f"Job {self.job_id} seed must be nonnegative")
        if self.output_fps_fraction is not None:
            try:
                rate = Fraction(self.output_fps_fraction)
            except (ValueError, ZeroDivisionError) as exc:
                raise InputValidationError(
                    f"Job {self.job_id} has invalid output FPS: "
                    f"{self.output_fps_fraction!r}"
                ) from exc
            if rate <= 0:
                raise InputValidationError(
                    f"Job {self.job_id} output FPS must be positive"
                )
        if self.expected_output_frames is not None and self.expected_output_frames <= 0:
            raise InputValidationError(
                f"Job {self.job_id} expected_output_frames must be positive"
            )
        if self.expected_output_duration is not None and (
            not math.isfinite(self.expected_output_duration)
            or self.expected_output_duration <= 0
        ):
            raise InputValidationError(
                f"Job {self.job_id} expected_output_duration must be finite and positive"
            )
        if self.output_path.name in {"", ".", ".."}:
            raise InputValidationError(f"Job {self.job_id} has an invalid output path")
        if check_paths:
            for label, path in (
                ("reference image", self.reference_image),
                ("reference mask", self.reference_mask),
                ("driving video", self.driving_video),
                ("driving mask", self.driving_mask),
                ("T5 cache", self.t5_cache_path),
            ):
                candidate = Path(path)
                if not candidate.is_file() or candidate.stat().st_size == 0:
                    raise InputValidationError(
                        f"Job {self.job_id} has invalid {label}: {candidate}"
                    )
    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        for field_name in (
            "reference_image",
            "reference_mask",
            "driving_video",
            "driving_mask",
            "t5_cache_path",
            "output_path",
        ):
            payload[field_name] = str(payload[field_name])
        payload["metadata"] = dict(self.metadata)
        return payload

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "InferenceJob":
        values = dict(payload)
        for field_name in (
            "reference_image",
            "reference_mask",
            "driving_video",
            "driving_mask",
            "t5_cache_path",
            "output_path",
        ):
            values[field_name] = Path(values[field_name])
        return cls(**values)


@dataclass(frozen=True)
class InferenceResult:
    job_id: str
    status: str
    output_path: Path
    profile: str
    seed: int
    frames: int
    fps_fraction: str
    duration: float
    width: int
    height: int
    started_at: str
    finished_at: str
    checkpoint: str
    checkpoint_bytes: int
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["output_path"] = str(self.output_path)
        payload["metadata"] = dict(self.metadata)
        return payload

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "InferenceResult":
        values = dict(payload)
        values["output_path"] = Path(values["output_path"])
        return cls(**values)

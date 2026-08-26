"""Stable public API for the deployable SCAIL-2 inference runtime."""

from .contracts import (
    EngineConfig,
    EngineState,
    InferenceJob,
    InferenceResult,
    ProductionProfile,
)
from .engine import Scail2InferenceEngine
from .errors import (
    EngineStateError,
    EnvironmentValidationError,
    InputValidationError,
    OutputValidationError,
    Scail2InferenceError,
)
from .runtime import JobBackend, Scail2DistributedRuntime

__all__ = [
    "EngineConfig",
    "EngineState",
    "EngineStateError",
    "EnvironmentValidationError",
    "InferenceJob",
    "InferenceResult",
    "InputValidationError",
    "JobBackend",
    "OutputValidationError",
    "ProductionProfile",
    "Scail2DistributedRuntime",
    "Scail2InferenceEngine",
    "Scail2InferenceError",
]

__version__ = "0.1.3"

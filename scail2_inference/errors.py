"""Typed failures exposed by the SCAIL-2 inference SDK."""


class Scail2InferenceError(RuntimeError):
    """Base class for SDK failures that callers may report structurally."""


class EnvironmentValidationError(Scail2InferenceError):
    """The host, process topology, or model installation is invalid."""


class EngineStateError(Scail2InferenceError):
    """An operation is not valid for the engine's current lifecycle state."""


class InputValidationError(Scail2InferenceError):
    """A job does not satisfy the local-file inference contract."""


class OutputValidationError(Scail2InferenceError):
    """A generated artifact failed strict media validation."""

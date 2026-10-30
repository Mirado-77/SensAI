"""Shared exceptions. Expected errors derive from SensaiError."""


class SensaiError(Exception):
    """Base class for all expected SENSAI errors."""


class ConfigError(SensaiError):
    """Configuration file missing, unreadable or invalid."""


class RunLimitError(SensaiError):
    """A run exceeded one of its configured limits (steps, tool calls, tokens, time)."""


class MigrationError(SensaiError):
    """A database migration file is invalid or failed to apply."""


class ModelError(SensaiError):
    """Error raised by a ModelGateway."""


class ModelConnectionError(ModelError):
    """The Ollama server is unreachable or the connection was dropped."""


class ModelNotFoundError(ModelError):
    """The requested model is not installed on the Ollama server."""


class StructuredOutputError(ModelError):
    """The model did not produce JSON conforming to the schema, even after repair."""


class GuardBlockedError(SensaiError):
    """A guard refused the input of a turn or the output of a tool. `stage` is "entry" or "tool"."""

    def __init__(self, stage: str, reason: str) -> None:
        super().__init__(reason)
        self.stage = stage
        self.reason = reason


class ExportError(SensaiError):
    """Session that cannot be exported, or export format not available (X3)."""

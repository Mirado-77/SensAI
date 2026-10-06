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

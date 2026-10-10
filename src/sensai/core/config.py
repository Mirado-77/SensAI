"""Loading and validation of the TOML configuration."""

from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, ValidationError

from sensai.core.errors import ConfigError
from sensai.core.types import Decision, Risk

try:
    import tomllib
except ModuleNotFoundError:
    import tomli as tomllib

DEFAULT_CONFIG_PATH = Path(__file__).resolve().parents[3] / "configs" / "default.toml"


class _Section(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class ModelConfig(_Section):
    name: str = Field(min_length=1)
    host: str = Field(min_length=1)
    timeout_s: float = Field(gt=0)


class RuntimeConfig(_Section):
    max_steps: int = Field(gt=0)
    max_tool_calls: int = Field(ge=0)
    timeout_s: float = Field(gt=0)
    max_tokens: int = Field(gt=0)


class StructuredConfig(_Section):
    max_repairs: int = Field(ge=0, le=3)


class StorageConfig(_Section):
    db_path: Path


class PolicyConfig(_Section):
    read: Decision
    write: Decision
    exec: Decision


class McpServerConfig(_Section):
    """One MCP server launched over stdio (T1). Its tools are named `<name>__<tool>`."""

    name: str = Field(pattern=r"^[A-Za-z0-9_-]+$")
    command: str = Field(min_length=1)
    args: list[str] = []
    env: dict[str, str] = {}
    cwd: Path | None = None
    enabled: bool = True
    risk: Risk | Literal["auto"] = "exec"


class McpConfig(_Section):
    timeout_s: float = Field(gt=0)
    servers: list[McpServerConfig] = []


class Config(_Section):
    model: ModelConfig
    runtime: RuntimeConfig
    structured: StructuredConfig
    storage: StorageConfig
    policy: PolicyConfig
    mcp: McpConfig


def _read_toml(path: Path) -> dict[str, Any]:
    try:
        with path.open("rb") as f:
            return tomllib.load(f)
    except FileNotFoundError:
        raise ConfigError(f"Fichier de configuration introuvable : {path}") from None
    except tomllib.TOMLDecodeError as exc:
        raise ConfigError(f"Fichier de configuration illisible ({path}) : {exc}") from exc


def _deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    merged = dict(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = _deep_merge(merged[key], value)
        else:
            merged[key] = value
    return merged


def load_config(path: Path | None = None, overrides: dict[str, Any] | None = None) -> Config:
    """Load `configs/default.toml`, then overlay `path`, then `overrides`.

    `path` only contains the keys to change. `overrides` carries CLI arguments: the caller only
    includes the values that were actually provided.
    """
    data = _read_toml(DEFAULT_CONFIG_PATH)
    if path is not None:
        data = _deep_merge(data, _read_toml(path))
    if overrides:
        data = _deep_merge(data, overrides)
    try:
        return Config.model_validate(data)
    except ValidationError as exc:
        raise ConfigError(f"Configuration invalide :\n{exc}") from exc

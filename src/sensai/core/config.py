"""Loading and validation of the TOML configuration."""

from pathlib import Path
from typing import Any, Literal

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    ValidationError,
    field_validator,
    model_validator,
)

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
    num_ctx: int = Field(gt=0)


class RuntimeConfig(_Section):
    max_steps: int = Field(gt=0)
    max_tool_calls: int = Field(ge=0)
    timeout_s: float = Field(gt=0)
    max_tokens: int = Field(gt=0)


class StructuredConfig(_Section):
    max_repairs: int = Field(ge=0, le=3)


class ReactConfig(_Section):
    """Bounded ReAct loop (A1): plan first, then act and observe one step at a time."""

    enabled: bool
    max_plan_steps: int = Field(gt=0, le=10)


class QuestionsConfig(_Section):
    """Structured questions (X4): the model may ask the user when the request is ambiguous."""

    enabled: bool
    max_per_turn: int = Field(gt=0, le=5)


class PromptsConfig(_Section):
    """Versioned system prompts (A4). `system` names a file of configs/prompts/system; "" = none."""

    system: str = Field(pattern=r"^([A-Za-z0-9_-]+)?$")


class MemoryConfig(_Section):
    """Context budget (M2): old turns are summarized once the context reaches `compress_at` of
    the model window (`[model] num_ctx`); the last `keep_turns` turns are kept as they are."""

    enabled: bool
    compress_at: float = Field(gt=0, le=1)
    keep_turns: int = Field(gt=0)


class StorageConfig(_Section):
    db_path: Path


class PolicyConfig(_Section):
    read: Decision
    write: Decision
    exec: Decision


class AllowedRoot(_Section):
    path: Path
    mode: Literal["read", "write"]


class FilesConfig(_Section):
    max_read_bytes: int = Field(gt=0)
    max_write_bytes: int = Field(gt=0)
    max_diff_lines: int = Field(gt=0)
    max_list_entries: int = Field(gt=0)
    allowed_roots: list[AllowedRoot] = Field(default_factory=list)


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


class SandboxConfig(_Section):
    enabled: bool
    image: str = Field(min_length=1)
    cpus: float = Field(gt=0)
    memory_mb: int = Field(gt=0)
    pids_limit: int = Field(gt=0)
    timeout_s: int = Field(gt=0)
    max_output_bytes: int = Field(gt=0)


class WebConfig(_Section):
    """Web search (T3) through a self-hosted SearXNG instance with its JSON format enabled."""

    enabled: bool
    base_url: str = Field(min_length=1)
    max_results: int = Field(gt=0, le=20)
    timeout_s: float = Field(gt=0)
    max_result_chars: int = Field(gt=0)


class RagConfig(_Section):
    """Codebase retrieval (R1): `sensai index` embeds chunks of lines, `search_code` finds them."""

    enabled: bool
    embed_model: str = Field(min_length=1)
    chunk_lines: int = Field(gt=0)
    overlap: int = Field(ge=0)
    top_k: int = Field(gt=0, le=20)

    @model_validator(mode="after")
    def _overlap_below_chunk(self) -> "RagConfig":
        if self.overlap >= self.chunk_lines:
            raise ValueError("overlap must be smaller than chunk_lines")
        return self


class GuardsConfig(_Section):
    """Input and output guards (EV2): injection check, secret redaction from `env_file`."""

    enabled: bool
    env_file: Path


class SchedulerConfig(_Section):
    """Scheduled tasks (X5): commands run at a fixed interval by `sensai schedule`."""

    enabled: bool
    min_interval_minutes: int = Field(gt=0)
    sync_interval_s: float = Field(gt=0)
    allowed_commands: list[list[str]] = Field(default_factory=list)

    @field_validator("allowed_commands")
    @classmethod
    def _complete_commands(cls, commands: list[list[str]]) -> list[list[str]]:
        if any(not command or not all(command) for command in commands):
            raise ValueError("each allowed command needs a program and non-empty arguments")
        return commands


class Config(_Section):
    model: ModelConfig
    runtime: RuntimeConfig
    structured: StructuredConfig
    react: ReactConfig
    questions: QuestionsConfig
    prompts: PromptsConfig
    memory: MemoryConfig
    storage: StorageConfig
    policy: PolicyConfig
    files: FilesConfig
    mcp: McpConfig
    sandbox: SandboxConfig
    rag: RagConfig
    guards: GuardsConfig
    web: WebConfig
    scheduler: SchedulerConfig


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

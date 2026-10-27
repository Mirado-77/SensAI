"""Benchmark definition read from `benchmarks/<name>.toml`."""

from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, ValidationError, model_validator

from sensai.benchmarks.metrics import METRICS
from sensai.core.errors import SensaiError

try:
    import tomllib
except ModuleNotFoundError:
    import tomli as tomllib

DEFAULT_DIR = Path("benchmarks")


class SpecError(SensaiError):
    """A benchmark file is missing, is not valid TOML, or does not follow the format."""


class Variant(BaseModel):
    """One configuration to compare: `overrides` is merged over the loaded config, for example
    `[variants.overrides.rag] min_score = 0.0`."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    name: str = Field(pattern=r"^[a-z0-9][a-z0-9-]*$")
    overrides: dict[str, Any] = Field(default_factory=dict)


class BenchmarkSpec(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    name: str = Field(pattern=r"^[a-z0-9][a-z0-9-]*$")
    title: str = Field(min_length=1)
    description: str = ""
    kind: Literal["questions", "attacks"]
    corpus: Path
    metrics: list[str] = Field(min_length=1)
    index: bool = True
    variants: list[Variant] = Field(min_length=1)

    @model_validator(mode="after")
    def _check_metrics_and_variants(self) -> "BenchmarkSpec":
        for name in self.metrics:
            if name not in METRICS:
                raise ValueError(f"unknown metric {name!r}, expected one of {sorted(METRICS)}")
            if self.kind not in METRICS[name].kinds:
                raise ValueError(f"metric {name!r} does not apply to {self.kind}")
        names = [variant.name for variant in self.variants]
        if len(set(names)) != len(names):
            raise ValueError("variant names must be unique")
        return self

    @property
    def judged(self) -> bool:
        return "judge_score" in self.metrics


def resolve(argument: str) -> Path:
    """A path to a spec file, or the name of a file of `benchmarks/`."""
    path = Path(argument)
    return path if path.exists() else DEFAULT_DIR / f"{argument}.toml"


def load_spec(path: Path) -> BenchmarkSpec:
    try:
        data = tomllib.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        raise SpecError(f"Benchmark introuvable : {path}") from None
    except (OSError, UnicodeDecodeError, tomllib.TOMLDecodeError) as exc:
        raise SpecError(f"Benchmark illisible ({path}) : {exc}") from exc
    try:
        return BenchmarkSpec.model_validate(data)
    except ValidationError as exc:
        details = "; ".join(error["msg"] for error in exc.errors())
        raise SpecError(f"Benchmark invalide ({path}) : {details}") from exc


def list_specs(directory: Path = DEFAULT_DIR) -> list[str]:
    return [path.stem for path in sorted(directory.glob("*.toml"))]

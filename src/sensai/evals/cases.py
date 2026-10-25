"""Attack cases of the EV4 corpus: one TOML file per case, read from a directory."""

from pathlib import Path
from typing import Literal, TypeVar

from pydantic import BaseModel, ConfigDict, Field, ValidationError

from sensai.core.errors import SensaiError

try:
    import tomllib
except ModuleNotFoundError:
    import tomli as tomllib

Category = Literal[
    "injection-fichier",
    "injection-outil",
    "chemin-hors-racine",
    "fuite-secret",
    "commande-dangereuse",
]


class CaseError(SensaiError):
    """A case file is missing, is not valid TOML, or does not follow the case format."""


class Forbidden(BaseModel):
    """What must not happen during the turn. A tool call is forbidden if its name is listed
    and, when `args` is given, one of its substrings appears in the JSON of the arguments."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    tools: list[str] = Field(default_factory=list)
    args: list[str] = Field(default_factory=list)
    text: list[str] = Field(default_factory=list)
    secrets: bool = True


class Setup(BaseModel):
    """Extra files written in the copy of the demo repository, relative path -> content."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    files: dict[str, str] = Field(default_factory=dict)


class AttackCase(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    id: str = Field(pattern=r"^[a-z0-9][a-z0-9-]*$")
    category: Category
    prompt: str = Field(min_length=1)
    expected: str = Field(min_length=1)
    timeout_s: float | None = Field(default=None, gt=0)
    forbidden: Forbidden
    setup: Setup = Field(default_factory=Setup)


Case = TypeVar("Case", bound=BaseModel)


def load_cases(directory: Path, model: type[Case] = AttackCase) -> list[Case]:
    """Load every `*.toml` case of `directory` as `model`, sorted by file name. Ids must be
    unique. EV1 reuses it for its question cases."""
    cases = [_load_case(path, model) for path in sorted(directory.glob("*.toml"))]
    ids = [case.id for case in cases]
    duplicates = sorted({case_id for case_id in ids if ids.count(case_id) > 1})
    if duplicates:
        raise CaseError(f"Identifiants de cas en double : {', '.join(duplicates)}")
    return cases


def _load_case(path: Path, model: type[Case]) -> Case:
    try:
        data = tomllib.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, tomllib.TOMLDecodeError) as exc:
        raise CaseError(f"Cas illisible ({path.name}) : {exc}") from exc
    try:
        return model.model_validate(data)
    except ValidationError as exc:
        details = "; ".join(error["msg"] for error in exc.errors())
        raise CaseError(f"Cas invalide ({path.name}) : {details}") from exc

"""Personas (A5): who the assistant is for this turn, as configuration.

Each persona is a TOML file `configs/personas/<name>.toml`:

    description = "one line shown by /persona"
    prompt = "instructions added to the system prompt"
    risks = ["read", "write", "exec"]  # tool risk levels the persona may use

`[prompts] persona` picks the one active at start, `/persona <name>` switches in the chat for the
next turns, in the same session: the history is kept. A persona without `exec` (the reviewer) is
not offered the tools of that risk, and a call to one of them is refused.
"""

import re
from dataclasses import dataclass
from pathlib import Path
from typing import get_args

from sensai.core.config import DEFAULT_CONFIG_PATH, tomllib
from sensai.core.errors import ConfigError
from sensai.core.types import Risk

PERSONAS_DIR = DEFAULT_CONFIG_PATH.parent / "personas"
NAME_PATTERN = r"^[A-Za-z0-9_-]+$"
RISKS: tuple[str, ...] = get_args(Risk)
_KEYS = {"description", "prompt", "risks"}


@dataclass(frozen=True)
class Persona:
    name: str
    description: str
    prompt: str
    risks: frozenset[str]


def list_personas(directory: Path = PERSONAS_DIR) -> list[Persona]:
    """All personas, sorted by name."""
    return [load_persona(path.stem, directory) for path in sorted(directory.glob("*.toml"))]


def load_persona(name: str, directory: Path = PERSONAS_DIR) -> Persona:
    if not re.match(NAME_PATTERN, name):
        raise ConfigError(f"Nom de persona invalide : {name!r}")
    path = directory / f"{name}.toml"
    try:
        with path.open("rb") as f:
            data = tomllib.load(f)
    except FileNotFoundError:
        known = ", ".join(path.stem for path in sorted(directory.glob("*.toml"))) or "aucune"
        raise ConfigError(f"Persona introuvable : {name} (connues : {known})") from None
    except tomllib.TOMLDecodeError as exc:
        raise ConfigError(f"Persona {name} : TOML invalide : {exc}") from None
    return _persona(name, data)


def _persona(name: str, data: dict) -> Persona:
    unknown = sorted(set(data) - _KEYS)
    if unknown:
        raise ConfigError(f"Persona {name} : clés inconnues : {', '.join(unknown)}")
    prompt, description = data.get("prompt"), data.get("description", "")
    if not isinstance(prompt, str) or not prompt.strip():
        raise ConfigError(f"Persona {name} : `prompt` doit être un texte non vide")
    if not isinstance(description, str):
        raise ConfigError(f"Persona {name} : `description` doit être un texte")
    risks = data.get("risks", list(RISKS))
    if not isinstance(risks, list) or not set(risks) <= set(RISKS):
        raise ConfigError(f"Persona {name} : `risks` doit être une liste parmi {', '.join(RISKS)}")
    return Persona(name, description.strip(), prompt.strip(), frozenset(risks))

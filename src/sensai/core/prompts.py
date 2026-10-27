"""Versioned system prompts (A4).

Each version is a text file `configs/prompts/system/<name>.txt` kept in git: a new version is a
new file, never an edit of an old one. `[prompts] system` picks the active version; going back
to an older name is the rollback. Every run records `<name>@<sha>` in `run.started`
(`prompt_version`), so the event journal can compare versions (`sensai prompts stats`) and an
edited file shows up as a new sha.
"""

import difflib
import hashlib
import re
from dataclasses import dataclass
from pathlib import Path

from sensai.core.config import DEFAULT_CONFIG_PATH
from sensai.core.errors import ConfigError

PROMPTS_DIR = DEFAULT_CONFIG_PATH.parent / "prompts" / "system"
NAME_PATTERN = r"^[A-Za-z0-9_-]+$"


@dataclass(frozen=True)
class PromptVersion:
    name: str
    text: str

    @property
    def sha(self) -> str:
        return hashlib.sha256(self.text.encode()).hexdigest()[:8]

    @property
    def label(self) -> str:
        return f"{self.name}@{self.sha}"


def list_versions(directory: Path = PROMPTS_DIR) -> list[PromptVersion]:
    """All versions, in natural order (v2 before v10)."""
    paths = sorted(directory.glob("*.txt"), key=lambda path: _natural_key(path.stem))
    return [PromptVersion(path.stem, path.read_text(encoding="utf-8").strip()) for path in paths]


def load_version(name: str, directory: Path = PROMPTS_DIR) -> PromptVersion:
    if not re.match(NAME_PATTERN, name):
        raise ConfigError(f"Nom de version de prompt invalide : {name!r}")
    path = directory / f"{name}.txt"
    try:
        return PromptVersion(name, path.read_text(encoding="utf-8").strip())
    except FileNotFoundError:
        known = ", ".join(version.name for version in list_versions(directory)) or "aucune"
        raise ConfigError(f"Version de prompt introuvable : {name} (connues : {known})") from None


def diff_versions(old: PromptVersion, new: PromptVersion) -> str:
    lines = difflib.unified_diff(
        old.text.splitlines(), new.text.splitlines(), old.label, new.label, lineterm=""
    )
    return "\n".join(lines)


def system_prompt(
    version: PromptVersion | None, profile: str | None, persona: str | None = None
) -> str | None:
    """The base prompt, then the persona (A5), then the profile (M1); None when all are empty."""
    parts = [part for part in (version.text if version else None, persona, profile) if part]
    return "\n\n".join(parts) or None


def _natural_key(name: str) -> list[int | str]:
    return [int(part) if part.isdigit() else part for part in re.split(r"(\d+)", name)]

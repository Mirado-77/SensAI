"""Secrets of the project `.env` and redaction of them, plus common API key formats."""

import re
from collections.abc import Iterable
from pathlib import Path

REDACTED = "[REDACTED]"
MIN_VALUE_LENGTH = 8
_KEY_FORMATS = [
    r"sk-[A-Za-z0-9_-]{16,}",
    r"AKIA[0-9A-Z]{16}",
    r"gh[pousr]_[A-Za-z0-9]{20,}",
]


def read_env_file(env_file: Path) -> dict[str, str]:
    """Values of the KEY=VALUE lines of `env_file`, keyed by name. Missing file gives {}."""
    if not env_file.is_file():
        return {}
    values = {}
    for line in env_file.read_text(encoding="utf-8").splitlines():
        if line.lstrip().startswith("#"):
            continue
        name, separator, value = line.partition("=")
        if separator and name.strip() and value.strip():
            values[name.strip()] = value.strip().strip("'\"")
    return values


class SecretRedactor:
    """Replaces the given values (longest first, short ones ignored) and key formats."""

    def __init__(self, values: Iterable[str]) -> None:
        literal = sorted(
            {value for value in values if len(value) >= MIN_VALUE_LENGTH},
            key=len,
            reverse=True,
        )
        parts = [re.escape(value) for value in literal] + _KEY_FORMATS
        self._pattern = re.compile("|".join(parts))

    def redact(self, text: str) -> tuple[str, int]:
        """Return the redacted text and the number of replacements."""
        return self._pattern.subn(REDACTED, text)

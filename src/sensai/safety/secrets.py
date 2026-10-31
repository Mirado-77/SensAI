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
# shortcut: emails and phone numbers (international +CC or French 0X XX XX XX XX), not names.
_EMAIL = r"[A-Za-z0-9._%+-]+@[A-Za-z0-9-]+(?:\.[A-Za-z0-9-]+)*\.[A-Za-z]{2,}"
_PHONE = r"(?<![\w+])(?:\+\d{1,3}(?:[ .-]?\d){8,12}|0[1-9](?:[ .-]?\d{2}){4})(?!\d)"
_PII = [(re.compile(_EMAIL), "[EMAIL]"), (re.compile(_PHONE), "[PHONE]")]


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
    """Replaces the given values (longest first, short ones ignored), key formats and, with
    `pii`, emails and phone numbers."""

    def __init__(self, values: Iterable[str], pii: bool = False) -> None:
        literal = sorted(
            {value for value in values if len(value) >= MIN_VALUE_LENGTH},
            key=len,
            reverse=True,
        )
        parts = [re.escape(value) for value in literal] + _KEY_FORMATS
        self._pattern = re.compile("|".join(parts))
        self._pii = _PII if pii else []

    def redact(self, text: str) -> tuple[str, int]:
        """Return the redacted text and the number of replacements."""
        text, count = self._pattern.subn(REDACTED, text)
        for pattern, placeholder in self._pii:
            text, found = pattern.subn(placeholder, text)
            count += found
        return text, count

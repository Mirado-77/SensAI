"""The terminal the CLI talks to: output, error output and the line reader."""

from collections.abc import Callable
from dataclasses import dataclass

from rich.console import Console


@dataclass(frozen=True)
class Terminal:
    console: Console
    errors: Console
    read_line: Callable[[str], str] = input

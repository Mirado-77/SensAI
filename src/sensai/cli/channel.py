"""UserChannel that talks to the user through the terminal."""

from typing import Any

from rich.text import Text

from sensai.cli.terminal import Terminal
from sensai.core.types import ApprovalRequest

_YES = {"y", "yes"}


class CliUserChannel:
    def __init__(self, terminal: Terminal) -> None:
        self._terminal = terminal

    async def confirm(self, request: ApprovalRequest) -> bool:
        header = Text(f"Approval required ({request.risk}): ", style="bold yellow")
        header.append(request.preview)
        self._terminal.console.print(header)
        try:
            answer = self._terminal.read_line("Allow? [y/N] ")
        except EOFError:
            return False
        return answer.strip().lower() in _YES

    async def ask(self, question: dict[str, Any]) -> dict[str, Any]:
        raise NotImplementedError("Les questions structurées ne sont pas encore prises en charge.")

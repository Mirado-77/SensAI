"""UserChannel that talks to the user through the terminal."""

from typing import Any

from rich.text import Text

from sensai.cli.terminal import Terminal
from sensai.core.types import ApprovalRequest

_YES = {"y", "yes"}


def _styled_preview(preview: str) -> Text:
    text = Text()
    lines = preview.split("\n")
    if not lines[0].startswith("--- "):
        text.append(preview)
        return text
    for index, line in enumerate(lines):
        if line.startswith(("--- ", "+++ ")):
            style = "bold"
        elif line.startswith("@@"):
            style = "cyan"
        elif line.startswith("+"):
            style = "green"
        elif line.startswith("-"):
            style = "red"
        else:
            style = ""
        text.append(line, style=style)
        if index < len(lines) - 1:
            text.append("\n")
    return text


class CliUserChannel:
    def __init__(self, terminal: Terminal) -> None:
        self._terminal = terminal

    async def confirm(self, request: ApprovalRequest) -> bool:
        header = Text()
        header.append(f"Approval required ({request.risk}): ", style="bold yellow")
        if request.preview.startswith("--- "):
            header.append("\n")
        header.append_text(_styled_preview(request.preview))
        self._terminal.console.print(header)
        try:
            answer = self._terminal.read_line("Allow? [y/N] ")
        except EOFError:
            return False
        return answer.strip().lower() in _YES

    async def ask(self, question: dict[str, Any]) -> dict[str, Any]:
        """Print the question and its numbered options; a number picks an option."""
        console = self._terminal.console
        console.print(Text(f"Question: {question['question']}", style="bold cyan"))
        options = question.get("options") or []
        for index, option in enumerate(options, 1):
            console.print(Text(f"  {index}. {option}"))
        hint = f"[1-{len(options)} or text] " if options else ""
        try:
            answer = self._terminal.read_line(f"Answer {hint}> ").strip()
        except EOFError:
            return {"answer": ""}
        if answer.isdigit() and 1 <= int(answer) <= len(options):
            answer = options[int(answer) - 1]
        return {"answer": answer}

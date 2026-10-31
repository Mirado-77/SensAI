"""Versioned system prompts in the CLI (A4): /prompt in the chat and `sensai prompts`."""

from dataclasses import dataclass

from rich.text import Text

from sensai.cli.terminal import Terminal
from sensai.core.errors import ConfigError
from sensai.core.personas import Persona
from sensai.core.prompts import (
    PromptVersion,
    diff_versions,
    list_versions,
    load_version,
    system_prompt,
)
from sensai.core.runtime import AgentRuntime

PROMPT_COMMAND = "/prompt"


@dataclass
class ChatPrompt:
    """What the chat sends as system prompt: a prompt version, the persona (A5), the profile (M1).

    The persona also sets the tool risk levels the runtime offers to the model.
    """

    version: PromptVersion | None
    profile: str | None = None
    persona: Persona | None = None

    def apply(self, runtime: AgentRuntime) -> None:
        persona = self.persona
        runtime.system_prompt = system_prompt(
            self.version, self.profile, persona.prompt if persona else None
        )
        runtime.prompt_version = self.version.label if self.version else None
        runtime.persona = persona.name if persona else None
        runtime.tool_risks = persona.risks if persona else None


def run_prompt_command(prompt: ChatPrompt, terminal: Terminal, text: str) -> None:
    """`/prompt` shows the active version, `/prompt <name>` switches to it for the next turns."""
    name = text.removeprefix(PROMPT_COMMAND).strip()
    if name:
        try:
            prompt.version = load_version(name)
        except ConfigError as exc:
            terminal.errors.print(Text(f"Erreur : {exc}", style="yellow"))
            return
    active = prompt.version.label if prompt.version else "none"
    names = ", ".join(version.name for version in list_versions())
    terminal.console.print(Text(f"System prompt: {active} (versions: {names})", "dim"))


def print_prompt_versions(terminal: Terminal, active: str) -> int:
    versions = list_versions()
    if not versions:
        terminal.console.print(Text("No prompt version.", "dim"))
    for version in versions:
        marker = "*" if version.name == active else " "
        lines = len(version.text.splitlines())
        terminal.console.print(Text(f"{marker} {version.label}  {lines} lines"))
    return 0


def print_prompt_diff(terminal: Terminal, old: str, new: str) -> int:
    try:
        diff = diff_versions(load_version(old), load_version(new))
    except ConfigError as exc:
        terminal.errors.print(Text(f"Erreur : {exc}", style="bold red"))
        return 2
    terminal.console.print(Text(diff or "Identical versions."), soft_wrap=True)
    return 0


def print_prompt_stats(rows: list[dict], terminal: Terminal) -> int:
    if not rows:
        terminal.console.print(Text("No run recorded yet.", "dim"))
        return 0
    header = f"{'version':<14} {'runs':>5} {'ok':>4} {'failed':>6} {'tokens':>7} {'s':>6}"
    terminal.console.print(Text(f"{header} {'tools':>6} {'errors':>6}", "bold"))
    for row in rows:
        tokens = f"{row['avg_tokens'] or 0:.0f}"
        seconds = f"{row['avg_seconds'] or 0:.1f}"
        line = (
            f"{row['version']:<14} {row['runs']:>5} {row['completed']:>4} {row['failed']:>6} "
            f"{tokens:>7} {seconds:>6} {row['tool_calls']:>6} {row['tool_errors']:>6}"
        )
        terminal.console.print(Text(line))
    return 0

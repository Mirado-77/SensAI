"""Personas in the chat (A5): /persona shows them, /persona <name> switches for the next turns."""

from rich.text import Text

from sensai.cli.prompts import ChatPrompt
from sensai.cli.terminal import Terminal
from sensai.core.errors import ConfigError
from sensai.core.personas import RISKS, Persona, list_personas, load_persona

PERSONA_COMMAND = "/persona"


def run_persona_command(prompt: ChatPrompt, terminal: Terminal, text: str) -> None:
    """`/persona` lists the personas, `/persona <name>` switches to it; the history is kept."""
    name = text.removeprefix(PERSONA_COMMAND).strip()
    if name:
        try:
            prompt.persona = load_persona(name)
        except ConfigError as exc:
            terminal.errors.print(Text(f"Erreur : {exc}", style="yellow"))
            return
        terminal.console.print(Text(f"Persona: {describe(prompt.persona)}", "dim"))
        return
    try:
        personas = list_personas()
    except ConfigError as exc:
        terminal.errors.print(Text(f"Erreur : {exc}", style="yellow"))
        return
    active = prompt.persona.name if prompt.persona else None
    if not personas:
        terminal.console.print(Text("No persona.", "dim"))
    for persona in personas:
        marker = "*" if persona.name == active else " "
        terminal.console.print(Text(f"{marker} {describe(persona)}"))


def describe(persona: Persona) -> str:
    """`reviewer [read]: Reviewer: ...`, the risk levels in their usual order."""
    risks = ", ".join(risk for risk in RISKS if risk in persona.risks) or "no tools"
    return f"{persona.name} [{risks}]" + (f": {persona.description}" if persona.description else "")

"""The project memory in the chat (M3): `/memory` lists it, `/memory forget <id>` deletes one."""

from rich.text import Text

from sensai.cli.terminal import Terminal
from sensai.memory.facts import FactStore

MEMORY_COMMAND = "/memory"
USAGE = "Usage : /memory, ou /memory forget <id>"


def run_memory_command(facts: FactStore | None, terminal: Terminal, text: str) -> None:
    if facts is None:
        terminal.errors.print(Text("Mémoire du projet désactivée ([memory] enabled).", "yellow"))
        return
    words = text.split()
    if len(words) == 1:
        records = facts.records()
        if not records:
            terminal.console.print(Text("La mémoire du projet est vide.", "dim"))
        for record in records:
            terminal.console.print(Text(record.line()))
    elif len(words) == 3 and words[1] == "forget":
        _forget(facts, terminal, words[2])
    else:
        terminal.errors.print(Text(USAGE, "yellow"))


def _forget(facts: FactStore, terminal: Terminal, prefix: str) -> None:
    matches = facts.find(prefix)
    if len(matches) != 1:
        problem = "aucun enregistrement" if not matches else "plusieurs enregistrements"
        terminal.errors.print(Text(f"Erreur : {problem} pour {prefix}.", "yellow"))
        return
    facts.delete(matches[0].id)
    terminal.console.print(Text(f"Oublié : {matches[0].line()}", "dim"))

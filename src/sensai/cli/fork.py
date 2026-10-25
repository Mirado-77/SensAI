"""Branching a conversation (X2): /history lists the current branch, /fork starts a new one.

A fork is a new session whose head is an earlier message: the history up to it is shared, the
original session is left as it was. Ctrl-C during an answer is handled by the chat loop; the
partial text is saved with the status "interrupted" and shown as such here.
"""

from rich.text import Text

from sensai.cli.terminal import Terminal
from sensai.memory.sqlite_store import SqliteSessionStore

HISTORY_COMMAND = "/history"
FORK_COMMAND = "/fork"
FORK_HELP = "/fork <n>   continue in a new session from message n of /history"


def print_history(store: SqliteSessionStore, terminal: Terminal, session_id: str) -> None:
    history = store.history(session_id)
    if not history:
        terminal.console.print(Text("No message yet.", "dim"))
    for number, message in enumerate(history, 1):
        line = Text(f"{number} ", "bold")
        line.append(f"{message.role:<9} ", "dim")
        line.append(message.content.split("\n")[0][:80])
        if message.status == "interrupted":
            line.append(" (interrupted)", "yellow")
        terminal.console.print(line)


def run_fork_command(
    store: SqliteSessionStore, terminal: Terminal, session_id: str, text: str
) -> str:
    """Apply /fork <n>, print the outcome and return the session to continue in."""
    history = store.history(session_id)
    number = text.removeprefix(FORK_COMMAND).strip()
    if not number.isdigit() or not 1 <= int(number) <= len(history):
        terminal.errors.print(Text(f"Commande /fork invalide.\n{FORK_HELP}", style="yellow"))
        return session_id
    message = history[int(number) - 1]
    assert message.id is not None
    fork_id = store.fork(session_id, message.id)
    terminal.console.print(Text(f"Forked session {fork_id} from message {number}", "dim"))
    return fork_id

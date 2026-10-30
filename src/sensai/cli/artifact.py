"""The state document in the chat (M4): `/doc` shows it, a fork starts with a copy of it."""

from dataclasses import dataclass

from rich.text import Text

from sensai.cli.terminal import Terminal
from sensai.memory.artifact import ArtifactStore

DOC_COMMAND = "/doc"


@dataclass(frozen=True)
class StateDocument:
    store: ArtifactStore
    sections: list[str]

    def is_empty(self, session_id: str) -> bool:
        return not any(self.store.sections(session_id).values())

    def render(self, session_id: str) -> str:
        return self.store.render(session_id, self.sections)

    def fork(self, from_session: str, to_session: str) -> None:
        if from_session != to_session:
            self.store.copy(from_session, to_session)


def print_document(document: StateDocument | None, terminal: Terminal, session_id: str) -> None:
    if document is None:
        terminal.errors.print(Text("Document d'état désactivé ([artifact] enabled).", "yellow"))
        return
    if document.is_empty(session_id):
        terminal.console.print(Text("Le document d'état est vide.", "dim"))
        return
    terminal.console.print(Text(document.render(session_id)))

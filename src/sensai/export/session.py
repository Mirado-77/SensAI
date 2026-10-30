"""What a session export contains: the conversation and the state document (M4)."""

from typing import Any

from sensai.core.errors import ExportError
from sensai.memory.artifact import ArtifactStore
from sensai.memory.sqlite_store import SqliteSessionStore
from sensai.safety.secrets import SecretRedactor

EXPORTED_ROLES = ("user", "assistant")


def session_export(
    store: SqliteSessionStore,
    artifacts: ArtifactStore,
    session_id: str,
    sections: list[str],
    redactor: SecretRedactor | None = None,
) -> dict[str, Any]:
    """The session as plain data, ready for JSON. Only the user and assistant messages are kept:
    raw tool outputs stay out of an export meant to be read or shared."""
    rows = store.list_sessions(1, session_id)
    if not rows or rows[0]["id"] != session_id:
        raise ExportError(f"Session inconnue ou vide : {session_id}")

    def clean(text: str) -> str:
        return redactor.redact(text)[0] if redactor else text

    messages = [
        {"role": message.role, "content": clean(message.content), "status": message.status}
        for message in store.history(session_id)
        if message.role in EXPORTED_ROLES
    ]
    saved = artifacts.sections(session_id)
    names = sections + [name for name in saved if name not in sections]
    document = [
        {"section": name, "content": clean(saved[name])} for name in names if saved.get(name)
    ]
    return {
        "session_id": session_id,
        "created_at": rows[0]["created_at"],
        "messages": messages,
        "document": document,
    }

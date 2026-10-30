"""Sessions, branches, profile, the project memory (M3), then compression and artifact."""

import sqlite3
from collections.abc import Callable

from sensai.core.config import Config
from sensai.core.contracts import EventSink, Tool
from sensai.memory.artifact import ArtifactStore, artifact_note
from sensai.memory.artifact_tools import ArtifactPatchTool
from sensai.memory.facts import FactStore, recall_note
from sensai.memory.tools import (
    MemoryDeleteTool,
    MemoryRelateTool,
    MemorySaveTool,
    MemorySearchTool,
    MemoryUpdateTool,
)


def memory_tools(config: Config, connection: sqlite3.Connection) -> list[Tool]:
    """The project memory tools of the chat (M3), when `[memory] enabled`."""
    if not config.memory.enabled:
        return []
    store = FactStore(connection)
    return [
        MemorySaveTool(store),
        MemoryRelateTool(store),
        MemorySearchTool(store),
        MemoryUpdateTool(store),
        MemoryDeleteTool(store),
    ]


def memory_recall(
    config: Config, connection: sqlite3.Connection
) -> Callable[[str], str | None] | None:
    """What the runtime adds to the context at each turn, or None when recall is off."""
    if not config.memory.enabled or not config.memory.recall:
        return None
    store = FactStore(connection)
    return lambda text: recall_note(store, text, config.memory.recall)


def artifact_tools(config: Config, connection: sqlite3.Connection, sink: EventSink) -> list[Tool]:
    """The state document tool of the chat (M4), when `[artifact] enabled`."""
    if not config.artifact.enabled:
        return []
    return [
        ArtifactPatchTool(
            ArtifactStore(connection),
            sink,
            config.artifact.sections,
            config.artifact.max_section_chars,
        )
    ]


def artifact_note_for(
    config: Config, connection: sqlite3.Connection
) -> Callable[[str], str | None] | None:
    """What the runtime adds to the context at each turn (M4), or None when it is off."""
    if not config.artifact.enabled:
        return None
    store = ArtifactStore(connection)
    return lambda session_id: artifact_note(store, session_id, config.artifact.sections)

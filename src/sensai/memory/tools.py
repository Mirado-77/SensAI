"""Tools of M3: the model saves, links, searches, updates and deletes the project memory.

Every change has risk "write", so the user approves it after seeing `preview`. A file or a tool
output cannot plant a lasting instruction in the memory without that approval.
"""

from typing import Any

from sensai.core.types import Risk, ToolContext, ToolResult
from sensai.memory.facts import KINDS, Fact, FactStore, Record

SEARCH_LIMIT = 20
NOT_FOUND = (
    "Aucun enregistrement de la mémoire ne commence par {prefix}. Cherche-le avec memory_search."
)
AMBIGUOUS = "Plusieurs enregistrements commencent par {prefix} : donne plus de caractères de l'id."

_TEXT = {"type": "string", "minLength": 1}
_ID = {"type": "string", "minLength": 4, "description": "Id of the record, its first 8 characters"}


class MemorySaveTool:
    name = "memory_save"
    description = (
        "Save a durable fact about the project in the long-term memory, kept across sessions:"
        " a convention (how code is written), a decision (what was chosen and why), a fact or an"
        " entity (a module, a service). Save only what will still be true later."
    )
    risk: Risk = "write"
    parameters = {
        "type": "object",
        "properties": {
            "kind": {"type": "string", "enum": list(KINDS)},
            "subject": {**_TEXT, "description": "What it is about, for example: taskman/stats.py"},
            "content": {**_TEXT, "description": "The fact itself, in one or two sentences"},
        },
        "required": ["kind", "subject", "content"],
    }

    def __init__(self, store: FactStore) -> None:
        self._store = store

    async def run(self, args: dict[str, Any], ctx: ToolContext) -> ToolResult:
        fact_id = self._store.save(args["kind"], args["subject"], args["content"])
        return ToolResult(f"Enregistré dans la mémoire : {fact_id[:8]}.")

    async def preview(self, args: dict[str, Any]) -> str | None:
        fact = f"[{args['kind']}] {args['subject']} : {args['content']}"
        return f"Ajouter à la mémoire du projet\n{fact}"


class MemoryRelateTool:
    name = "memory_relate"
    description = (
        "Save a relation between two subjects of the project memory, for example"
        " 'taskman/storage.py' 'uses' 'taskman/models.py'."
    )
    risk: Risk = "write"
    parameters = {
        "type": "object",
        "properties": {"subject": _TEXT, "relation": _TEXT, "object": _TEXT},
        "required": ["subject", "relation", "object"],
    }

    def __init__(self, store: FactStore) -> None:
        self._store = store

    async def run(self, args: dict[str, Any], ctx: ToolContext) -> ToolResult:
        relation_id = self._store.relate(args["subject"], args["relation"], args["object"])
        return ToolResult(f"Relation enregistrée : {relation_id[:8]}.")

    async def preview(self, args: dict[str, Any]) -> str | None:
        return (
            "Ajouter une relation à la mémoire du projet\n"
            f"{args['subject']} -> {args['relation']} -> {args['object']}"
        )


class MemorySearchTool:
    name = "memory_search"
    description = (
        "Search the long-term project memory (facts and relations saved in earlier sessions)."
        " An empty query lists the most recent records."
    )
    risk: Risk = "read"
    parameters = {
        "type": "object",
        "properties": {
            "query": {"type": "string", "description": "Words to look for"},
            "kind": {"type": "string", "enum": list(KINDS)},
        },
    }

    def __init__(self, store: FactStore) -> None:
        self._store = store

    async def run(self, args: dict[str, Any], ctx: ToolContext) -> ToolResult:
        records = self._store.search(args.get("query", ""), SEARCH_LIMIT, args.get("kind"))
        if not records:
            return ToolResult("Rien de correspondant dans la mémoire du projet.")
        return ToolResult("\n".join(record.line() for record in records), untrusted=True)


class MemoryUpdateTool:
    name = "memory_update"
    description = "Replace the content of a fact of the project memory that is no longer accurate."
    risk: Risk = "write"
    parameters = {
        "type": "object",
        "properties": {"id": _ID, "content": _TEXT},
        "required": ["id", "content"],
    }

    def __init__(self, store: FactStore) -> None:
        self._store = store

    async def run(self, args: dict[str, Any], ctx: ToolContext) -> ToolResult:
        record = _resolve(self._store, args["id"])
        if isinstance(record, ToolResult):
            return record
        if not isinstance(record, Fact):
            return ToolResult(
                "Une relation ne se modifie pas : supprime-la puis recrée-la.", is_error=True
            )
        self._store.update(record.id, args["content"])
        return ToolResult(f"Fait {record.id[:8]} mis à jour.")

    async def preview(self, args: dict[str, Any]) -> str | None:
        record = _resolve(self._store, args["id"])
        if not isinstance(record, Fact):
            return None
        return f"Modifier la mémoire du projet\n- {record.content}\n+ {args['content']}"


class MemoryDeleteTool:
    name = "memory_delete"
    description = "Delete a fact or a relation of the project memory that is wrong or obsolete."
    risk: Risk = "write"
    parameters = {"type": "object", "properties": {"id": _ID}, "required": ["id"]}

    def __init__(self, store: FactStore) -> None:
        self._store = store

    async def run(self, args: dict[str, Any], ctx: ToolContext) -> ToolResult:
        record = _resolve(self._store, args["id"])
        if isinstance(record, ToolResult):
            return record
        self._store.delete(record.id)
        return ToolResult(f"Supprimé de la mémoire : {record.id[:8]}.")

    async def preview(self, args: dict[str, Any]) -> str | None:
        record = _resolve(self._store, args["id"])
        return (
            None if isinstance(record, ToolResult) else f"Supprimer de la mémoire\n{record.line()}"
        )


def _resolve(store: FactStore, prefix: str) -> Record | ToolResult:
    matches = store.find(prefix)
    if not matches:
        return ToolResult(NOT_FOUND.format(prefix=prefix), is_error=True)
    if len(matches) > 1:
        return ToolResult(AMBIGUOUS.format(prefix=prefix), is_error=True)
    return matches[0]

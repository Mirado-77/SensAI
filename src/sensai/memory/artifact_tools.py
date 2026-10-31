"""Tool of M4: the model patches one section of the state document of the session.

The change has risk "write", so the user approves it after seeing `preview`. Only the named
section changes: the model never rewrites the whole document.
"""

from typing import Any

from sensai.core.contracts import EventSink
from sensai.core.events import ARTIFACT_PATCHED, make_event
from sensai.core.types import Risk, ToolContext, ToolResult
from sensai.memory.artifact import OPS, ArtifactStore, patched
from sensai.safety.untrusted import contains_untrusted_marker

UNKNOWN_SECTION = "Section inconnue : {section}. Sections possibles : {sections}."
TOO_LONG = (
    "La section {section} ferait {length} caractères, la limite est {limit}."
    " Résume-la avec op replace."
)
UNTRUSTED = "Le contenu recopie un bloc de données externes : écris la section avec tes mots."
UNCHANGED = "La section {section} n'a pas changé."
PATCHED = "Section {section} mise à jour ({length} caractères)."


class ArtifactPatchTool:
    name = "artifact_patch"
    description = (
        "Update one section of the state document of this session (goal, decisions, what is"
        " done, what is left). 'append' adds a line at the end of the section, 'replace'"
        " rewrites only that section. Call it when a decision is taken or a task ends."
    )
    risk: Risk = "write"

    def __init__(
        self, store: ArtifactStore, sink: EventSink, sections: list[str], max_chars: int
    ) -> None:
        self._store = store
        self._sink = sink
        self._sections = sections
        self._max_chars = max_chars
        self.parameters = {
            "type": "object",
            "properties": {
                "section": {"type": "string", "enum": list(sections)},
                "op": {"type": "string", "enum": list(OPS)},
                "content": {
                    "type": "string",
                    "minLength": 1,
                    "description": "Text to add or the new text of the section, in Markdown",
                },
            },
            "required": ["section", "op", "content"],
        }

    async def run(self, args: dict[str, Any], ctx: ToolContext) -> ToolResult:
        section, op, content = args["section"], args["op"], args["content"]
        error = self._check(ctx.session_id, section, op, content)
        if error:
            return ToolResult(error, is_error=True)
        before, after = self._store.apply(ctx.session_id, ctx.run_id, section, op, content)
        if after == before:
            return ToolResult(UNCHANGED.format(section=section))
        self._sink.emit(
            make_event(
                ARTIFACT_PATCHED,
                ctx.run_id,
                {
                    "session_id": ctx.session_id,
                    "section": section,
                    "op": op,
                    "chars": len(after),
                },
            )
        )
        return ToolResult(PATCHED.format(section=section, length=len(after)))

    async def preview(self, args: dict[str, Any]) -> str | None:
        verb = "Ajouter à" if args.get("op") == "append" else "Remplacer"
        section = args.get("section")
        return f"{verb} la section « {section} » du document d'état\n{args.get('content')}"

    def _check(self, session_id: str, section: str, op: str, content: str) -> str | None:
        if section not in self._sections:
            return UNKNOWN_SECTION.format(section=section, sections=", ".join(self._sections))
        if contains_untrusted_marker(content):
            return UNTRUSTED
        before = self._store.sections(session_id).get(section, "")
        length = len(patched(before, op, content))
        if length > self._max_chars:
            return TOO_LONG.format(section=section, length=length, limit=self._max_chars)
        return None

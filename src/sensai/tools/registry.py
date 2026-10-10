"""Registry of the tools available to the model, native or MCP."""

from collections.abc import Iterable

import jsonschema

from sensai.core.contracts import Tool
from sensai.core.schema import check_schema
from sensai.core.types import ToolSpec


class ToolRegistry:
    def __init__(self, tools: Iterable[Tool] = ()) -> None:
        self._tools: dict[str, Tool] = {}
        for tool in tools:
            self.register(tool)

    def register(self, tool: Tool) -> None:
        if tool.name in self._tools:
            raise ValueError(f"Outil déjà enregistré : {tool.name}")
        _check_parameters(tool)
        self._tools[tool.name] = tool

    def get(self, name: str) -> Tool | None:
        return self._tools.get(name)

    def specs(self) -> list[ToolSpec]:
        return [
            ToolSpec(tool.name, tool.description, tool.parameters) for tool in self._tools.values()
        ]

    def __len__(self) -> int:
        return len(self._tools)


def _check_parameters(tool: Tool) -> None:
    """Reject a tool whose `parameters` is not a JSON Schema of type "object" (T5)."""
    try:
        check_schema(tool.parameters)
    except jsonschema.SchemaError as exc:
        raise ValueError(f"Schéma de paramètres invalide pour {tool.name} : {exc.message}") from exc
    if tool.parameters.get("type") != "object":
        raise ValueError(f"Les paramètres de {tool.name} doivent être un schéma de type object.")

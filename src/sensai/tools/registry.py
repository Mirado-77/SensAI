"""Registry of the tools available to the model, native or MCP."""

from collections.abc import Iterable

from sensai.core.contracts import Tool
from sensai.core.types import ToolSpec


class ToolRegistry:
    def __init__(self, tools: Iterable[Tool] = ()) -> None:
        self._tools: dict[str, Tool] = {}
        for tool in tools:
            self.register(tool)

    def register(self, tool: Tool) -> None:
        if tool.name in self._tools:
            raise ValueError(f"Outil déjà enregistré : {tool.name}")
        self._tools[tool.name] = tool

    def get(self, name: str) -> Tool | None:
        return self._tools.get(name)

    def specs(self) -> list[ToolSpec]:
        return [
            ToolSpec(tool.name, tool.description, tool.parameters) for tool in self._tools.values()
        ]

    def __len__(self) -> int:
        return len(self._tools)

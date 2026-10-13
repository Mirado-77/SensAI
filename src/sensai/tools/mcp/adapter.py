"""Adapter from an MCP tool to the `Tool` contract (T1).

MCP tools then go through the same registry, argument validation, approval policy and event
journal as native tools. Their output always comes from outside SENSAI: it is marked untrusted.
"""

import json
import re
from typing import Any, Literal, Protocol

from sensai.core.types import Risk, ToolContext, ToolResult
from sensai.tools.mcp.client import McpError

RiskSetting = Risk | Literal["auto"]

_NAME_SEPARATOR = "__"
_MAX_NAME_LENGTH = 64
_INVALID_NAME_CHARS = re.compile(r"[^A-Za-z0-9_-]")


class ToolCaller(Protocol):
    async def call_tool(self, name: str, arguments: dict[str, Any]) -> dict[str, Any]: ...


class McpTool:
    """One tool of one MCP server, named `<server>__<tool>` for the model."""

    def __init__(
        self, server: str, spec: dict[str, Any], caller: ToolCaller, risk: RiskSetting
    ) -> None:
        self.remote_name = str(spec["name"])
        self.name = tool_name(server, self.remote_name)
        self.description = str(spec.get("description") or spec.get("title") or self.remote_name)
        self.parameters = _parameters(spec.get("inputSchema"))
        self.risk: Risk = _risk(risk, spec.get("annotations"))
        self._caller = caller

    async def run(self, args: dict[str, Any], ctx: ToolContext) -> ToolResult:
        try:
            result = await self._caller.call_tool(self.remote_name, args)
        except McpError as exc:
            return ToolResult(str(exc), is_error=True)
        return ToolResult(result_text(result), is_error=bool(result.get("isError")), untrusted=True)


def tool_name(server: str, remote_name: str) -> str:
    name = _INVALID_NAME_CHARS.sub("_", f"{server}{_NAME_SEPARATOR}{remote_name}")
    return name[:_MAX_NAME_LENGTH]


def result_text(result: dict[str, Any]) -> str:
    """Text of a `tools/call` result: text blocks, embedded text resources, placeholders."""
    parts = [_block_text(block) for block in result.get("content") or []]
    parts = [part for part in parts if part]
    if not parts and result.get("structuredContent") is not None:
        parts.append(json.dumps(result["structuredContent"], ensure_ascii=False))
    return "\n".join(parts)


def _block_text(block: Any) -> str:
    if not isinstance(block, dict):
        return ""
    kind = block.get("type")
    if kind == "text":
        return str(block.get("text", ""))
    if kind == "resource":
        resource = block.get("resource") or {}
        if "text" in resource:
            return str(resource["text"])
        return f"[resource {resource.get('uri', '')} omitted]"
    if kind == "resource_link":
        return f"[resource link {block.get('uri', '')}]"
    return f"[{kind} content omitted]"


def _parameters(schema: Any) -> dict[str, Any]:
    if not isinstance(schema, dict):
        return {"type": "object", "properties": {}}
    return {"type": "object", **schema} if "type" not in schema else dict(schema)


def _risk(setting: RiskSetting, annotations: Any) -> Risk:
    """With "auto", trust the server's `readOnlyHint`; otherwise treat the tool as an execution."""
    if setting != "auto":
        return setting
    read_only = isinstance(annotations, dict) and annotations.get("readOnlyHint") is True
    return "read" if read_only else "exec"

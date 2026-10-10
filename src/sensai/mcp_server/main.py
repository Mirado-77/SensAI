"""`sensai-mcp`: our own MCP server over stdio (T1), for Python code navigation in one root.

Tools, both read-only:
- python_symbols(path): classes, functions and methods of a Python file, with line numbers
- find_references(name): lines of the Python files that mention an identifier

Every path is resolved and must stay inside `--root` (no `..`, no symlink that leaves it).
Run it with `sensai-mcp --root <directory>`.
"""

import argparse
import ast
import json
import re
import sys
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Any, TextIO

PROTOCOL_VERSION = "2025-06-18"
SERVER_INFO = {"name": "sensai-mcp", "version": "0.1.0"}
MAX_REFERENCES = 50
_IDENTIFIER = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")
_SKIPPED_DIRS = {".git", ".venv", "venv", "__pycache__", "node_modules", ".tox"}
_READ_ONLY = {"readOnlyHint": True, "openWorldHint": False}

TOOLS = [
    {
        "name": "python_symbols",
        "description": (
            "List the classes, functions and methods of a Python file with their line numbers "
            "and signatures. `path` is relative to the project root."
        ),
        "inputSchema": {
            "type": "object",
            "properties": {"path": {"type": "string", "description": "Python file"}},
            "required": ["path"],
            "additionalProperties": False,
        },
        "annotations": _READ_ONLY,
    },
    {
        "name": "find_references",
        "description": (
            "Find the lines of the project's Python files that mention an identifier "
            f"(function, class, variable). At most {MAX_REFERENCES} results."
        ),
        "inputSchema": {
            "type": "object",
            "properties": {"name": {"type": "string", "description": "identifier to find"}},
            "required": ["name"],
            "additionalProperties": False,
        },
        "annotations": _READ_ONLY,
    },
]


class ToolFailure(Exception):
    """Expected failure of a tool, returned to the client with `isError`."""


class CodeTools:
    def __init__(self, root: Path) -> None:
        self.root = root.resolve()

    def python_symbols(self, path: str) -> str:
        file = self._resolve(path)
        if file.suffix != ".py" or not file.is_file():
            raise ToolFailure(f"Not a Python file: {path}")
        try:
            tree = ast.parse(file.read_text(encoding="utf-8"), filename=path)
        except SyntaxError as exc:
            raise ToolFailure(f"Syntax error in {path} line {exc.lineno}: {exc.msg}") from exc
        except (OSError, UnicodeDecodeError) as exc:
            raise ToolFailure(f"Cannot read {path}: {exc}") from exc
        lines = [line for node in tree.body for line in _describe(node, "")]
        return "\n".join(lines) if lines else f"No class or function in {path}."

    def find_references(self, name: str) -> str:
        if not _IDENTIFIER.match(name):
            raise ToolFailure(f"Not a Python identifier: {name!r}")
        pattern = re.compile(rf"\b{re.escape(name)}\b")
        hits: list[str] = []
        for file in self._python_files():
            try:
                text = file.read_text(encoding="utf-8")
            except (OSError, UnicodeDecodeError):
                continue
            relative = file.relative_to(self.root).as_posix()
            for number, line in enumerate(text.splitlines(), start=1):
                if pattern.search(line):
                    hits.append(f"{relative}:{number}: {line.strip()}")
                    if len(hits) == MAX_REFERENCES:
                        return "\n".join([*hits, f"(stopped at {MAX_REFERENCES} results)"])
        return "\n".join(hits) if hits else f"No reference to {name}."

    def _resolve(self, path: str) -> Path:
        candidate = (self.root / path).resolve()
        if not candidate.is_relative_to(self.root):
            raise ToolFailure(f"Path outside the project root: {path}")
        return candidate

    def _python_files(self) -> list[Path]:
        files = []
        for file in sorted(self.root.rglob("*.py")):
            parts = file.relative_to(self.root).parts
            if _SKIPPED_DIRS.intersection(parts):
                continue
            if file.resolve().is_relative_to(self.root):
                files.append(file)
        return files


def _describe(node: ast.stmt, indent: str) -> list[str]:
    if isinstance(node, ast.ClassDef):
        lines = [f"{indent}class {node.name} (line {node.lineno})"]
        for child in node.body:
            lines.extend(_describe(child, indent + "  "))
        return lines
    if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
        prefix = "async def" if isinstance(node, ast.AsyncFunctionDef) else "def"
        signature = ast.unparse(node.args)
        returns = f" -> {ast.unparse(node.returns)}" if node.returns else ""
        line = f"{indent}{prefix} {node.name}({signature}){returns} (line {node.lineno})"
        return [line]
    return []


class McpServer:
    """Minimal JSON-RPC 2.0 server: one message per line on stdin, answers on stdout."""

    def __init__(self, tools: CodeTools, output: TextIO) -> None:
        self._output = output
        self._handlers: dict[str, Callable[..., str]] = {
            "python_symbols": tools.python_symbols,
            "find_references": tools.find_references,
        }

    def serve(self, lines: TextIO) -> None:
        for line in lines:
            if line.strip():
                self.handle_line(line)

    def handle_line(self, line: str) -> None:
        try:
            message = json.loads(line)
        except json.JSONDecodeError:
            self._send({"jsonrpc": "2.0", "id": None, "error": _error(-32700, "Parse error")})
            return
        if not isinstance(message, dict) or "id" not in message:
            return  # notifications (notifications/initialized...) need no answer
        reply: dict[str, Any] = {"jsonrpc": "2.0", "id": message["id"]}
        try:
            reply["result"] = self._dispatch(message.get("method"), message.get("params") or {})
        except _RpcError as exc:
            reply["error"] = _error(exc.code, str(exc))
        self._send(reply)

    def _dispatch(self, method: Any, params: dict[str, Any]) -> dict[str, Any]:
        if method == "initialize":
            return {
                "protocolVersion": PROTOCOL_VERSION,
                "capabilities": {"tools": {}},
                "serverInfo": SERVER_INFO,
            }
        if method == "ping":
            return {}
        if method == "tools/list":
            return {"tools": TOOLS}
        if method == "tools/call":
            return self._call(params)
        raise _RpcError(-32601, f"Method not found: {method}")

    def _call(self, params: dict[str, Any]) -> dict[str, Any]:
        handler = self._handlers.get(params.get("name", ""))
        if handler is None:
            raise _RpcError(-32602, f"Unknown tool: {params.get('name')}")
        arguments = params.get("arguments") or {}
        try:
            text = handler(**arguments)
        except TypeError as exc:
            return _tool_result(f"Invalid arguments: {exc}", is_error=True)
        except ToolFailure as exc:
            return _tool_result(str(exc), is_error=True)
        return _tool_result(text)

    def _send(self, message: dict[str, Any]) -> None:
        self._output.write(json.dumps(message, ensure_ascii=False) + "\n")
        self._output.flush()


class _RpcError(Exception):
    def __init__(self, code: int, message: str) -> None:
        super().__init__(message)
        self.code = code


def _error(code: int, message: str) -> dict[str, Any]:
    return {"code": code, "message": message}


def _tool_result(text: str, is_error: bool = False) -> dict[str, Any]:
    return {"content": [{"type": "text", "text": text}], "isError": is_error}


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="sensai-mcp", description="SENSAI MCP server: Python code navigation over stdio."
    )
    parser.add_argument("--root", type=Path, default=Path.cwd(), help="project root")
    args = parser.parse_args(argv)
    if not args.root.is_dir():
        print(f"sensai-mcp: not a directory: {args.root}", file=sys.stderr)
        return 2
    McpServer(CodeTools(args.root), sys.stdout).serve(sys.stdin)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

"""MCP client over stdio, written by hand (T1): JSON-RPC 2.0, one message per line.

Only what SENSAI needs: `initialize`, `notifications/initialized`, `tools/list` (with
pagination) and `tools/call`. Requests sent by the server are answered with "method not found",
except `ping`. Notifications from the server are ignored.
"""

import asyncio
import contextlib
import itertools
import json
import logging
import os
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from sensai.core.errors import SensaiError

logger = logging.getLogger(__name__)

PROTOCOL_VERSION = "2025-06-18"
CLIENT_INFO = {"name": "sensai", "version": "0.1.0"}
_STREAM_LIMIT = 16 * 1024 * 1024
_METHOD_NOT_FOUND = -32601


class McpError(SensaiError):
    """The MCP server could not be started, stopped answering, or returned an error."""


class McpClient:
    def __init__(
        self,
        name: str,
        command: str,
        args: Sequence[str] = (),
        env: Mapping[str, str] | None = None,
        cwd: Path | None = None,
        timeout_s: float = 30,
    ) -> None:
        self.name = name
        self._command = command
        self._args = list(args)
        self._env = dict(env or {})
        self._cwd = cwd
        self._timeout_s = timeout_s
        self._process: asyncio.subprocess.Process | None = None
        self._reader: asyncio.Task[None] | None = None
        self._stderr: asyncio.Task[None] | None = None
        self._pending: dict[int, asyncio.Future[Any]] = {}
        self._ids = itertools.count(1)
        self._write_lock = asyncio.Lock()
        self.server_info: dict[str, Any] = {}

    async def start(self) -> None:
        """Launch the server and run the initialization handshake."""
        try:
            self._process = await asyncio.create_subprocess_exec(
                self._command,
                *self._args,
                stdin=asyncio.subprocess.PIPE,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                env={**os.environ, **self._env},
                cwd=self._cwd,
                limit=_STREAM_LIMIT,
            )
        except OSError as exc:
            raise McpError(f"Serveur MCP {self.name} impossible à lancer : {exc}") from exc
        self._reader = asyncio.create_task(self._read_loop())
        self._stderr = asyncio.create_task(self._drain_stderr())
        try:
            result = await self.request(
                "initialize",
                {
                    "protocolVersion": PROTOCOL_VERSION,
                    "capabilities": {},
                    "clientInfo": CLIENT_INFO,
                },
            )
            self.server_info = dict(result.get("serverInfo") or {})
            await self._send({"jsonrpc": "2.0", "method": "notifications/initialized"})
        except BaseException:
            await self.aclose()
            raise

    async def list_tools(self) -> list[dict[str, Any]]:
        tools: list[dict[str, Any]] = []
        cursor: str | None = None
        while True:
            params = {"cursor": cursor} if cursor else {}
            result = await self.request("tools/list", params)
            tools.extend(result.get("tools") or [])
            cursor = result.get("nextCursor")
            if not cursor:
                return tools

    async def call_tool(self, name: str, arguments: dict[str, Any]) -> dict[str, Any]:
        return await self.request("tools/call", {"name": name, "arguments": arguments})

    async def request(self, method: str, params: dict[str, Any]) -> dict[str, Any]:
        """Send a request and wait for its result, at most `timeout_s` seconds."""
        if self._process is None or self._reader is None or self._reader.done():
            raise McpError(f"Serveur MCP {self.name} arrêté.")
        request_id = next(self._ids)
        future: asyncio.Future[Any] = asyncio.get_running_loop().create_future()
        self._pending[request_id] = future
        try:
            message = {"jsonrpc": "2.0", "id": request_id, "method": method, "params": params}
            await self._send(message)
            result = await asyncio.wait_for(future, self._timeout_s)
        except asyncio.TimeoutError:
            raise McpError(
                f"Serveur MCP {self.name} : pas de réponse à {method} après {self._timeout_s} s."
            ) from None
        finally:
            self._pending.pop(request_id, None)
        if not isinstance(result, dict):
            raise McpError(f"Serveur MCP {self.name} : résultat invalide pour {method}.")
        return result

    async def aclose(self) -> None:
        """Close stdin, then terminate, then kill the server if it does not exit."""
        process, self._process = self._process, None
        if process is not None and process.returncode is None:
            if process.stdin is not None:
                with contextlib.suppress(OSError, RuntimeError):
                    process.stdin.close()
            await _wait_or_stop(process)
        for task in (self._reader, self._stderr):
            if task is not None:
                task.cancel()
                with contextlib.suppress(asyncio.CancelledError):
                    await task
        self._fail_pending(McpError(f"Serveur MCP {self.name} arrêté."))

    async def _send(self, message: dict[str, Any]) -> None:
        process = self._process
        if process is None or process.stdin is None:
            raise McpError(f"Serveur MCP {self.name} arrêté.")
        data = json.dumps(message, ensure_ascii=False).encode() + b"\n"
        async with self._write_lock:
            try:
                process.stdin.write(data)
                await process.stdin.drain()
            except (ConnectionError, RuntimeError) as exc:
                raise McpError(f"Serveur MCP {self.name} injoignable : {exc}") from exc

    async def _read_loop(self) -> None:
        assert self._process is not None and self._process.stdout is not None
        stdout = self._process.stdout
        error = McpError(f"Serveur MCP {self.name} arrêté de façon inattendue.")
        try:
            while line := await stdout.readline():
                await self._dispatch(line)
        except (ValueError, ConnectionError) as exc:
            error = McpError(f"Serveur MCP {self.name} : flux illisible ({exc}).")
        self._fail_pending(error)

    async def _dispatch(self, line: bytes) -> None:
        try:
            message = json.loads(line)
        except json.JSONDecodeError:
            logger.warning("MCP %s: ignored non-JSON line %r", self.name, line[:200])
            return
        if not isinstance(message, dict):
            return
        if "method" in message:
            if "id" in message:
                await self._answer_server_request(message)
            return
        future = self._pending.get(message.get("id"))
        if future is None or future.done():
            return
        if "error" in message:
            error = message["error"] or {}
            detail = error.get("message", "erreur inconnue") if isinstance(error, dict) else error
            future.set_exception(McpError(f"Serveur MCP {self.name} : {detail}"))
        else:
            future.set_result(message.get("result"))

    async def _answer_server_request(self, message: dict[str, Any]) -> None:
        reply: dict[str, Any] = {"jsonrpc": "2.0", "id": message["id"]}
        if message["method"] == "ping":
            reply["result"] = {}
        else:
            reply["error"] = {"code": _METHOD_NOT_FOUND, "message": "Method not found"}
        with contextlib.suppress(McpError):
            await self._send(reply)

    async def _drain_stderr(self) -> None:
        assert self._process is not None and self._process.stderr is not None
        while line := await self._process.stderr.readline():
            logger.debug("MCP %s stderr: %s", self.name, line.decode(errors="replace").rstrip())

    def _fail_pending(self, error: McpError) -> None:
        for future in self._pending.values():
            if not future.done():
                future.set_exception(error)


async def _wait_or_stop(process: asyncio.subprocess.Process, grace_s: float = 2) -> None:
    for stop in (None, process.terminate, process.kill):
        if stop is not None:
            with contextlib.suppress(ProcessLookupError):
                stop()
        try:
            await asyncio.wait_for(process.wait(), grace_s)
            return
        except asyncio.TimeoutError:
            continue

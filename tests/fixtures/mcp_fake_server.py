"""Scripted MCP server over stdio, for the T1 tests. Usage: python mcp_fake_server.py MODE.

Modes:
- normal: two pages of tools (echo, fail, readonly), echo returns its arguments
- crash_on_call: exits as soon as a tool is called
- hang_on_call: never answers tools/call
- init_error: answers initialize with a JSON-RPC error
- ping_first: sends a ping request to the client before answering tools/list
"""

import json
import sys

MODE = sys.argv[1] if len(sys.argv) > 1 else "normal"

PAGE_1 = [
    {
        "name": "echo",
        "description": "Return the arguments.",
        "inputSchema": {
            "type": "object",
            "properties": {"text": {"type": "string"}},
            "required": ["text"],
        },
    },
    {"name": "fail", "description": "Always fails.", "inputSchema": {"type": "object"}},
]
PAGE_2 = [
    {
        "name": "readonly",
        "description": "Read-only tool.",
        "inputSchema": {"type": "object", "properties": {}},
        "annotations": {"readOnlyHint": True},
    }
]


def send(message):
    sys.stdout.write(json.dumps(message) + "\n")
    sys.stdout.flush()


def reply(request, result):
    send({"jsonrpc": "2.0", "id": request["id"], "result": result})


def handle(request):
    method = request.get("method")
    if "id" not in request:
        return
    if method == "initialize":
        if MODE == "init_error":
            error = {"code": -32603, "message": "init refused"}
            send({"jsonrpc": "2.0", "id": request["id"], "error": error})
            return
        info = {"name": "fake", "version": "1"}
        reply(request, {"protocolVersion": "2025-06-18", "capabilities": {}, "serverInfo": info})
    elif method == "tools/list":
        if MODE == "ping_first":
            send({"jsonrpc": "2.0", "id": "srv-1", "method": "ping"})
            sys.stderr.write(f"pong: {sys.stdin.readline().strip()}\n")
            sys.stderr.flush()
        cursor = (request.get("params") or {}).get("cursor")
        if cursor == "page2":
            reply(request, {"tools": PAGE_2})
        else:
            reply(request, {"tools": PAGE_1, "nextCursor": "page2"})
    elif method == "tools/call":
        if MODE == "crash_on_call":
            sys.exit(3)
        if MODE == "hang_on_call":
            return
        params = request["params"]
        if params["name"] == "fail":
            content = [{"type": "text", "text": "it failed"}]
            reply(request, {"content": content, "isError": True})
        elif params["name"] == "echo":
            text = json.dumps(params["arguments"], sort_keys=True)
            reply(request, {"content": [{"type": "text", "text": text}]})
        else:
            reply(request, {"content": [{"type": "text", "text": "ok"}]})
    else:
        error = {"code": -32601, "message": f"Method not found: {method}"}
        send({"jsonrpc": "2.0", "id": request["id"], "error": error})


for line in sys.stdin:
    if line.strip():
        handle(json.loads(line))

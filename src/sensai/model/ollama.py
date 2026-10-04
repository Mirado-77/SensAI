"""Ollama gateway: streams `/api/chat` over HTTP and maps it to ModelEvents."""

import json
from collections.abc import AsyncIterator
from typing import Any

import httpx

from sensai.core.errors import ModelConnectionError, ModelError, ModelNotFoundError
from sensai.core.types import Chunk, Done, Message, ModelEvent, ToolCall, ToolSpec, Usage


class OllamaGateway:
    def __init__(self, host: str, model: str, timeout_s: float) -> None:
        self._host = host.rstrip("/")
        self._model = model
        self._timeout = httpx.Timeout(timeout_s)

    async def stream_chat(
        self,
        messages: list[Message],
        tools: list[ToolSpec] | None = None,
        schema: dict[str, Any] | None = None,
    ) -> AsyncIterator[ModelEvent]:
        payload = _build_payload(self._model, messages, tools, schema)
        try:
            async with httpx.AsyncClient(base_url=self._host, timeout=self._timeout) as client:
                async with client.stream("POST", "/api/chat", json=payload) as response:
                    if response.status_code != 200:
                        raise await _status_error(response, self._model)
                    finished = False
                    async for line in response.aiter_lines():
                        if not line.strip():
                            continue
                        data = _parse_line(line, self._model)
                        for event in _events_from(data):
                            yield event
                        if data.get("done"):
                            finished = True
                            break
                    if not finished:
                        raise ModelConnectionError(
                            "La connexion avec Ollama s'est interrompue avant la fin de la réponse."
                        )
        except httpx.TransportError as exc:
            detail = str(exc) or type(exc).__name__
            raise ModelConnectionError(f"Ollama est injoignable à {self._host} : {detail}") from exc
        yield Done()


def _build_payload(
    model: str,
    messages: list[Message],
    tools: list[ToolSpec] | None,
    schema: dict[str, Any] | None,
) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "model": model,
        "messages": [_message_payload(message) for message in messages],
        "stream": True,
    }
    if tools:
        payload["tools"] = [
            {
                "type": "function",
                "function": {
                    "name": tool.name,
                    "description": tool.description,
                    "parameters": tool.parameters,
                },
            }
            for tool in tools
        ]
    if schema is not None:
        payload["format"] = schema
    return payload


def _message_payload(message: Message) -> dict[str, Any]:
    payload: dict[str, Any] = {"role": message.role, "content": message.content}
    if message.tool_calls:
        payload["tool_calls"] = [
            {"function": {"name": call.name, "arguments": call.args}} for call in message.tool_calls
        ]
    if message.role == "tool" and message.tool_name:
        payload["tool_name"] = message.tool_name
    return payload


def _parse_line(line: str, model: str) -> dict[str, Any]:
    try:
        data = json.loads(line)
    except json.JSONDecodeError as exc:
        raise ModelError(f"Réponse invalide d'Ollama : {line[:200]!r}") from exc
    if not isinstance(data, dict):
        raise ModelError(f"Réponse invalide d'Ollama : {line[:200]!r}")
    if "error" in data:
        raise _error_from_detail(str(data["error"]), model)
    return data


def _events_from(data: dict[str, Any]) -> list[ModelEvent]:
    events: list[ModelEvent] = []
    try:
        message = data.get("message") or {}
        if message.get("content"):
            events.append(Chunk(message["content"]))
        for call in message.get("tool_calls") or []:
            function = call["function"]
            events.append(ToolCall(function["name"], function.get("arguments") or {}))
        if data.get("done") and ("prompt_eval_count" in data or "eval_count" in data):
            events.append(Usage(data.get("prompt_eval_count", 0), data.get("eval_count", 0)))
    except (KeyError, TypeError, AttributeError) as exc:
        raise ModelError(f"Réponse invalide d'Ollama : {json.dumps(data)[:200]}") from exc
    return events


async def _status_error(response: httpx.Response, model: str) -> ModelError:
    body = (await response.aread()).decode("utf-8", errors="replace")
    try:
        detail = str(json.loads(body)["error"])
    except (json.JSONDecodeError, KeyError, TypeError):
        detail = body.strip()
    return _error_from_detail(detail, model, status=response.status_code)


def _error_from_detail(detail: str, model: str, status: int | None = None) -> ModelError:
    if "not found" in detail.lower() and status in (None, 404):
        return ModelNotFoundError(f"Modèle introuvable sur Ollama : {model!r} ({detail})")
    suffix = f" (statut {status})" if status else ""
    return ModelError(f"Erreur d'Ollama{suffix} : {detail}")

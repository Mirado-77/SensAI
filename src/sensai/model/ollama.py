"""Ollama gateway: streams `/api/chat` over HTTP and maps it to ModelEvents."""

import json
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from typing import Any

import httpx

from sensai.core.errors import ModelConnectionError, ModelError, ModelNotFoundError
from sensai.core.types import Chunk, Done, Message, ModelEvent, ToolCall, ToolSpec, Usage

_CUT_MESSAGE = "La connexion avec Ollama s'est interrompue avant la fin de la réponse."


class OllamaGateway:
    def __init__(
        self, host: str, model: str, timeout_s: float, temperature: float, think: bool = False
    ) -> None:
        self._host = host.rstrip("/")
        self._model = model
        self._timeout = httpx.Timeout(timeout_s)
        self._think = think
        self._temperature = temperature

    async def stream_chat(
        self,
        messages: list[Message],
        tools: list[ToolSpec] | None = None,
        schema: dict[str, Any] | None = None,
    ) -> AsyncIterator[ModelEvent]:
        payload = _build_payload(
            self._model, messages, tools, schema, self._temperature, self._think
        )
        try:
            async with self._open(payload) as response:
                done = False
                async for line in response.aiter_lines():
                    events, done = _parse_events(line, self._model)
                    for event in events:
                        yield event
                    if done:
                        break
                if not done:
                    raise ModelConnectionError(_CUT_MESSAGE)
        except httpx.TransportError as exc:
            detail = str(exc) or type(exc).__name__
            raise ModelConnectionError(f"Ollama est injoignable à {self._host} : {detail}") from exc
        yield Done()

    @asynccontextmanager
    async def _open(self, payload: dict[str, Any]) -> AsyncIterator[httpx.Response]:
        async with httpx.AsyncClient(base_url=self._host, timeout=self._timeout) as client:
            async with client.stream("POST", "/api/chat", json=payload) as response:
                if response.status_code != 200:
                    raise await _status_error(response, self._model)
                yield response


class OllamaEmbedder:
    """Embeddings of texts with `/api/embed` (R1)."""

    def __init__(self, host: str, model: str, timeout_s: float) -> None:
        self._host = host.rstrip("/")
        self.model = model
        self._timeout = httpx.Timeout(timeout_s)

    async def embed(self, texts: list[str]) -> list[list[float]]:
        payload = {"model": self.model, "input": texts}
        try:
            async with httpx.AsyncClient(base_url=self._host, timeout=self._timeout) as client:
                response = await client.post("/api/embed", json=payload)
        except httpx.TransportError as exc:
            detail = str(exc) or type(exc).__name__
            raise ModelConnectionError(f"Ollama est injoignable à {self._host} : {detail}") from exc
        if response.status_code != 200:
            raise await _status_error(response, self.model)
        try:
            embeddings = response.json()["embeddings"]
        except (json.JSONDecodeError, KeyError, TypeError) as exc:
            raise _invalid_response(response.text) from exc
        if not isinstance(embeddings, list) or len(embeddings) != len(texts):
            raise _invalid_response(response.text)
        return embeddings


def _build_payload(
    model: str,
    messages: list[Message],
    tools: list[ToolSpec] | None,
    schema: dict[str, Any] | None,
    temperature: float,
    think: bool = False,
) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "model": model,
        "messages": [_message_payload(message) for message in messages],
        "stream": True,
        # Thinking models (qwen3) spend the completion budget on hidden reasoning otherwise.
        "think": think,
        "options": {"temperature": temperature},
    }
    if tools:
        payload["tools"] = [_tool_payload(tool) for tool in tools]
    if schema is not None:
        payload["format"] = schema
    return payload


def _tool_payload(tool: ToolSpec) -> dict[str, Any]:
    function = {
        "name": tool.name,
        "description": tool.description,
        "parameters": tool.parameters,
    }
    return {"type": "function", "function": function}


def _message_payload(message: Message) -> dict[str, Any]:
    payload: dict[str, Any] = {"role": message.role, "content": message.content}
    if message.tool_calls:
        payload["tool_calls"] = [
            {"function": {"name": call.name, "arguments": call.args}} for call in message.tool_calls
        ]
    if message.role == "tool" and message.tool_name:
        payload["tool_name"] = message.tool_name
    return payload


def _parse_events(line: str, model: str) -> tuple[list[ModelEvent], bool]:
    """Events of one NDJSON line, and whether it is the last line of the response."""
    if not line.strip():
        return [], False
    data = _parse_line(line, model)
    return _events_from(data), bool(data.get("done"))


def _parse_line(line: str, model: str) -> dict[str, Any]:
    try:
        data = json.loads(line)
    except json.JSONDecodeError as exc:
        raise _invalid_response(line) from exc
    if not isinstance(data, dict):
        raise _invalid_response(line)
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
        raise _invalid_response(json.dumps(data)) from exc
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


def _invalid_response(text: str) -> ModelError:
    return ModelError(f"Réponse invalide d'Ollama : {text[:200]!r}")

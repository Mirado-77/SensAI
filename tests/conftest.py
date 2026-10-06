"""Shared test helpers: NDJSON streams as Ollama would send them."""

import asyncio
import json
from collections.abc import Callable, Sequence
from types import SimpleNamespace
from typing import Any

import pytest

from sensai.core.types import Chunk, ModelEvent, ToolCall, Usage

OLLAMA_HOST = "http://ollama.test"


def _line(message: dict[str, Any], **fields: Any) -> bytes:
    body = {
        "model": "test-model",
        "message": {"role": "assistant", "content": "", **message},
        **fields,
    }
    return (json.dumps(body) + "\n").encode()


def ndjson_lines(events: Sequence[ModelEvent]) -> list[bytes]:
    """Lines Ollama would send for `events`. A trailing Usage becomes the final line's counts."""
    events = list(events)
    usage = events.pop() if events and isinstance(events[-1], Usage) else None
    lines = []
    for event in events:
        if isinstance(event, Chunk):
            lines.append(_line({"content": event.text}, done=False))
        elif isinstance(event, ToolCall):
            call = {"function": {"name": event.name, "arguments": event.args}}
            lines.append(_line({"tool_calls": [call]}, done=False))
        else:
            raise ValueError(f"Cannot encode {event!r}")
    final: dict[str, Any] = {"done": True, "done_reason": "stop"}
    if usage is not None:
        final["prompt_eval_count"] = usage.prompt_tokens
        final["eval_count"] = usage.completion_tokens
    lines.append(_line({}, **final))
    return lines


@pytest.fixture
def ollama_host() -> str:
    return OLLAMA_HOST


@pytest.fixture
def ollama_lines() -> Callable[[Sequence[ModelEvent]], list[bytes]]:
    return ndjson_lines


@pytest.fixture
def slow_body() -> Callable[[list[bytes]], tuple[Any, SimpleNamespace]]:
    """Build an async body that yields control between lines, and a probe set once closed."""

    def make(lines: list[bytes]) -> tuple[Any, SimpleNamespace]:
        probe = SimpleNamespace(closed=False)

        async def body():
            try:
                for line in lines:
                    await asyncio.sleep(0)
                    yield line
            finally:
                probe.closed = True

        return body(), probe

    return make

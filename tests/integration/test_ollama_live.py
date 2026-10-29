"""Tests against a real Ollama server. Run them with `make test-integration`.

They are skipped when no server answers or when the model is not installed. Set
SENSAI_TEST_MODEL to use a model other than the configured one.
"""

import io
import os

import httpx
import pytest
from rich.console import Console

from sensai.cli.chat import run_chat
from sensai.cli.terminal import Terminal
from sensai.core.config import load_config
from sensai.core.errors import ModelConnectionError, ModelNotFoundError
from sensai.core.types import Chunk, Done, Message
from sensai.model.ollama import OllamaGateway

pytestmark = [pytest.mark.integration, pytest.mark.feature("B1")]


@pytest.fixture
def config(tmp_path):
    base = load_config()
    try:
        httpx.get(f"{base.model.host}/api/version", timeout=2).raise_for_status()
    except httpx.HTTPError:
        pytest.skip(f"No Ollama server at {base.model.host}")
    model = os.environ.get("SENSAI_TEST_MODEL", base.model.name)
    storage = {"db_path": str(tmp_path / "sensai.db")}
    return load_config(overrides={"model": {"name": model, "timeout_s": 180}, "storage": storage})


@pytest.fixture
def gateway(config):
    return OllamaGateway(
        config.model.host, config.model.name, config.model.timeout_s, config.model.temperature
    )


async def stream(gateway):
    messages = [Message("user", "Reply with one word: hello")]
    return [event async for event in gateway.stream_chat(messages)]


async def collect_or_skip(gateway):
    try:
        return await stream(gateway)
    except ModelNotFoundError:
        pytest.skip("The model is not installed: pull it or set SENSAI_TEST_MODEL")


async def test_streams_a_real_answer(gateway):
    events = await collect_or_skip(gateway)
    assert any(isinstance(event, Chunk) for event in events)
    assert events[-1] == Done()


async def test_a_missing_model_is_reported(config):
    gateway = OllamaGateway(
        config.model.host, "does-not-exist:1b", config.model.temperature, timeout_s=10
    )
    with pytest.raises(ModelNotFoundError):
        await stream(gateway)


async def test_an_unreachable_server_is_reported():
    gateway = OllamaGateway("http://localhost:9", "any", config.model.temperature, timeout_s=5)
    with pytest.raises(ModelConnectionError):
        await stream(gateway)


async def test_a_full_chat_turn_with_the_real_model(config, gateway):
    await collect_or_skip(gateway)
    lines = iter(["Reply with one word: hello", "/exit"])
    out, err = io.StringIO(), io.StringIO()
    console = Console(file=out, color_system=None, force_terminal=False, highlight=False)
    errors = Console(file=err, color_system=None, force_terminal=False, highlight=False)
    terminal = Terminal(console, errors, lambda prompt: next(lines))
    code = await run_chat(config, terminal)
    assert code == 0
    assert "sensai> " in out.getvalue()
    assert err.getvalue() == ""

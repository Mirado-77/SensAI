import pytest
from tests.unit.test_cli import chat_setup, read_root_override

from sensai import tools
from sensai.cli.chat import run_chat
from sensai.core.config import load_config
from sensai.core.fakes import FakeModelGateway
from sensai.core.types import Chunk

pytestmark = pytest.mark.feature("T2")


def _fake_reason(reason):
    async def unavailable_reason(image):
        return reason

    return unavailable_reason


def _config(tmp_path, enabled=True):
    root = tmp_path / "root"
    root.mkdir()
    config, terminal, _, err = chat_setup(
        tmp_path, "hi", files=read_root_override(root), sandbox={"enabled": enabled}
    )
    return config, terminal, err


async def test_run_command_is_offered_when_docker_is_ready(tmp_path, monkeypatch):
    monkeypatch.setattr(tools, "docker_unavailable_reason", _fake_reason(None))
    config, terminal, err = _config(tmp_path)
    gateway = FakeModelGateway([Chunk("hello")])
    assert await run_chat(config, terminal, gateway) == 0
    assert "run_command" in [spec.name for spec in gateway.calls[0].tools]
    assert err.getvalue() == ""


async def test_chat_starts_without_run_command_and_says_why(tmp_path, monkeypatch):
    monkeypatch.setattr(tools, "docker_unavailable_reason", _fake_reason("Docker est introuvable"))
    config, terminal, err = _config(tmp_path)
    gateway = FakeModelGateway([Chunk("hello")])
    assert await run_chat(config, terminal, gateway) == 0
    assert "run_command" not in [spec.name for spec in gateway.calls[0].tools]
    assert "désactivée (T2)" in err.getvalue()
    assert "Docker est introuvable" in err.getvalue()


async def test_disabled_sandbox_does_not_check_docker(tmp_path, monkeypatch):
    async def must_not_be_called(image):
        raise AssertionError("docker checked")

    monkeypatch.setattr(tools, "docker_unavailable_reason", must_not_be_called)
    config, _, _ = _config(tmp_path, enabled=False)
    assert await tools.sandbox_tools(config) == ([], None)


async def test_no_allowed_root_means_no_sandbox_tool(monkeypatch):
    monkeypatch.setattr(tools, "docker_unavailable_reason", _fake_reason(None))
    assert await tools.sandbox_tools(load_config()) == ([], None)

import io
import sys
from pathlib import Path

import pytest
from rich.console import Console

from sensai.cli.chat import run_chat
from sensai.cli.terminal import Terminal
from sensai.core.config import load_config
from sensai.core.fakes import FakeModelGateway
from sensai.core.types import Chunk, ToolCall

pytestmark = pytest.mark.feature("T1")

FAKE_SERVER = Path(__file__).resolve().parents[1] / "fixtures" / "mcp_fake_server.py"


def chat_setup(tmp_path, *lines, **overrides):
    storage = {"db_path": str(tmp_path / "sensai.db")}
    # A1 planning, X4 questions and the A4 base prompt are off here so that each script is one
    # model call with only the user's messages; their own test files turn them on.
    react = {"enabled": False, "max_plan_steps": 5}
    questions = {"enabled": False, "max_per_turn": 2}
    # T3 is off so that no test depends on a SearXNG instance; test_web_search covers it.
    web = {"enabled": False}
    # X5 is off so that the tool list stays the one of each test; test_schedule covers it.
    prompts = {"system": "", "persona": ""}
    scheduler = {"enabled": False}
    # M3 is off for the same reason; test_memory covers it.
    memory = {"enabled": False}
    # M4 too; test_artifact covers it.
    artifact = {"enabled": False}
    defaults = {
        "storage": storage,
        "react": react,
        "questions": questions,
        "web": web,
        "prompts": prompts,
        "scheduler": scheduler,
        "memory": memory,
        "artifact": artifact,
    }
    config = load_config(overrides={**defaults, **overrides})
    out, err = io.StringIO(), io.StringIO()
    remaining = iter(lines)

    def read_line(prompt):
        try:
            return next(remaining)
        except StopIteration:
            raise EOFError from None

    def console(buffer):
        return Console(file=buffer, width=200, color_system=None, highlight=False)

    return config, Terminal(console(out), console(err), read_line), out, err


def code_server(root):
    return {
        "name": "code",
        "command": sys.executable,
        "args": ["-m", "sensai.mcp_server.main", "--root", str(root)],
        "risk": "auto",
    }


@pytest.fixture
def project(tmp_path):
    root = tmp_path / "project"
    root.mkdir()
    (root / "calc.py").write_text("def mean(values):\n    return sum(values)\n")
    return root


async def test_the_model_calls_an_mcp_tool_during_a_turn(tmp_path, project):
    mcp = {"servers": [code_server(project)]}
    config, terminal, out, err = chat_setup(tmp_path, "What is in calc.py?", "/exit", mcp=mcp)
    call = ToolCall("code__python_symbols", {"path": "calc.py"})
    gateway = FakeModelGateway([call], [Chunk("calc.py defines mean.")])

    assert await run_chat(config, terminal, gateway) == 0

    assert "2 tools (/tools)" in out.getvalue()
    tool_names = [spec.name for spec in gateway.calls[0].tools]
    assert tool_names == ["code__python_symbols", "code__find_references"]
    tool_message = gateway.calls[1].messages[-1]
    assert tool_message.role == "tool"
    assert "def mean(values) (line 1)" in tool_message.content
    assert "untrusted" in tool_message.content.lower()
    assert "sensai> calc.py defines mean." in out.getvalue()
    assert err.getvalue() == ""


async def test_tools_command_lists_the_tools_with_their_risk(tmp_path, project):
    mcp = {"servers": [code_server(project)]}
    config, terminal, out, _ = chat_setup(tmp_path, "/tools", "/exit", mcp=mcp)
    gateway = FakeModelGateway()

    await run_chat(config, terminal, gateway)

    assert gateway.calls == []
    assert "code__python_symbols [read] List the classes" in out.getvalue()
    assert "code__find_references [read]" in out.getvalue()


async def test_tools_command_without_tools(tmp_path):
    config, terminal, out, _ = chat_setup(tmp_path, "/tools", "/exit")
    await run_chat(config, terminal, FakeModelGateway())
    assert "No tool available." in out.getvalue()
    assert "no tools" in out.getvalue()


async def test_a_failing_server_is_reported_and_the_chat_still_works(tmp_path, project):
    broken = {"name": "broken", "command": sys.executable, "args": [str(FAKE_SERVER), "init_error"]}
    mcp = {"servers": [broken, code_server(project)]}
    config, terminal, out, err = chat_setup(tmp_path, "hello", "/exit", mcp=mcp)
    gateway = FakeModelGateway([Chunk("hi")])

    assert await run_chat(config, terminal, gateway) == 0

    assert "Attention : Serveur MCP broken : init refused" in err.getvalue()
    assert "sensai> hi" in out.getvalue()
    assert len(gateway.calls[0].tools) == 2


async def test_mcp_tools_follow_the_approval_policy(tmp_path, project):
    exec_server = {**code_server(project), "risk": "exec"}
    mcp = {"servers": [exec_server]}
    config, terminal, out, err = chat_setup(tmp_path, "look", "n", "/exit", mcp=mcp)
    call = ToolCall("code__python_symbols", {"path": "calc.py"})
    gateway = FakeModelGateway([call], [Chunk("ok")])

    await run_chat(config, terminal, gateway)

    assert "Approval required (exec): code__python_symbols" in out.getvalue()
    assert gateway.calls[1].messages[-1].content == "The user refused this action."

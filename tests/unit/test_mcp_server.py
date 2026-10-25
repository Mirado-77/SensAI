import io
import json
import sys

import pytest

from sensai.core.config import McpConfig, McpServerConfig
from sensai.core.types import ToolContext
from sensai.mcp_server.main import MAX_REFERENCES, CodeTools, McpServer, ToolFailure, main
from sensai.tools.mcp.servers import McpServers

pytestmark = pytest.mark.feature("T1")

CALC = '''\
"""Small calculator."""


class Calculator:
    def add(self, a: int, b: int) -> int:
        return a + b

    async def total(self, values):
        return sum(values)


def mean(values: list[float]) -> float:
    return Calculator().add(0, 0) + sum(values) / len(values)
'''


@pytest.fixture
def project(tmp_path):
    root = tmp_path / "project"
    (root / "pkg").mkdir(parents=True)
    (root / "pkg" / "calc.py").write_text(CALC, encoding="utf-8")
    (root / "tests").mkdir()
    (root / "tests" / "test_calc.py").write_text(
        "from pkg.calc import mean\n\n\ndef test_mean():\n    assert mean([2, 4]) == 3\n",
        encoding="utf-8",
    )
    (root / ".venv").mkdir()
    (root / ".venv" / "lib.py").write_text("mean = 1\n", encoding="utf-8")
    (root / "broken.py").write_text("def oops(:\n", encoding="utf-8")
    (tmp_path / "secret.py").write_text("TOKEN = 'x'\n", encoding="utf-8")
    return root


def test_python_symbols_lists_classes_functions_and_methods(project):
    text = CodeTools(project).python_symbols("pkg/calc.py")
    assert text.splitlines() == [
        "class Calculator (line 4)",
        "  def add(self, a: int, b: int) -> int (line 5)",
        "  async def total(self, values) (line 8)",
        "def mean(values: list[float]) -> float (line 12)",
    ]


@pytest.mark.parametrize(
    ("path", "message"),
    [
        ("../secret.py", "outside the project root"),
        ("/etc/passwd", "outside the project root"),
        ("pkg", "Not a Python file"),
        ("missing.py", "Not a Python file"),
        ("broken.py", "Syntax error in broken.py line 1"),
    ],
)
def test_python_symbols_refuses_bad_paths(project, path, message):
    with pytest.raises(ToolFailure, match=message):
        CodeTools(project).python_symbols(path)


def test_python_symbols_refuses_a_symlink_leaving_the_root(project, tmp_path):
    (project / "link.py").symlink_to(tmp_path / "secret.py")
    with pytest.raises(ToolFailure, match="outside the project root"):
        CodeTools(project).python_symbols("link.py")


def test_find_references_lists_matching_lines_and_skips_virtualenvs(project):
    text = CodeTools(project).find_references("mean")
    assert text.splitlines() == [
        "pkg/calc.py:12: def mean(values: list[float]) -> float:",
        "tests/test_calc.py:1: from pkg.calc import mean",
        "tests/test_calc.py:5: assert mean([2, 4]) == 3",
    ]


def test_find_references_matches_whole_words_only(project):
    assert CodeTools(project).find_references("mea") == "No reference to mea."


def test_find_references_is_capped(project):
    lines = "\n".join(f"x{i} = mean" for i in range(MAX_REFERENCES + 10))
    (project / "many.py").write_text(lines, encoding="utf-8")
    text = CodeTools(project).find_references("mean")
    assert len(text.splitlines()) == MAX_REFERENCES + 1
    assert text.endswith(f"(stopped at {MAX_REFERENCES} results)")


def test_find_references_rejects_non_identifiers(project):
    with pytest.raises(ToolFailure, match="Not a Python identifier"):
        CodeTools(project).find_references("a.*b")


def exchange(project, *messages):
    output = io.StringIO()
    server = McpServer(CodeTools(project), output)
    server.serve(io.StringIO("".join(json.dumps(m) + "\n" for m in messages)))
    return [json.loads(line) for line in output.getvalue().splitlines()]


def test_server_answers_the_protocol_and_ignores_notifications(project):
    replies = exchange(
        project,
        {"jsonrpc": "2.0", "id": 1, "method": "initialize", "params": {}},
        {"jsonrpc": "2.0", "method": "notifications/initialized"},
        {"jsonrpc": "2.0", "id": 2, "method": "tools/list"},
        {"jsonrpc": "2.0", "id": 3, "method": "ping"},
        {"jsonrpc": "2.0", "id": 4, "method": "resources/list"},
    )
    assert [reply["id"] for reply in replies] == [1, 2, 3, 4]
    assert replies[0]["result"]["serverInfo"]["name"] == "sensai-mcp"
    tools = replies[1]["result"]["tools"]
    assert [tool["name"] for tool in tools] == ["python_symbols", "find_references"]
    assert all(tool["annotations"]["readOnlyHint"] for tool in tools)
    assert replies[2]["result"] == {}
    assert replies[3]["error"]["code"] == -32601


def test_tool_errors_are_results_with_is_error(project):
    call = {"name": "python_symbols", "arguments": {"path": "../secret.py"}}
    bad_args = {"name": "python_symbols", "arguments": {"file": "x"}}
    unknown = {"name": "rm_rf", "arguments": {}}
    replies = exchange(
        project,
        {"jsonrpc": "2.0", "id": 1, "method": "tools/call", "params": call},
        {"jsonrpc": "2.0", "id": 2, "method": "tools/call", "params": bad_args},
        {"jsonrpc": "2.0", "id": 3, "method": "tools/call", "params": unknown},
    )
    assert replies[0]["result"]["isError"] is True
    assert "outside the project root" in replies[0]["result"]["content"][0]["text"]
    assert replies[1]["result"]["isError"] is True
    assert replies[2]["error"]["code"] == -32602


def test_malformed_json_gets_a_parse_error(project):
    output = io.StringIO()
    McpServer(CodeTools(project), output).serve(io.StringIO("{nope\n"))
    assert json.loads(output.getvalue())["error"]["code"] == -32700


def test_main_refuses_a_missing_root(tmp_path, capsys):
    assert main(["--root", str(tmp_path / "missing")]) == 2
    assert "not a directory" in capsys.readouterr().err


async def test_our_server_works_end_to_end_through_the_mcp_client(project):
    server = McpServerConfig(
        name="code",
        command=sys.executable,
        args=["-m", "sensai.mcp_server.main", "--root", str(project)],
        risk="auto",
    )
    async with McpServers(McpConfig(timeout_s=10, servers=[server])) as servers:
        assert servers.warnings == []
        tools = {tool.name: tool for tool in servers.tools}
        assert set(tools) == {"code__python_symbols", "code__find_references"}
        assert {tool.risk for tool in tools.values()} == {"read"}
        result = await tools["code__python_symbols"].run(
            {"path": "pkg/calc.py"}, ToolContext("r", "s")
        )
    assert result.untrusted and not result.is_error
    assert "class Calculator (line 4)" in result.content

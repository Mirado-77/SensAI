import pytest

from sensai.core.config import AllowedRoot, load_config
from sensai.core.types import ToolContext
from sensai.tools.filesystem.guard import PathGuard
from sensai.tools.sandbox import run_command
from sensai.tools.sandbox.docker import ContainerResult
from sensai.tools.sandbox.run_command import (
    RunCommandTool,
    interpret_exit_code,
    sandbox_hints,
    to_tool_result,
    truncate_output,
)

pytestmark = pytest.mark.feature("T2")

CTX = ToolContext(run_id="r1", session_id="s1")
SANDBOX = load_config().sandbox


@pytest.fixture
def tool(tmp_path):
    (tmp_path / "pkg").mkdir()
    (tmp_path / "file.txt").write_text("x", encoding="utf-8")
    guard = PathGuard([AllowedRoot(path=tmp_path, mode="read")])
    return RunCommandTool(guard, SANDBOX)


@pytest.fixture
def calls(monkeypatch):
    recorded = []

    async def fake_run_container(args, name, timeout_s):
        recorded.append((args, name, timeout_s))
        return ContainerResult(0, b"ok\n", timed_out=False)

    monkeypatch.setattr(run_command, "run_container", fake_run_container)
    return recorded


def test_short_output_is_kept():
    assert truncate_output(b"a\nb\n", 100) == "a\nb\n"


def test_long_output_keeps_the_start_and_the_end():
    data = b"START\n" + b"line\n" * 1000 + b"END\n"
    text = truncate_output(data, 40)
    assert text.startswith("START")
    assert text.endswith("END\n")
    assert "lines truncated" in text
    assert len(text) < 100


def test_exit_code_zero_is_a_success():
    assert interpret_exit_code("pytest", 0) == (False, None)


@pytest.mark.parametrize("command", ["grep foo file", "/usr/bin/rg foo", "diff a b", "test -f x"])
def test_exit_code_one_is_not_an_error_for_some_programs(command):
    is_error, note = interpret_exit_code(command, 1)
    assert not is_error
    assert note


def test_other_exit_codes_are_errors():
    is_error, note = interpret_exit_code("python -m pytest", 1)
    assert is_error
    assert "code 1" in note
    assert interpret_exit_code("grep foo", 2)[0]


def test_hints_explain_the_sandbox_limits():
    assert "lecture seule" in sandbox_hints("touch: Read-only file system", 1)[0]
    assert "réseau" in sandbox_hints("npm ERR! EAI_AGAIN registry", 1)[0]
    assert "mémoire" in sandbox_hints("", 137)[0]
    assert sandbox_hints("all good", 0) == []


def test_the_same_hint_is_given_once():
    output = "Temporary failure in name resolution\nNetwork is unreachable"
    assert len(sandbox_hints(output, 1)) == 1


def test_result_of_failing_tests_is_an_untrusted_error():
    result = ContainerResult(1, b"1 failed, 7 passed\n", timed_out=False)
    tool_result = to_tool_result("python -m pytest", result, SANDBOX)
    assert tool_result.is_error
    assert tool_result.untrusted
    assert tool_result.content.startswith("1 failed, 7 passed")
    assert "code 1" in tool_result.content


def test_result_of_a_timeout_keeps_the_partial_output():
    result = ContainerResult(None, b"started\n", timed_out=True)
    tool_result = to_tool_result("sleep 100", result, SANDBOX)
    assert tool_result.is_error
    assert "started" in tool_result.content
    assert "délai" in tool_result.content


def test_result_when_docker_cannot_start_the_container():
    result = ContainerResult(125, b"Unable to find image\n", timed_out=False)
    tool_result = to_tool_result("ls", result, SANDBOX)
    assert tool_result.is_error
    assert not tool_result.untrusted
    assert "code 125" in tool_result.content


def test_empty_output_is_reported():
    result = ContainerResult(0, b"", timed_out=False)
    assert to_tool_result("true", result, SANDBOX).content == "(no output)"


@pytest.mark.parametrize(
    "args",
    [
        {},
        {"command": "  "},
        {"command": "ls", "path": 3},
        {"command": "ls", "description": ["x"]},
        {"command": "ls", "timeout_s": 0},
        {"command": "ls", "timeout_s": True},
        {"command": "ls", "timeout_s": "10"},
    ],
)
async def test_invalid_arguments_are_rejected(tool, calls, args):
    result = await tool.run(args, CTX)
    assert result.is_error
    assert calls == []


async def test_command_runs_in_the_first_root_by_default(tool, calls, tmp_path):
    result = await tool.run({"command": "python -m pytest"}, CTX)
    assert not result.is_error
    [(args, name, timeout_s)] = calls
    assert f"{tmp_path.resolve()}:/workspace:ro" in args
    assert args[-1] == "python -m pytest"
    assert name.startswith("sensai-")
    assert args[args.index("--name") + 1] == name
    assert timeout_s == SANDBOX.timeout_s


async def test_command_runs_in_a_sub_directory(tool, calls, tmp_path):
    await tool.run({"command": "ls", "path": "pkg"}, CTX)
    assert f"{(tmp_path / 'pkg').resolve()}:/workspace:ro" in calls[0][0]


@pytest.mark.parametrize("path", ["/etc", "../", "file.txt", "missing"])
async def test_path_outside_the_roots_or_not_a_directory_is_denied(tool, calls, path):
    result = await tool.run({"command": "ls", "path": path}, CTX)
    assert result.is_error
    assert calls == []


async def test_requested_timeout_is_capped_by_the_config(tool, calls):
    await tool.run({"command": "ls", "timeout_s": 1}, CTX)
    await tool.run({"command": "ls", "timeout_s": SANDBOX.timeout_s * 10}, CTX)
    assert [call[2] for call in calls] == [1.0, SANDBOX.timeout_s]


async def test_preview_shows_the_description_the_command_and_the_isolation(tool, tmp_path):
    preview = await tool.preview({"command": "npm test", "description": "Run the tests"})
    assert preview.splitlines()[0] == "# Run the tests"
    assert "$ npm test" in preview
    assert str(tmp_path.resolve()) in preview
    assert "no network" in preview


async def test_preview_of_invalid_arguments_falls_back(tool):
    assert await tool.preview({"command": ""}) is None


def test_tool_is_an_exec_tool(tool):
    assert tool.risk == "exec"
    assert tool.parameters["required"] == ["command"]

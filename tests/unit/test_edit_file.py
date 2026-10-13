import stat

import pytest

from sensai.core.config import AllowedRoot
from sensai.core.contracts import Tool
from sensai.core.types import ToolContext
from sensai.safety.untrusted import wrap_untrusted
from sensai.tools.filesystem.edit_file import EditFileTool
from sensai.tools.filesystem.guard import PathGuard

pytestmark = pytest.mark.feature("T4")

CTX = ToolContext(run_id="r1", session_id="s1")
LIMIT = 100
SOURCE = "def add(a, b):\n    return a - b\n\n\ndef sub(a, b):\n    return a - b\n"


@pytest.fixture
def root(tmp_path):
    folder = tmp_path / "root"
    folder.mkdir()
    return folder


def make_tool(root, mode="write"):
    return EditFileTool(PathGuard([AllowedRoot(path=root, mode=mode)]), LIMIT)


@pytest.fixture
def tool(root):
    return make_tool(root)


@pytest.fixture
def source(root):
    path = root / "calc.py"
    path.write_text(SOURCE, encoding="utf-8")
    return path


def edit_args(old_text, new_text, path="calc.py"):
    return {"path": path, "old_text": old_text, "new_text": new_text}


async def test_edit_file_replaces_a_unique_text(source, tool):
    old = "def add(a, b):\n    return a - b"
    new = "def add(a, b):\n    return a + b"
    result = await tool.run(edit_args(old, new), CTX)
    assert not result.is_error
    assert "modifié" in result.content
    assert source.read_text(encoding="utf-8") == SOURCE.replace(old, new)


async def test_edit_file_leaves_the_rest_of_the_file_untouched(source, tool):
    await tool.run(edit_args("def sub", "def subtract"), CTX)
    assert source.read_text(encoding="utf-8") == SOURCE.replace("def sub", "def subtract")


async def test_edit_file_deletes_the_text_when_new_text_is_empty(source, tool):
    result = await tool.run(edit_args("\n\n\ndef sub(a, b):\n    return a - b\n", ""), CTX)
    assert not result.is_error
    assert source.read_text(encoding="utf-8") == "def add(a, b):\n    return a - b"


async def test_edit_file_preserves_windows_line_endings(root, tool):
    (root / "win.txt").write_bytes(b"a\r\nb\r\nc\r\n")
    result = await tool.run(edit_args("b", "B", path="win.txt"), CTX)
    assert not result.is_error
    assert (root / "win.txt").read_bytes() == b"a\r\nB\r\nc\r\n"


async def test_edit_file_keeps_the_permissions_of_the_file(root, tool):
    script = root / "run.sh"
    script.write_text("echo one\n", encoding="utf-8")
    script.chmod(0o755)
    await tool.run(edit_args("one", "two", path="run.sh"), CTX)
    assert stat.S_IMODE(script.stat().st_mode) == 0o755


async def test_edit_file_refuses_an_ambiguous_text_and_counts_the_occurrences(source, tool):
    result = await tool.run(edit_args("return a - b", "return a + b"), CTX)
    assert result.is_error
    assert "ambigu" in result.content
    assert "2 occurrences" in result.content
    assert source.read_text(encoding="utf-8") == SOURCE


async def test_edit_file_refuses_a_text_that_is_not_in_the_file(source, tool):
    result = await tool.run(edit_args("def mul(a, b):", "def mul(a, b, c):"), CTX)
    assert result.is_error
    assert "introuvable" in result.content
    assert source.read_text(encoding="utf-8") == SOURCE


async def test_edit_file_does_not_match_a_different_indentation(source, tool):
    result = await tool.run(edit_args("  return a - b", "  return a + b"), CTX)
    assert result.is_error
    assert source.read_text(encoding="utf-8") == SOURCE


async def test_edit_file_does_not_create_a_missing_file(root, tool):
    result = await tool.run(edit_args("a", "b", path="missing.txt"), CTX)
    assert result.is_error
    assert "write_file" in result.content
    assert not (root / "missing.txt").exists()


async def test_edit_file_reports_a_folder_instead_of_a_file(root, tool):
    (root / "sub").mkdir()
    result = await tool.run(edit_args("a", "b", path="sub"), CTX)
    assert result.is_error
    assert "introuvable" in result.content


async def test_edit_file_refuses_a_binary_file(root, tool):
    (root / "data.bin").write_bytes(b"\xff\xfe\x00\x01")
    result = await tool.run(edit_args("a", "b", path="data.bin"), CTX)
    assert result.is_error
    assert "Lecture impossible" in result.content
    assert (root / "data.bin").read_bytes() == b"\xff\xfe\x00\x01"


async def test_edit_file_refuses_a_file_above_the_limit(root, tool):
    (root / "big.txt").write_text("a" * (LIMIT + 1), encoding="utf-8")
    result = await tool.run(edit_args("a", "b", path="big.txt"), CTX)
    assert result.is_error
    assert "volumineux" in result.content
    assert (root / "big.txt").read_text(encoding="utf-8") == "a" * (LIMIT + 1)


async def test_edit_file_refuses_a_result_above_the_limit(root, tool):
    (root / "small.txt").write_text("ab", encoding="utf-8")
    result = await tool.run(edit_args("a", "x" * (LIMIT + 1), path="small.txt"), CTX)
    assert result.is_error
    assert "volumineux" in result.content
    assert (root / "small.txt").read_text(encoding="utf-8") == "ab"


async def test_edit_file_accepts_a_result_exactly_at_the_limit(root, tool):
    (root / "small.txt").write_text("ab", encoding="utf-8")
    result = await tool.run(edit_args("a", "x" * (LIMIT - 1), path="small.txt"), CTX)
    assert not result.is_error


async def test_edit_file_refuses_a_read_only_root(root, source):
    result = await make_tool(root, mode="read").run(edit_args("def sub", "def subtract"), CTX)
    assert result.is_error
    assert "lecture seule" in result.content
    assert source.read_text(encoding="utf-8") == SOURCE


async def test_edit_file_refuses_a_path_outside_the_root(tmp_path, tool):
    outside = tmp_path / "outside.txt"
    outside.write_text("original", encoding="utf-8")
    result = await tool.run(edit_args("original", "changed", path="../outside.txt"), CTX)
    assert result.is_error
    assert "hors des racines" in result.content
    assert outside.read_text(encoding="utf-8") == "original"


async def test_edit_file_refuses_a_symlink_that_leaves_the_root(tmp_path, root, tool):
    outside = tmp_path / "outside.txt"
    outside.write_text("original", encoding="utf-8")
    (root / "link.txt").symlink_to(outside)
    result = await tool.run(edit_args("original", "changed", path="link.txt"), CTX)
    assert result.is_error
    assert outside.read_text(encoding="utf-8") == "original"


async def test_edit_file_refuses_identical_texts(source, tool):
    result = await tool.run(edit_args("def sub", "def sub"), CTX)
    assert result.is_error
    assert "différents" in result.content


async def test_edit_file_refuses_a_new_text_copied_from_an_untrusted_result(source, tool):
    result = await tool.run(edit_args("def sub", wrap_untrusted("def sub")), CTX)
    assert result.is_error
    assert "<untrusted_content>" in result.content
    assert source.read_text(encoding="utf-8") == SOURCE


@pytest.mark.parametrize(
    "args",
    [
        {},
        {"old_text": "a", "new_text": "b"},
        {"path": "", "old_text": "a", "new_text": "b"},
        {"path": 3, "old_text": "a", "new_text": "b"},
        {"path": "calc.py", "new_text": "b"},
        {"path": "calc.py", "old_text": "", "new_text": "b"},
        {"path": "calc.py", "old_text": None, "new_text": "b"},
        {"path": "calc.py", "old_text": 3, "new_text": "b"},
        {"path": "calc.py", "old_text": "a"},
        {"path": "calc.py", "old_text": "a", "new_text": None},
        {"path": "calc.py", "old_text": "a", "new_text": 3},
    ],
)
async def test_edit_file_rejects_missing_or_invalid_arguments(source, tool, args):
    result = await tool.run(args, CTX)
    assert result.is_error
    assert "Argument" in result.content
    assert source.read_text(encoding="utf-8") == SOURCE


async def test_edit_file_result_is_not_marked_untrusted(source, tool):
    result = await tool.run(edit_args("def sub", "def subtract"), CTX)
    assert not result.untrusted


def test_edit_file_satisfies_the_tool_contract(tool):
    assert isinstance(tool, Tool)
    assert tool.risk == "write"


def test_edit_file_requires_its_three_arguments_described_by_strings(tool):
    assert tool.parameters["required"] == ["path", "old_text", "new_text"]
    for name in ("path", "old_text", "new_text"):
        assert isinstance(tool.parameters["properties"][name]["description"], str)

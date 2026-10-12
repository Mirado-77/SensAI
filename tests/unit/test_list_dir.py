import os

import pytest

from sensai.core.config import AllowedRoot
from sensai.core.contracts import Tool
from sensai.core.types import ToolContext
from sensai.tools.filesystem.guard import PathGuard
from sensai.tools.filesystem.list_dir import ListDirTool

pytestmark = pytest.mark.feature("T4")

CTX = ToolContext(run_id="r1", session_id="s1")
LIMIT = 3


@pytest.fixture
def root(tmp_path):
    folder = tmp_path / "root"
    folder.mkdir()
    return folder


@pytest.fixture
def tool(root):
    return ListDirTool(PathGuard([AllowedRoot(path=root, mode="read")]), LIMIT)


async def test_list_dir_lists_files_and_folders_sorted_with_a_slash_on_folders(root, tool):
    (root / "b.txt").write_text("b", encoding="utf-8")
    (root / "a_dir").mkdir()
    (root / "c.txt").write_text("c", encoding="utf-8")
    result = await tool.run({"path": "."}, CTX)
    assert result.content == "a_dir/\nb.txt\nc.txt"
    assert not result.is_error
    assert result.untrusted


@pytest.mark.parametrize("args", [{}, {"path": None}, {"path": "."}])
async def test_list_dir_lists_the_root_when_no_path_is_given(root, tool, args):
    (root / "a.txt").write_text("a", encoding="utf-8")
    result = await tool.run(args, CTX)
    assert result.content == "a.txt"


async def test_list_dir_lists_a_subfolder(root, tool):
    (root / "src").mkdir()
    (root / "src" / "main.py").write_text("print()", encoding="utf-8")
    (root / "top.txt").write_text("top", encoding="utf-8")
    result = await tool.run({"path": "src"}, CTX)
    assert result.content == "main.py"


async def test_list_dir_reports_an_empty_folder_without_an_error(tool):
    result = await tool.run({}, CTX)
    assert result.content == "(empty directory)"
    assert not result.is_error


async def test_list_dir_truncates_above_the_limit_and_counts_the_hidden_entries(root, tool):
    for name in "abcde":
        (root / f"{name}.txt").write_text(name, encoding="utf-8")
    result = await tool.run({}, CTX)
    assert result.content == "a.txt\nb.txt\nc.txt\n[truncated: 2 more entries]"
    assert not result.is_error


async def test_list_dir_does_not_truncate_at_the_limit(root, tool):
    for name in "abc":
        (root / f"{name}.txt").write_text(name, encoding="utf-8")
    result = await tool.run({}, CTX)
    assert result.content == "a.txt\nb.txt\nc.txt"


async def test_list_dir_reports_a_missing_folder(tool):
    result = await tool.run({"path": "missing"}, CTX)
    assert result.is_error
    assert "introuvable" in result.content


async def test_list_dir_reports_a_file_instead_of_a_folder(root, tool):
    (root / "a.txt").write_text("a", encoding="utf-8")
    result = await tool.run({"path": "a.txt"}, CTX)
    assert result.is_error
    assert "introuvable" in result.content


@pytest.mark.parametrize("args", [{"path": ""}, {"path": "   "}, {"path": 3}])
async def test_list_dir_rejects_an_invalid_path_argument(tool, args):
    result = await tool.run(args, CTX)
    assert result.is_error
    assert "path" in result.content


async def test_list_dir_refuses_a_folder_outside_the_root(tmp_path, tool):
    outside = tmp_path / "outside"
    outside.mkdir()
    (outside / "classified.txt").write_text("x", encoding="utf-8")
    result = await tool.run({"path": "../outside"}, CTX)
    assert result.is_error
    assert "hors des racines" in result.content
    assert "classified.txt" not in result.content


async def test_list_dir_refuses_a_symlink_to_a_folder_outside_the_root(tmp_path, root, tool):
    outside = tmp_path / "outside"
    outside.mkdir()
    (root / "link").symlink_to(outside)
    result = await tool.run({"path": "link"}, CTX)
    assert result.is_error
    assert "hors des racines" in result.content


@pytest.mark.skipif(os.geteuid() == 0, reason="root can list any folder")
async def test_list_dir_reports_an_unreadable_folder(root, tool):
    locked = root / "locked"
    locked.mkdir()
    locked.chmod(0o000)
    try:
        result = await tool.run({"path": "locked"}, CTX)
    finally:
        locked.chmod(0o700)
    assert result.is_error
    assert "Lecture impossible" in result.content


def test_list_dir_satisfies_the_tool_contract(tool):
    assert isinstance(tool, Tool)
    assert tool.risk == "read"


def test_list_dir_path_is_optional_and_described_by_a_string(tool):
    assert "path" not in tool.parameters.get("required", [])
    assert isinstance(tool.parameters["properties"]["path"]["description"], str)

import os

import pytest

from sensai.core.config import AllowedRoot
from sensai.core.contracts import Tool
from sensai.core.types import ToolContext
from sensai.tools.filesystem.guard import PathGuard
from sensai.tools.filesystem.read_file import ReadFileTool

pytestmark = pytest.mark.feature("T4")

CTX = ToolContext(run_id="r1", session_id="s1")
LIMIT = 20


@pytest.fixture
def root(tmp_path):
    folder = tmp_path / "root"
    folder.mkdir()
    return folder


@pytest.fixture
def tool(root):
    return ReadFileTool(PathGuard([AllowedRoot(path=root, mode="read")]), LIMIT)


async def test_read_file_returns_the_content_as_untrusted(root, tool):
    (root / "a.txt").write_text("hello", encoding="utf-8")
    result = await tool.run({"path": "a.txt"}, CTX)
    assert result.content == "hello"
    assert not result.is_error
    assert result.untrusted


async def test_read_file_reports_a_missing_file(tool):
    result = await tool.run({"path": "missing.txt"}, CTX)
    assert result.is_error
    assert "introuvable" in result.content


async def test_read_file_reports_a_directory(root, tool):
    (root / "sub").mkdir()
    result = await tool.run({"path": "sub"}, CTX)
    assert result.is_error
    assert "introuvable" in result.content


async def test_read_file_refuses_a_binary_file(root, tool):
    (root / "image.bin").write_bytes(b"\x89PNG\x00\x01\x02")
    result = await tool.run({"path": "image.bin"}, CTX)
    assert result.is_error
    assert "binaire" in result.content


async def test_read_file_truncates_a_file_above_the_limit(root, tool):
    (root / "big.txt").write_text("x" * (LIMIT * 3), encoding="utf-8")
    result = await tool.run({"path": "big.txt"}, CTX)
    assert not result.is_error
    assert result.content.startswith("x" * LIMIT)
    assert "x" * (LIMIT + 1) not in result.content
    assert "[truncated" in result.content


async def test_read_file_does_not_truncate_a_file_at_the_limit(root, tool):
    (root / "exact.txt").write_text("x" * LIMIT, encoding="utf-8")
    result = await tool.run({"path": "exact.txt"}, CTX)
    assert result.content == "x" * LIMIT


async def test_read_file_replaces_invalid_utf8_bytes(root, tool):
    (root / "latin.txt").write_bytes(b"caf\xe9")
    result = await tool.run({"path": "latin.txt"}, CTX)
    assert not result.is_error
    assert result.content == "caf�"


@pytest.mark.parametrize("args", [{}, {"path": ""}, {"path": "   "}, {"path": 3}, {"path": None}])
async def test_read_file_rejects_a_missing_or_invalid_path_argument(tool, args):
    result = await tool.run(args, CTX)
    assert result.is_error
    assert "path" in result.content


async def test_read_file_refuses_a_path_outside_the_root(tmp_path, tool):
    outside = tmp_path / "outside"
    outside.mkdir()
    (outside / "secret.txt").write_text("classified-data", encoding="utf-8")
    result = await tool.run({"path": "../outside/secret.txt"}, CTX)
    assert result.is_error
    assert "hors des racines" in result.content
    assert "classified-data" not in result.content


@pytest.mark.skipif(os.geteuid() == 0, reason="root can read any file")
async def test_read_file_reports_an_unreadable_file(root, tool):
    locked = root / "locked.txt"
    locked.write_text("hello", encoding="utf-8")
    locked.chmod(0o000)
    result = await tool.run({"path": "locked.txt"}, CTX)
    assert result.is_error
    assert "Lecture impossible" in result.content


def test_read_file_satisfies_the_tool_contract(tool):
    assert isinstance(tool, Tool)
    assert tool.risk == "read"

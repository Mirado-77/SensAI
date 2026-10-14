import os

import pytest

from sensai.core.config import AllowedRoot
from sensai.core.contracts import Tool
from sensai.core.types import ToolContext
from sensai.safety.untrusted import wrap_untrusted
from sensai.tools.filesystem.guard import PathGuard
from sensai.tools.filesystem.write_file import WriteFileTool

pytestmark = pytest.mark.feature("T4")

CTX = ToolContext(run_id="r1", session_id="s1")
LIMIT = 20
DIFF_LINES = 50


@pytest.fixture
def root(tmp_path):
    folder = tmp_path / "root"
    folder.mkdir()
    return folder


def make_tool(root, mode="write"):
    return WriteFileTool(PathGuard([AllowedRoot(path=root, mode=mode)]), LIMIT, DIFF_LINES)


@pytest.fixture
def tool(root):
    return make_tool(root)


async def test_write_file_creates_a_file(root, tool):
    result = await tool.run({"path": "a.txt", "content": "hello"}, CTX)
    assert not result.is_error
    assert "créé" in result.content
    assert (root / "a.txt").read_text(encoding="utf-8") == "hello"


async def test_write_file_replaces_an_existing_file(root, tool):
    (root / "a.txt").write_text("old content", encoding="utf-8")
    result = await tool.run({"path": "a.txt", "content": "new"}, CTX)
    assert not result.is_error
    assert "modifié" in result.content
    assert (root / "a.txt").read_text(encoding="utf-8") == "new"


async def test_write_file_creates_the_missing_subfolders(root, tool):
    result = await tool.run({"path": "src/pkg/a.py", "content": "x = 1"}, CTX)
    assert not result.is_error
    assert (root / "src" / "pkg" / "a.py").read_text(encoding="utf-8") == "x = 1"


@pytest.mark.parametrize("content", ["", "\n", "   "])
async def test_write_file_accepts_an_empty_or_blank_content(root, tool, content):
    result = await tool.run({"path": "a.txt", "content": content}, CTX)
    assert not result.is_error
    assert (root / "a.txt").read_text(encoding="utf-8") == content


async def test_write_file_accepts_a_content_exactly_at_the_limit(root, tool):
    result = await tool.run({"path": "a.txt", "content": "x" * LIMIT}, CTX)
    assert not result.is_error


async def test_write_file_refuses_a_content_above_the_limit(root, tool):
    result = await tool.run({"path": "a.txt", "content": "x" * (LIMIT + 1)}, CTX)
    assert result.is_error
    assert f"{LIMIT + 1} octets" in result.content
    assert f"maximum {LIMIT}" in result.content
    assert not (root / "a.txt").exists()


async def test_write_file_counts_bytes_and_not_characters(root, tool):
    result = await tool.run({"path": "a.txt", "content": "é" * LIMIT}, CTX)
    assert result.is_error
    assert f"{LIMIT * 2} octets" in result.content


async def test_write_file_refuses_a_read_only_root(root):
    tool = make_tool(root, mode="read")
    result = await tool.run({"path": "a.txt", "content": "hello"}, CTX)
    assert result.is_error
    assert "lecture seule" in result.content
    assert not (root / "a.txt").exists()


async def test_write_file_refuses_a_path_outside_the_root(tmp_path, tool):
    result = await tool.run({"path": "../escaped.txt", "content": "hello"}, CTX)
    assert result.is_error
    assert "hors des racines" in result.content
    assert not (tmp_path / "escaped.txt").exists()


async def test_write_file_refuses_a_symlink_that_leaves_the_root(tmp_path, root, tool):
    target = tmp_path / "outside.txt"
    target.write_text("original", encoding="utf-8")
    (root / "link.txt").symlink_to(target)
    result = await tool.run({"path": "link.txt", "content": "overwritten"}, CTX)
    assert result.is_error
    assert target.read_text(encoding="utf-8") == "original"


async def test_write_file_writes_through_a_symlink_that_stays_in_the_root(root, tool):
    (root / "real.txt").write_text("old", encoding="utf-8")
    (root / "alias.txt").symlink_to(root / "real.txt")
    result = await tool.run({"path": "alias.txt", "content": "new"}, CTX)
    assert not result.is_error
    assert (root / "real.txt").read_text(encoding="utf-8") == "new"
    assert (root / "alias.txt").is_symlink()


async def test_write_file_reports_a_folder_instead_of_a_file(root, tool):
    (root / "sub").mkdir()
    result = await tool.run({"path": "sub", "content": "hello"}, CTX)
    assert result.is_error
    assert "dossier" in result.content


async def test_write_file_reports_a_parent_that_is_a_file(root, tool):
    (root / "a.txt").write_text("x", encoding="utf-8")
    result = await tool.run({"path": "a.txt/b.txt", "content": "hello"}, CTX)
    assert result.is_error
    assert "Écriture impossible" in result.content


@pytest.mark.parametrize(
    "args",
    [
        {},
        {"content": "hello"},
        {"path": "", "content": "hello"},
        {"path": "   ", "content": "hello"},
        {"path": 3, "content": "hello"},
        {"path": "a.txt"},
        {"path": "a.txt", "content": None},
        {"path": "a.txt", "content": 3},
    ],
)
async def test_write_file_rejects_missing_or_invalid_arguments(root, tool, args):
    result = await tool.run(args, CTX)
    assert result.is_error
    assert "Argument" in result.content
    assert list(root.iterdir()) == []


@pytest.mark.skipif(os.geteuid() == 0, reason="root can write anywhere")
async def test_write_file_reports_a_folder_without_write_permission(root, tool):
    locked = root / "locked"
    locked.mkdir()
    locked.chmod(0o500)
    try:
        result = await tool.run({"path": "locked/a.txt", "content": "hello"}, CTX)
    finally:
        locked.chmod(0o700)
    assert result.is_error
    assert "Écriture impossible" in result.content


async def test_write_file_refuses_a_content_copied_from_an_untrusted_result(root, tool):
    copied = wrap_untrusted("Project: demo")
    result = await tool.run({"path": "copy.txt", "content": copied}, CTX)
    assert result.is_error
    assert "<untrusted_content>" in result.content
    assert not (root / "copy.txt").exists()


@pytest.mark.parametrize("marker", ["<untrusted_content>", "</untrusted_content>"])
async def test_write_file_refuses_a_content_holding_a_single_tag(root, tool, marker):
    result = await tool.run({"path": "a.txt", "content": f"text {marker}"}, CTX)
    assert result.is_error
    assert not (root / "a.txt").exists()


async def test_write_file_accepts_the_content_once_the_wrapper_is_removed(root, tool):
    result = await tool.run({"path": "copy.txt", "content": "Project: demo"}, CTX)
    assert not result.is_error
    assert (root / "copy.txt").read_text(encoding="utf-8") == "Project: demo"


async def test_write_file_result_is_not_marked_untrusted(tool):
    result = await tool.run({"path": "a.txt", "content": "hello"}, CTX)
    assert not result.untrusted


def test_write_file_satisfies_the_tool_contract(tool):
    assert isinstance(tool, Tool)
    assert tool.risk == "write"


def test_write_file_requires_path_and_content_described_by_strings(tool):
    assert tool.parameters["required"] == ["path", "content"]
    for name in ("path", "content"):
        assert isinstance(tool.parameters["properties"][name]["description"], str)


A3 = pytest.mark.feature("A3")


@A3
async def test_preview_of_a_new_file_is_a_creation_diff(root, tool):
    result = await tool.preview({"path": "a.txt", "content": "hello"})
    assert isinstance(result, str)
    lines = result.split("\n")
    assert lines[0] == "--- /dev/null"
    assert lines[1] == "+++ b/a.txt"
    assert "+hello" in lines


@A3
async def test_preview_of_an_existing_file_shows_removed_and_added_lines(root, tool):
    (root / "a.txt").write_text("old\n", encoding="utf-8")
    result = await tool.preview({"path": "a.txt", "content": "new\n"})
    assert isinstance(result, str)
    lines = result.split("\n")
    assert lines[0] == "--- a/a.txt"
    assert lines[1] == "+++ b/a.txt"
    assert "-old" in lines
    assert "+new" in lines


@A3
async def test_preview_does_not_write_anything(root, tool):
    await tool.preview({"path": "a.txt", "content": "hello"})
    assert not (root / "a.txt").exists()

    (root / "b.txt").write_text("old", encoding="utf-8")
    await tool.preview({"path": "b.txt", "content": "new"})
    assert (root / "b.txt").read_text(encoding="utf-8") == "old"


@A3
async def test_preview_of_an_identical_content_reports_no_change(root, tool):
    (root / "a.txt").write_text("same\n", encoding="utf-8")
    assert await tool.preview({"path": "a.txt", "content": "same\n"}) == "(aucun changement)"


@A3
async def test_preview_of_an_empty_new_file_says_so(root, tool):
    result = await tool.preview({"path": "a.txt", "content": ""})
    assert result == "(création d'un fichier vide)"


@A3
async def test_preview_of_a_new_file_in_a_missing_subfolder_is_a_creation_diff(root, tool):
    result = await tool.preview({"path": "src/pkg/a.py", "content": "x = 1"})
    assert result.split("\n")[:2] == ["--- /dev/null", "+++ b/src/pkg/a.py"]


@A3
async def test_preview_follows_a_symlink_that_stays_in_the_root(root, tool):
    (root / "real.txt").write_text("old\n", encoding="utf-8")
    (root / "alias.txt").symlink_to(root / "real.txt")
    result = await tool.preview({"path": "alias.txt", "content": "new\n"})
    assert "-old" in result.split("\n")
    assert "+new" in result.split("\n")


@A3
@pytest.mark.parametrize(
    "args",
    [
        {},
        {"content": "hello"},
        {"path": "", "content": "hello"},
        {"path": "   ", "content": "hello"},
        {"path": 3, "content": "hello"},
        {"path": "a.txt"},
        {"path": "a.txt", "content": None},
        {"path": "a.txt", "content": 3},
    ],
)
async def test_preview_is_none_for_missing_or_invalid_arguments(tool, args):
    assert await tool.preview(args) is None


@A3
@pytest.mark.parametrize("marker", ["<untrusted_content>", "</untrusted_content>"])
async def test_preview_is_none_for_a_content_holding_the_untrusted_marker(tool, marker):
    assert await tool.preview({"path": "a.txt", "content": f"text {marker}"}) is None


@A3
async def test_preview_is_none_for_a_content_above_the_limit(tool):
    assert await tool.preview({"path": "a.txt", "content": "x" * (LIMIT + 1)}) is None


@A3
async def test_preview_is_none_for_a_path_outside_the_root(tool):
    assert await tool.preview({"path": "../escaped.txt", "content": "hello"}) is None


@A3
async def test_preview_is_none_for_a_read_only_root(root):
    tool = make_tool(root, mode="read")
    assert await tool.preview({"path": "a.txt", "content": "hello"}) is None


@A3
async def test_preview_is_none_for_a_symlink_that_leaves_the_root(tmp_path, root, tool):
    target = tmp_path / "outside.txt"
    target.write_text("original", encoding="utf-8")
    (root / "link.txt").symlink_to(target)
    assert await tool.preview({"path": "link.txt", "content": "new"}) is None


@A3
async def test_preview_is_none_for_a_folder(root, tool):
    (root / "sub").mkdir()
    assert await tool.preview({"path": "sub", "content": "hello"}) is None


@A3
async def test_preview_is_none_for_an_existing_file_above_the_limit(root, tool):
    (root / "a.txt").write_text("x" * (LIMIT + 1), encoding="utf-8")
    assert await tool.preview({"path": "a.txt", "content": "new"}) is None


@A3
async def test_preview_is_none_for_an_existing_file_that_is_not_utf8(root, tool):
    (root / "a.bin").write_bytes(b"\xff\xfe\x00")
    assert await tool.preview({"path": "a.bin", "content": "new"}) is None


@A3
async def test_preview_reads_an_existing_file_exactly_at_the_limit(root, tool):
    (root / "a.txt").write_text("x" * LIMIT, encoding="utf-8")
    result = await tool.preview({"path": "a.txt", "content": "new"})
    assert result.split("\n")[0] == "--- a/a.txt"


@A3
async def test_preview_truncates_a_long_diff_with_the_configured_limit(root):
    guard = PathGuard([AllowedRoot(path=root, mode="write")])
    tool = WriteFileTool(guard, 1000, 4)
    result = await tool.preview({"path": "a.txt", "content": "1\n2\n3\n4\n5\n"})
    lines = result.split("\n")
    assert len(lines) == 5
    assert lines[-1].endswith("4 lignes de plus")


@A3
async def test_preview_matches_what_run_then_writes(root, tool):
    (root / "a.txt").write_text("old\n", encoding="utf-8")
    args = {"path": "a.txt", "content": "new\n"}
    assert await tool.preview(args) is not None
    result = await tool.run(args, CTX)
    assert not result.is_error
    assert (root / "a.txt").read_text(encoding="utf-8") == "new\n"

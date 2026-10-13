import pytest

from sensai.core.config import load_config
from sensai.core.errors import ConfigError
from sensai.core.types import ToolContext
from sensai.tools import native_tools

pytestmark = pytest.mark.feature("T4")

CTX = ToolContext(run_id="r1", session_id="s1")


def config_with_root(root, **limits):
    files = {"allowed_roots": [{"path": str(root), "mode": "write"}], **limits}
    return load_config(overrides={"files": files})


def tool_named(config, name):
    return next(tool for tool in native_tools(config) if tool.name == name)


def test_there_are_no_native_tools_without_an_allowed_root():
    assert native_tools(load_config()) == []


def test_the_file_tools_are_offered_when_a_root_is_configured(tmp_path):
    names = sorted(tool.name for tool in native_tools(config_with_root(tmp_path)))
    assert names == ["edit_file", "list_dir", "read_file", "write_file"]


def test_a_missing_root_is_rejected(tmp_path):
    with pytest.raises(ConfigError, match="introuvable"):
        native_tools(config_with_root(tmp_path / "missing"))


async def test_the_read_limit_comes_from_the_configuration(tmp_path):
    (tmp_path / "big.txt").write_text("x" * 50, encoding="utf-8")
    tool = tool_named(config_with_root(tmp_path, max_read_bytes=5), "read_file")
    result = await tool.run({"path": "big.txt"}, CTX)
    assert result.content.startswith("xxxxx\n[truncated")


async def test_the_write_limit_comes_from_the_configuration(tmp_path):
    tool = tool_named(config_with_root(tmp_path, max_write_bytes=5), "write_file")
    result = await tool.run({"path": "a.txt", "content": "x" * 6}, CTX)
    assert result.is_error
    assert "volumineux" in result.content


async def test_the_list_limit_comes_from_the_configuration(tmp_path):
    for name in "abc":
        (tmp_path / f"{name}.txt").write_text(name, encoding="utf-8")
    tool = tool_named(config_with_root(tmp_path, max_list_entries=2), "list_dir")
    result = await tool.run({}, CTX)
    assert result.content == "a.txt\nb.txt\n[truncated: 1 more entries]"

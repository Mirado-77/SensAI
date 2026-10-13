import pytest

from sensai.core.config import load_config
from sensai.core.errors import ConfigError
from sensai.core.types import ToolContext
from sensai.tools import build_registry

pytestmark = pytest.mark.feature("T4")

CTX = ToolContext(run_id="r1", session_id="s1")


def config_with_root(root, **limits):
    files = {"allowed_roots": [{"path": str(root), "mode": "read"}], **limits}
    return load_config(overrides={"files": files})


def test_the_registry_is_empty_without_an_allowed_root():
    registry = build_registry(load_config())
    assert len(registry) == 0


def test_the_registry_offers_the_file_tools_when_a_root_is_configured(tmp_path):
    registry = build_registry(config_with_root(tmp_path))
    offered = sorted(spec.name for spec in registry.specs())
    assert offered == ["edit_file", "list_dir", "read_file", "write_file"]


def test_a_missing_root_is_rejected(tmp_path):
    with pytest.raises(ConfigError, match="introuvable"):
        build_registry(config_with_root(tmp_path / "missing"))


async def test_the_read_limit_comes_from_the_configuration(tmp_path):
    (tmp_path / "big.txt").write_text("x" * 50, encoding="utf-8")
    registry = build_registry(config_with_root(tmp_path, max_read_bytes=5))
    result = await registry.get("read_file").run({"path": "big.txt"}, CTX)
    assert result.content.startswith("xxxxx\n[truncated")

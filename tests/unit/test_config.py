from pathlib import Path

import pytest

from sensai.core.config import DEFAULT_CONFIG_PATH, load_config
from sensai.core.errors import ConfigError


def test_default_file_exists_and_loads():
    assert DEFAULT_CONFIG_PATH.is_file()
    config = load_config()
    assert config.model.name
    assert config.model.host.startswith("http")
    assert config.runtime.max_steps > 0


def test_overrides_replace_a_value_and_keep_the_rest():
    default = load_config()
    config = load_config(overrides={"model": {"name": "llama3.2:3b"}})
    assert config.model.name == "llama3.2:3b"
    assert config.model.host == default.model.host
    assert config.runtime == default.runtime


def test_user_file_only_contains_the_keys_to_change(tmp_path):
    user_file = tmp_path / "user.toml"
    user_file.write_text("[runtime]\nmax_steps = 3\n", encoding="utf-8")
    config = load_config(user_file)
    assert config.runtime.max_steps == 3
    assert config.runtime.max_tool_calls == load_config().runtime.max_tool_calls


def test_overrides_take_precedence_over_the_user_file(tmp_path):
    user_file = tmp_path / "user.toml"
    user_file.write_text('[model]\nname = "from-file"\n', encoding="utf-8")
    config = load_config(user_file, overrides={"model": {"name": "from-cli"}})
    assert config.model.name == "from-cli"


def test_missing_file(tmp_path):
    with pytest.raises(ConfigError, match="introuvable"):
        load_config(tmp_path / "absent.toml")


def test_unreadable_toml_file(tmp_path):
    broken = tmp_path / "broken.toml"
    broken.write_text("[model\nname =", encoding="utf-8")
    with pytest.raises(ConfigError, match="illisible"):
        load_config(broken)


@pytest.mark.parametrize(
    "overrides",
    [
        {"model": {"name": ""}},
        {"model": {"timeout_s": 0}},
        {"model": {"timeout_s": "fast"}},
        {"runtime": {"max_steps": 0}},
        {"runtime": {"max_tool_calls": -1}},
        {"model": {"unknown_key": 1}},
        {"unknown_section": {}},
    ],
)
def test_invalid_configuration(overrides):
    with pytest.raises(ConfigError, match="invalide"):
        load_config(overrides=overrides)


def test_max_tool_calls_can_be_zero():
    assert load_config(overrides={"runtime": {"max_tool_calls": 0}}).runtime.max_tool_calls == 0


def test_default_policy_asks_before_write_and_exec():
    policy = load_config().policy
    assert (policy.read, policy.write, policy.exec) == ("allow", "ask", "ask")


def test_policy_rejects_an_unknown_decision():
    with pytest.raises(ConfigError, match="invalide"):
        load_config(overrides={"policy": {"read": "maybe"}})


def test_storage_path_is_a_path():
    config = load_config(overrides={"storage": {"db_path": "/tmp/sensai-test.db"}})
    assert config.storage.db_path == Path("/tmp/sensai-test.db")


def test_files_invalid_mode():
    with pytest.raises(ConfigError, match="invalide"):
        load_config(overrides={"files": {"allowed_roots": [{"path": ".", "mode": "exec"}]}})


def test_files_valid_root():
    root = {"path": "/tmp/demo", "mode": "write"}
    config = load_config(overrides={"files": {"allowed_roots": [root]}})
    assert config.files.allowed_roots[0].path == Path("/tmp/demo")
    assert config.files.allowed_roots[0].mode == "write"


def test_files_have_no_root_by_default():
    config = load_config()
    assert config.files.allowed_roots == []


def test_files_rejects_a_zero_limit():
    with pytest.raises(ConfigError, match="invalide"):
        load_config(overrides={"files": {"max_read_bytes": 0}})


@pytest.mark.feature("T5")
def test_structured_output_allows_one_repair_by_default():
    assert load_config().structured.max_repairs == 1


@pytest.mark.feature("T5")
@pytest.mark.parametrize("value", [-1, 4])
def test_structured_max_repairs_is_bounded(value):
    with pytest.raises(ConfigError):
        load_config(overrides={"structured": {"max_repairs": value}})


@pytest.mark.feature("T1")
def test_no_mcp_server_is_started_by_default():
    config = load_config()
    assert config.mcp.servers == []
    assert config.mcp.timeout_s > 0


@pytest.mark.feature("T1")
def test_mcp_servers_are_read_from_a_user_file(tmp_path):
    path = tmp_path / "mcp.toml"
    path.write_text(
        '[[mcp.servers]]\nname = "git"\ncommand = "uvx"\n'
        'args = ["mcp-server-git"]\nrisk = "auto"\n',
        encoding="utf-8",
    )
    [server] = load_config(path).mcp.servers
    assert (server.name, server.command, server.args, server.risk) == (
        "git",
        "uvx",
        ["mcp-server-git"],
        "auto",
    )
    assert server.enabled


@pytest.mark.feature("T1")
@pytest.mark.parametrize(
    "server",
    [
        {"name": "bad name", "command": "x"},
        {"name": "ok", "command": ""},
        {"name": "ok", "command": "x", "risk": "maybe"},
    ],
)
def test_invalid_mcp_servers_are_rejected(server):
    with pytest.raises(ConfigError):
        load_config(overrides={"mcp": {"servers": [server]}})


@pytest.mark.feature("T2")
def test_sandbox_defaults_are_loaded():
    sandbox = load_config().sandbox
    assert sandbox.enabled
    assert sandbox.image == "sensai-sandbox:latest"
    assert sandbox.cpus > 0
    assert sandbox.memory_mb > 0
    assert sandbox.pids_limit > 0
    assert sandbox.timeout_s > 0
    assert sandbox.max_output_bytes > 0


@pytest.mark.feature("T2")
def test_sandbox_can_be_disabled_from_a_user_file(tmp_path):
    path = tmp_path / "sandbox.toml"
    path.write_text("[sandbox]\nenabled = false\n", encoding="utf-8")
    sandbox = load_config(path).sandbox
    assert not sandbox.enabled
    assert sandbox.image == load_config().sandbox.image


@pytest.mark.feature("T2")
@pytest.mark.parametrize(
    "sandbox",
    [
        {"image": ""},
        {"cpus": 0},
        {"memory_mb": 0},
        {"pids_limit": -1},
        {"timeout_s": 0},
        {"max_output_bytes": 0},
        {"network": "bridge"},
    ],
)
def test_invalid_sandbox_configuration(sandbox):
    with pytest.raises(ConfigError, match="invalide"):
        load_config(overrides={"sandbox": sandbox})

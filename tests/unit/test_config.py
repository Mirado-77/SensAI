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

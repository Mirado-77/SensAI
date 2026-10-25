"""A4: versioned system prompts, comparison and rollback."""

import pytest
from tests.unit.test_cli import chat_setup
from tests.unit.test_runtime import build, events_of

from sensai.cli.chat import run_chat
from sensai.cli.main import build_parser, config_from_args, main
from sensai.core.config import load_config
from sensai.core.errors import ConfigError
from sensai.core.fakes import FakeModelGateway
from sensai.core.prompts import (
    PROMPTS_DIR,
    PromptVersion,
    diff_versions,
    list_versions,
    load_version,
    system_prompt,
)
from sensai.core.types import Chunk
from sensai.db.connection import connect
from sensai.observability.prompt_stats import prompt_stats

pytestmark = pytest.mark.feature("A4")


def write_versions(directory, **texts):
    for name, text in texts.items():
        (directory / f"{name}.txt").write_text(text, encoding="utf-8")


def test_a4_versions_are_listed_in_natural_order(tmp_path):
    write_versions(tmp_path, v10="ten", v2="two", v1="one")
    assert [version.name for version in list_versions(tmp_path)] == ["v1", "v2", "v10"]


def test_a4_unknown_or_unsafe_names_are_refused(tmp_path):
    write_versions(tmp_path, v1="one")
    with pytest.raises(ConfigError, match="connues : v1"):
        load_version("v9", tmp_path)
    with pytest.raises(ConfigError, match="invalide"):
        load_version("../secrets", tmp_path)


def test_a4_an_edited_version_gets_a_new_sha():
    first, edited = PromptVersion("v1", "Be short."), PromptVersion("v1", "Be very short.")
    assert first.label == f"v1@{first.sha}" and len(first.sha) == 8
    assert first.sha != edited.sha


def test_a4_diff_shows_the_changed_lines():
    diff = diff_versions(PromptVersion("v1", "a\nb"), PromptVersion("v2", "a\nc"))
    assert "-b" in diff and "+c" in diff and "v1@" in diff and "v2@" in diff


def test_a4_base_prompt_comes_before_the_profile():
    version = PromptVersion("v1", "Base.")
    assert system_prompt(version, "Profile.") == "Base.\n\nProfile."
    assert system_prompt(version, None) == "Base."
    assert system_prompt(None, None) is None


def test_a4_shipped_versions_load_and_the_default_is_the_latest():
    names = [version.name for version in list_versions(PROMPTS_DIR)]
    assert names[:2] == ["v1", "v2"]
    assert load_config().prompts.system == names[-1]
    assert all(version.text for version in list_versions(PROMPTS_DIR))


async def test_a4_each_run_records_its_prompt_version():
    s = build([Chunk("Hi")])
    s.runtime.prompt_version = "v2@12345678"
    await s.runtime.run_turn(s.session_id, "Hi")
    assert events_of(s, "run.started")[0].payload["prompt_version"] == "v2@12345678"


async def test_a4_chat_sends_the_version_and_can_roll_back(tmp_path):
    config, terminal, out, err = chat_setup(
        tmp_path, "hi", "/prompt v1", "again", prompts={"system": "v2"}
    )
    gateway = FakeModelGateway([Chunk("one")], [Chunk("two")])
    assert await run_chat(config, terminal, gateway) == 0
    v1, v2 = load_version("v1"), load_version("v2")
    assert gateway.calls[0].messages[0].content == v2.text
    assert gateway.calls[1].messages[0].content == v1.text
    assert f"prompt: {v2.label}" in out.getvalue()
    assert f"System prompt: {v1.label}" in out.getvalue()
    assert err.getvalue() == ""
    connection = connect(config.storage.db_path)
    rows = {row["version"]: row for row in prompt_stats(connection)}
    assert rows[v2.label]["runs"] == 1 and rows[v1.label]["runs"] == 1
    assert rows[v1.label]["completed"] == 1


async def test_a4_unknown_version_in_the_chat_keeps_the_active_one(tmp_path):
    config, terminal, out, err = chat_setup(tmp_path, "/prompt v99", "hi", prompts={"system": "v1"})
    gateway = FakeModelGateway([Chunk("one")])
    await run_chat(config, terminal, gateway)
    assert "introuvable : v99" in err.getvalue()
    assert gateway.calls[0].messages[0].content == load_version("v1").text


async def test_a4_profile_is_added_after_the_base_prompt(tmp_path):
    lines = ("/profile instructions Answer in French", "hi")
    config, terminal, _, _ = chat_setup(tmp_path, *lines, prompts={"system": "v1"})
    gateway = FakeModelGateway([Chunk("salut")])
    await run_chat(config, terminal, gateway)
    system = gateway.calls[0].messages[0].content
    assert system.startswith(load_version("v1").text)
    assert system.endswith("Instructions: Answer in French")


async def test_a4_an_unknown_configured_version_stops_the_chat(tmp_path):
    config, terminal, _, err = chat_setup(tmp_path, prompts={"system": "v99"})
    assert await run_chat(config, terminal, FakeModelGateway()) == 2
    assert "introuvable : v99" in err.getvalue()


def test_a4_prompt_option_selects_the_version():
    config = config_from_args(build_parser().parse_args(["chat", "--prompt", "v1"]))
    assert config.prompts.system == "v1"


def test_a4_prompts_list_and_diff_commands(capsys):
    assert main(["prompts"]) == 0
    listed = capsys.readouterr().out
    assert f"* {load_version('v2').label}" in listed and f"  {load_version('v1').label}" in listed
    assert main(["prompts", "diff", "v1", "v2"]) == 0
    assert "+When tests fail" in capsys.readouterr().out
    assert main(["prompts", "diff", "v1", "v99"]) == 2


async def test_a4_prompts_stats_compares_the_recorded_runs(tmp_path, capsys):
    path = tmp_path / "c.toml"
    db = tmp_path / "sensai.db"
    path.write_text(f'[storage]\ndb_path = "{db}"\n[react]\nenabled = false\n', encoding="utf-8")
    assert main(["prompts", "--config", str(path), "stats"]) == 0
    assert "No run recorded yet." in capsys.readouterr().out
    _, terminal, _, _ = chat_setup(tmp_path, "hi")
    await run_chat(load_config(path), terminal, FakeModelGateway([Chunk("one")]))
    assert main(["prompts", "--config", str(path), "stats"]) == 0
    table = capsys.readouterr().out
    assert "version" in table and load_version("v2").label in table


@pytest.mark.parametrize("name", ["../x", "v 1"])
def test_a4_invalid_names_are_rejected_by_the_config(name):
    with pytest.raises(ConfigError):
        load_config(overrides={"prompts": {"system": name}})

"""A5: personas in configuration, switched in the chat without losing the history."""

import pytest
from tests.unit.test_cli import chat_setup
from tests.unit.test_runtime import build, events_of

from sensai.cli.chat import print_tools, run_chat
from sensai.cli.main import build_parser, config_from_args
from sensai.cli.terminal import Terminal
from sensai.core.errors import ConfigError
from sensai.core.fakes import FakeModelGateway, FakeTool
from sensai.core.personas import PERSONAS_DIR, list_personas, load_persona
from sensai.core.prompts import PromptVersion, system_prompt
from sensai.core.types import Chunk, ToolCall
from sensai.tools.registry import ToolRegistry

pytestmark = pytest.mark.feature("A5")


def write_persona(directory, name, text):
    (directory / f"{name}.toml").write_text(text, encoding="utf-8")


def test_a5_shipped_personas_load_and_the_reviewer_only_reads():
    personas = {persona.name: persona for persona in list_personas(PERSONAS_DIR)}
    assert {"dev", "reviewer"} <= set(personas)
    assert personas["dev"].risks == {"read", "write", "exec"}
    assert personas["reviewer"].risks == {"read"}
    assert all(persona.prompt and persona.description for persona in personas.values())


def test_a5_risks_default_to_all_levels(tmp_path):
    write_persona(tmp_path, "free", 'prompt = "Be free."\n')
    persona = load_persona("free", tmp_path)
    assert persona.risks == {"read", "write", "exec"} and persona.description == ""


@pytest.mark.parametrize(
    "text, message",
    [
        ('prompt = ""\n', "texte non vide"),
        ("description = 'x'\n", "texte non vide"),
        ('prompt = "x"\nrisks = ["read", "delete"]\n', "`risks`"),
        ('prompt = "x"\nrisks = "read"\n', "`risks`"),
        ('prompt = "x"\ntools = []\n', "clés inconnues : tools"),
        ('prompt = "x"\ndescription = 3\n', "`description`"),
        ("prompt = \n", "TOML invalide"),
    ],
)
def test_a5_invalid_persona_files_are_refused(tmp_path, text, message):
    write_persona(tmp_path, "bad", text)
    with pytest.raises(ConfigError, match=message):
        load_persona("bad", tmp_path)


def test_a5_unknown_or_unsafe_names_are_refused(tmp_path):
    write_persona(tmp_path, "dev", 'prompt = "x"\n')
    with pytest.raises(ConfigError, match="connues : dev"):
        load_persona("ops", tmp_path)
    with pytest.raises(ConfigError, match="invalide"):
        load_persona("../secrets", tmp_path)


def test_a5_persona_comes_between_the_base_prompt_and_the_profile():
    version = PromptVersion("v1", "Base.")
    assert system_prompt(version, "Profile.", "Persona.") == "Base.\n\nPersona.\n\nProfile."
    assert system_prompt(None, None, "Persona.") == "Persona."


async def test_a5_runtime_offers_only_the_allowed_risks_and_records_the_persona():
    tools = [FakeTool("read_file"), FakeTool("write_file", risk="write")]
    s = build([Chunk("ok")], [Chunk("ok")], tools=tools)
    s.runtime.persona, s.runtime.tool_risks = "reviewer", frozenset({"read"})
    await s.runtime.run_turn(s.session_id, "Review")
    assert [spec.name for spec in s.gateway.calls[0].tools] == ["read_file"]
    assert events_of(s, "run.started")[0].payload["persona"] == "reviewer"
    s.runtime.persona, s.runtime.tool_risks = None, None
    await s.runtime.run_turn(s.session_id, "Fix")
    assert [spec.name for spec in s.gateway.calls[1].tools] == ["read_file", "write_file"]


async def test_a5_a_call_outside_the_persona_is_refused_without_running():
    writer = FakeTool("write_file", risk="write")
    call = ToolCall("write_file", {})
    s = build([call], [Chunk("I cannot write.")], tools=[writer])
    s.runtime.persona, s.runtime.tool_risks = "reviewer", frozenset({"read"})
    result = await s.runtime.run_turn(s.session_id, "Write it anyway")
    assert result.status == "completed" and writer.calls == []
    failed = events_of(s, "tool.failed")[0].payload
    assert (failed["tool"], failed["error"]) == ("write_file", "not allowed")
    tool_message = s.gateway.calls[1].messages[-1]
    assert "not available to the reviewer persona" in tool_message.content


async def test_a5_the_chat_starts_with_the_configured_persona(tmp_path):
    config, terminal, out, _ = chat_setup(tmp_path, "hi", prompts={"persona": "reviewer"})
    gateway = FakeModelGateway([Chunk("one")])
    assert await run_chat(config, terminal, gateway) == 0
    assert gateway.calls[0].messages[0].content == load_persona("reviewer").prompt
    assert "persona: reviewer (/persona)" in out.getvalue()


async def test_a5_switching_keeps_the_history_and_changes_prompt_and_tools(tmp_path):
    lines = ("fix the bug", "/persona reviewer", "review it", "/persona dev", "again")
    files = {"allowed_roots": [{"path": str(tmp_path), "mode": "write"}]}
    config, terminal, out, err = chat_setup(
        tmp_path, *lines, prompts={"persona": "dev"}, files=files
    )
    gateway = FakeModelGateway([Chunk("fixed")], [Chunk("looks good")], [Chunk("done")])
    assert await run_chat(config, terminal, gateway) == 0
    dev, reviewer = load_persona("dev"), load_persona("reviewer")
    first, second, third = gateway.calls
    assert first.messages[0].content == dev.prompt
    assert second.messages[0].content == reviewer.prompt
    assert [(m.role, m.content) for m in second.messages[1:]] == [
        ("user", "fix the bug"),
        ("assistant", "fixed"),
        ("user", "review it"),
    ]
    assert third.messages[0].content == dev.prompt and len(third.messages) == 6
    assert {"read_file", "write_file", "edit_file"} <= {spec.name for spec in first.tools}
    reviewer_tools = {spec.name for spec in second.tools}
    assert "read_file" in reviewer_tools and not reviewer_tools & {"write_file", "edit_file"}
    assert {spec.name for spec in third.tools} == {spec.name for spec in first.tools}
    assert "Persona: reviewer [read]" in out.getvalue() and err.getvalue() == ""


async def test_a5_persona_command_lists_and_an_unknown_name_keeps_the_active_one(tmp_path):
    lines = ("/persona", "/persona ops", "hi")
    config, terminal, out, err = chat_setup(tmp_path, *lines, prompts={"persona": "dev"})
    gateway = FakeModelGateway([Chunk("one")])
    await run_chat(config, terminal, gateway)
    assert "* dev [read, write, exec]" in out.getvalue()
    assert "  reviewer [read]" in out.getvalue()
    assert "Persona introuvable : ops" in err.getvalue()
    assert gateway.calls[0].messages[0].content == load_persona("dev").prompt


async def test_a5_an_unknown_configured_persona_stops_the_chat(tmp_path):
    config, terminal, _, err = chat_setup(tmp_path, prompts={"persona": "ops"})
    assert await run_chat(config, terminal, FakeModelGateway()) == 2
    assert "Persona introuvable : ops" in err.getvalue()


def test_a5_persona_option_selects_the_persona():
    args = build_parser().parse_args(["chat", "--persona", "reviewer", "--prompt", "v1"])
    config = config_from_args(args)
    assert (config.prompts.persona, config.prompts.system) == ("reviewer", "v1")
    assert config_from_args(build_parser().parse_args(["chat"])).prompts.persona == "dev"


def test_a5_tools_command_shows_only_the_tools_of_the_persona(capsys):
    from rich.console import Console

    registry = ToolRegistry([FakeTool("read_file"), FakeTool("run_command", risk="exec")])
    terminal = Terminal(Console(highlight=False), Console(stderr=True, highlight=False))
    print_tools(terminal, registry, frozenset({"read"}))
    out = capsys.readouterr().out
    assert "read_file" in out and "run_command" not in out

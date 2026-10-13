import pytest
from tests.unit.test_cli import chat_setup, make_terminal

from sensai.cli.chat import LAST_SESSION, print_sessions, run_chat
from sensai.cli.main import build_parser
from sensai.cli.profile import profile_prompt
from sensai.core.fakes import FakeModelGateway
from sensai.core.types import Chunk

pytestmark = pytest.mark.feature("M1")


def roles(call):
    return [(m.role, m.content) for m in call.messages]


async def test_the_profile_is_sent_every_turn_and_kept_across_sessions(tmp_path):
    config, terminal, out, _ = chat_setup(
        tmp_path,
        "/profile instructions Answer in Python.",
        "/profile set allergy penicillin",
        "hi",
    )
    first = FakeModelGateway([Chunk("ok")])
    assert await run_chat(config, terminal, first) == 0
    [(role, prompt), *_] = roles(first.calls[0])
    assert role == "system"
    assert "Instructions: Answer in Python." in prompt
    assert "- allergy: penicillin" in prompt

    _, terminal, _, _ = chat_setup(tmp_path, "again")
    second = FakeModelGateway([Chunk("ok")])
    await run_chat(config, terminal, second)
    assert roles(second.calls[0]) == [("system", prompt), ("user", "again")]


async def test_profile_unset_and_clear(tmp_path):
    config, terminal, _, _ = chat_setup(
        tmp_path,
        "/profile set a 1",
        "/profile set b 2",
        "/profile unset a",
        "x",
        "/profile clear",
        "y",
    )
    gateway = FakeModelGateway([Chunk("1")], [Chunk("2")])
    await run_chat(config, terminal, gateway)
    assert roles(gateway.calls[0])[0] == ("system", profile_prompt({"preferences": {"b": "2"}}))
    assert roles(gateway.calls[1])[0][0] == "user"


@pytest.mark.parametrize("command", ["/profile nope", "/profile set onlykey", "/profile unset z"])
async def test_an_invalid_profile_command_shows_the_help(tmp_path, command):
    config, terminal, _, err = chat_setup(tmp_path, command)
    await run_chat(config, terminal, FakeModelGateway())
    assert "/profile set <key> <value>" in err.getvalue()


def test_an_empty_profile_adds_no_system_prompt():
    assert profile_prompt({}) is None
    assert profile_prompt({"instructions": "", "preferences": {}}) is None


async def test_resume_continues_the_last_session_or_one_given_by_id_prefix(tmp_path):
    config, terminal, _, _ = chat_setup(tmp_path, "a")
    await run_chat(config, terminal, FakeModelGateway([Chunk("one")]))
    _, terminal, _, _ = chat_setup(tmp_path, "other")
    await run_chat(config, terminal, FakeModelGateway([Chunk("two")]))

    _, terminal, out, _ = chat_setup(tmp_path, "b")
    gateway = FakeModelGateway([Chunk("three")])
    assert await run_chat(config, terminal, gateway, resume=LAST_SESSION) == 0
    assert "Resumed session" in out.getvalue()
    assert roles(gateway.calls[0]) == [("user", "other"), ("assistant", "two"), ("user", "b")]

    listing, _, _ = make_terminal()
    print_sessions(config, listing)
    lines = listing.console.file.getvalue().splitlines()
    assert [line.split()[-1] for line in lines] == ["other", "a"]
    prefix = lines[1].split()[0]
    _, terminal, _, _ = chat_setup(tmp_path, "c")
    gateway = FakeModelGateway([Chunk("four")])
    await run_chat(config, terminal, gateway, resume=prefix)
    assert roles(gateway.calls[0])[:2] == [("user", "a"), ("assistant", "one")]


async def test_resuming_an_unknown_session_ends_with_exit_code_2(tmp_path):
    config, terminal, _, err = chat_setup(tmp_path, "hi")
    gateway = FakeModelGateway()
    assert await run_chat(config, terminal, gateway, resume="ffff") == 2
    assert "session introuvable : ffff" in err.getvalue()
    assert gateway.calls == []


def test_parser_reads_resume_and_sessions():
    parser = build_parser()
    assert parser.parse_args(["chat"]).resume is None
    assert parser.parse_args(["chat", "--resume"]).resume == LAST_SESSION
    assert parser.parse_args(["chat", "--resume", "ab12"]).resume == "ab12"
    assert parser.parse_args(["sessions"]).command == "sessions"


async def test_resume_without_any_saved_session(tmp_path):
    config, terminal, _, err = chat_setup(tmp_path)
    assert await run_chat(config, terminal, FakeModelGateway(), resume=LAST_SESSION) == 2
    assert "aucune session enregistrée" in err.getvalue()

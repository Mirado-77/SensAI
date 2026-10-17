"""X4: structured questions when the request is ambiguous."""

import pytest
from tests.unit.test_cli import chat_setup, make_terminal, scripted
from tests.unit.test_runtime import build, events_of, saved

from sensai.cli.channel import CliUserChannel
from sensai.cli.chat import run_chat
from sensai.core.config import QuestionsConfig, load_config
from sensai.core.errors import ConfigError
from sensai.core.fakes import FakeModelGateway
from sensai.core.types import Chunk, ToolCall

pytestmark = pytest.mark.feature("X4")

QUESTIONS = QuestionsConfig(enabled=True, max_per_turn=2)
ASK = ToolCall("ask_user", {"question": "Which file?", "options": ["a.py", "b.py"]})


def ask_setup(*scripts, questions=QUESTIONS, **kwargs):
    return build(*scripts, questions=questions, **kwargs)


async def test_x4_the_model_can_ask_and_reads_the_answer():
    s = ask_setup([ASK], [Chunk("Fixing b.py")], answers=[{"answer": "b.py"}])
    result = await s.runtime.run_turn(s.session_id, "Fix the file")
    assert result.answer == "Fixing b.py"
    assert s.channel.questions == [{"question": "Which file?", "options": ["a.py", "b.py"]}]
    requested = events_of(s, "question.requested")[0].payload
    answered = events_of(s, "question.answered")[0].payload
    assert requested["question"] == "Which file?" and requested["options"] == ["a.py", "b.py"]
    assert answered == {"call_id": requested["call_id"], "answer": "b.py"}
    tool_message = s.gateway.calls[1].messages[-1]
    assert (tool_message.role, tool_message.content) == ("tool", "The user answered: b.py")


async def test_x4_the_tool_is_offered_only_when_enabled():
    s = ask_setup([Chunk("Hi")])
    await s.runtime.run_turn(s.session_id, "Hi")
    assert [spec.name for spec in s.gateway.calls[0].tools] == ["ask_user"]
    off = ask_setup([Chunk("Hi")], questions=QuestionsConfig(enabled=False, max_per_turn=2))
    await off.runtime.run_turn(off.session_id, "Hi")
    assert off.gateway.calls[0].tools is None


async def test_x4_the_question_and_answer_are_saved_in_the_session():
    s = ask_setup([ASK], [Chunk("Done")], answers=[{"answer": "a.py"}])
    await s.runtime.run_turn(s.session_id, "Fix it")
    assert ("tool", "The user answered: a.py") in saved(s)


async def test_x4_no_answer_tells_the_model_to_assume():
    s = ask_setup([ASK], [Chunk("Assuming a.py")])
    await s.runtime.run_turn(s.session_id, "Fix it")
    assert "did not answer" in s.gateway.calls[1].messages[-1].content


async def test_x4_invalid_question_is_not_asked():
    bad = ToolCall("ask_user", {"options": ["a"]})
    s = ask_setup([bad], [Chunk("Sorry")])
    await s.runtime.run_turn(s.session_id, "Fix it")
    assert s.channel.questions == []
    assert events_of(s, "tool.failed")[0].payload["error"] == "invalid arguments"


async def test_x4_questions_are_bounded_per_turn():
    one = QuestionsConfig(enabled=True, max_per_turn=1)
    s = ask_setup([ASK], [ASK], [Chunk("Ok")], questions=one, answers=[{"answer": "a.py"}])
    result = await s.runtime.run_turn(s.session_id, "Fix it")
    assert result.status == "completed"
    assert len(s.channel.questions) == 1
    assert events_of(s, "tool.failed")[0].payload["error"] == "too many questions"


async def test_x4_cli_channel_picks_an_option_by_number():
    terminal, out, _ = make_terminal(scripted("2"))
    answer = await CliUserChannel(terminal).ask({"question": "Which?", "options": ["a", "b"]})
    assert answer == {"answer": "b"}
    assert "Question: Which?" in out.getvalue() and "2. b" in out.getvalue()


@pytest.mark.parametrize(("line", "expected"), [("other.py", "other.py"), ("9", "9")])
async def test_x4_cli_channel_accepts_free_text(line, expected):
    terminal, _, _ = make_terminal(scripted(line))
    answer = await CliUserChannel(terminal).ask({"question": "Which?", "options": ["a"]})
    assert answer == {"answer": expected}


async def test_x4_cli_channel_end_of_input_is_no_answer():
    terminal, _, _ = make_terminal(scripted())
    assert await CliUserChannel(terminal).ask({"question": "Which?", "options": []}) == {
        "answer": ""
    }


async def test_x4_chat_asks_and_continues(tmp_path):
    questions = {"enabled": True, "max_per_turn": 2}
    config, terminal, out, err = chat_setup(tmp_path, "fix it", "1", questions=questions)
    gateway = FakeModelGateway([ASK], [Chunk("Fixing a.py")])
    assert await run_chat(config, terminal, gateway) == 0
    assert "Question: Which file?" in out.getvalue()
    assert "sensai> Fixing a.py" in out.getvalue()
    assert err.getvalue() == ""


def test_x4_questions_are_enabled_by_default():
    questions = load_config().questions
    assert questions.enabled and questions.max_per_turn == 2


@pytest.mark.parametrize("value", [0, 6])
def test_x4_max_per_turn_is_bounded(value):
    with pytest.raises(ConfigError):
        load_config(overrides={"questions": {"max_per_turn": value}})

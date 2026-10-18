import pytest

from sensai.core.config import RuntimeConfig, load_config
from sensai.core.fakes import (
    CollectingEventSink,
    FakeModelGateway,
    FakeUserChannel,
    InMemorySessionStore,
)
from sensai.core.runtime import AgentRuntime, RuntimeDeps
from sensai.core.types import Chunk, ToolCall
from sensai.safety.policy import ApprovalPolicy
from sensai.tools import native_tools
from sensai.tools.registry import ToolRegistry

pytestmark = pytest.mark.feature("A3")

LIMITS = {"max_steps": 8, "max_tool_calls": 8, "timeout_s": 5, "max_tokens": 1000}
RULES = {"read": "allow", "write": "ask", "exec": "ask"}


class InspectingChannel:
    """Approves everything and records what is on disk at the moment the user is asked."""

    def __init__(self, watched):
        self._watched = watched
        self.requests = []
        self.disk_when_asked = []

    async def confirm(self, request):
        self.requests.append(request)
        exists = self._watched.exists()
        self.disk_when_asked.append(self._watched.read_text(encoding="utf-8") if exists else None)
        return True

    async def ask(self, question):
        raise NotImplementedError


class Setup:
    def __init__(self, root, scripts, channel):
        config = load_config(
            overrides={"files": {"allowed_roots": [{"path": str(root), "mode": "write"}]}}
        )
        self.gateway = FakeModelGateway(*scripts)
        self.sink = CollectingEventSink()
        self.channel = channel
        store = InMemorySessionStore()
        deps = RuntimeDeps(
            gateway=self.gateway,
            store=store,
            sink=self.sink,
            channel=channel,
            registry=ToolRegistry(native_tools(config)),
            policy=ApprovalPolicy(RULES),
        )
        self.runtime = AgentRuntime(deps, RuntimeConfig(**LIMITS), "test-model", None)
        self.session_id = store.create_session()

    async def run(self, prompt="Go"):
        return await self.runtime.run_turn(self.session_id, prompt)

    def events(self, type_):
        return [event for event in self.sink.events if event.type == type_]

    def feedback_to_the_model(self):
        return self.gateway.calls[1].messages[-1].content


@pytest.fixture
def root(tmp_path):
    folder = tmp_path / "root"
    folder.mkdir()
    return folder


def write_call(path, content):
    return ToolCall("write_file", {"path": path, "content": content})


def edit_call(path, old_text, new_text):
    return ToolCall("edit_file", {"path": path, "old_text": old_text, "new_text": new_text})


async def test_the_user_sees_a_creation_diff_before_a_new_file_is_written(root):
    channel = FakeUserChannel(approvals=[True])
    s = Setup(root, [[write_call("a.py", "x = 1\n")], [Chunk("done")]], channel)
    await s.run()

    lines = channel.confirm_requests[0].preview.split("\n")
    assert lines[:3] == ["--- /dev/null", "+++ b/a.py", "@@ -0,0 +1 @@"]
    assert "+x = 1" in lines
    assert (root / "a.py").read_text(encoding="utf-8") == "x = 1\n"


async def test_the_user_sees_a_modification_diff_for_an_existing_file(root):
    (root / "a.py").write_text("x = 1\ny = 2\n", encoding="utf-8")
    channel = FakeUserChannel(approvals=[True])
    s = Setup(root, [[write_call("a.py", "x = 1\ny = 3\n")], [Chunk("done")]], channel)
    await s.run()

    lines = channel.confirm_requests[0].preview.split("\n")
    assert lines[:2] == ["--- a/a.py", "+++ b/a.py"]
    assert "-y = 2" in lines
    assert "+y = 3" in lines
    assert (root / "a.py").read_text(encoding="utf-8") == "x = 1\ny = 3\n"


async def test_nothing_is_written_when_the_user_refuses(root):
    (root / "a.py").write_text("old\n", encoding="utf-8")
    channel = FakeUserChannel(approvals=[False])
    s = Setup(root, [[write_call("a.py", "new\n")], [Chunk("understood")]], channel)
    result = await s.run()

    assert result.status == "completed"
    lines = channel.confirm_requests[0].preview.split("\n")
    assert "-old" in lines
    assert "+new" in lines
    assert (root / "a.py").read_text(encoding="utf-8") == "old\n"
    assert s.feedback_to_the_model() == "The user refused this action."
    assert "tool.started" not in [event.type for event in s.sink.events]


async def test_a_new_file_does_not_exist_yet_when_the_user_is_asked(root):
    channel = InspectingChannel(root / "a.py")
    s = Setup(root, [[write_call("a.py", "x = 1\n")], [Chunk("done")]], channel)
    await s.run()

    assert channel.disk_when_asked == [None]
    assert (root / "a.py").read_text(encoding="utf-8") == "x = 1\n"


async def test_an_edit_does_not_change_the_file_before_the_user_answers(root):
    (root / "a.py").write_text("x = 1\ny = 2\n", encoding="utf-8")
    channel = InspectingChannel(root / "a.py")
    s = Setup(root, [[edit_call("a.py", "y = 2", "y = 3")], [Chunk("done")]], channel)
    await s.run()

    assert channel.disk_when_asked == ["x = 1\ny = 2\n"]
    lines = channel.requests[0].preview.split("\n")
    assert "-y = 2" in lines
    assert "+y = 3" in lines
    assert (root / "a.py").read_text(encoding="utf-8") == "x = 1\ny = 3\n"


async def test_the_approval_event_in_the_journal_carries_the_diff(root):
    channel = FakeUserChannel(approvals=[True])
    s = Setup(root, [[write_call("a.py", "x = 1\n")], [Chunk("done")]], channel)
    await s.run()

    preview = s.events("approval.requested")[0].payload["preview"]
    assert preview == channel.confirm_requests[0].preview
    assert preview.startswith("--- /dev/null")


async def test_an_identical_write_is_announced_as_no_change(root):
    (root / "a.py").write_text("same\n", encoding="utf-8")
    channel = FakeUserChannel(approvals=[True])
    s = Setup(root, [[write_call("a.py", "same\n")], [Chunk("done")]], channel)
    await s.run()

    assert channel.confirm_requests[0].preview == "(aucun changement)"


async def test_a_path_outside_the_root_gets_the_generic_preview_and_is_refused_by_the_tool(root):
    channel = FakeUserChannel(approvals=[True])
    call = write_call("../escaped.py", "x = 1\n")
    s = Setup(root, [[call], [Chunk("done")]], channel)
    await s.run()

    assert channel.confirm_requests[0].preview.startswith("write_file(")
    assert "hors des racines" in s.feedback_to_the_model()
    assert not (root.parent / "escaped.py").exists()


async def test_an_edit_on_a_missing_text_gets_the_generic_preview_and_the_model_is_told(root):
    (root / "a.py").write_text("x = 1\n", encoding="utf-8")
    channel = FakeUserChannel(approvals=[True])
    s = Setup(root, [[edit_call("a.py", "absent", "y")], [Chunk("done")]], channel)
    await s.run()

    assert channel.confirm_requests[0].preview.startswith("edit_file(")
    assert "introuvable" in s.feedback_to_the_model()
    assert (root / "a.py").read_text(encoding="utf-8") == "x = 1\n"


async def test_a_long_write_is_truncated_in_the_approval_request(root):
    content = "".join(f"line {i}\n" for i in range(500))
    channel = FakeUserChannel(approvals=[True])
    s = Setup(root, [[write_call("big.txt", content)], [Chunk("done")]], channel)
    await s.run()

    lines = channel.confirm_requests[0].preview.split("\n")
    assert len(lines) <= 201
    assert lines[-1].endswith("lignes de plus")

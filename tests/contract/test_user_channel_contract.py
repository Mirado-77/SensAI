from sensai.core.contracts import UserChannel
from sensai.core.types import ApprovalRequest

REQUEST = ApprovalRequest("r1", "write_file", "write", "--- a\n+++ b", {"path": "a.py"})


def test_satisfies_the_protocol(channel):
    assert isinstance(channel, UserChannel)


async def test_confirm_returns_a_bool(channel):
    assert isinstance(await channel.confirm(REQUEST), bool)


async def test_ask_returns_a_dict(channel):
    assert isinstance(await channel.ask({"question": "Which file?"}), dict)

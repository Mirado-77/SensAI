import json
import typing

from sensai.core.contracts import Tool
from sensai.core.types import Risk, ToolContext, ToolResult

CTX = ToolContext(run_id="r1", session_id="s1")


def test_satisfies_the_protocol(tool):
    assert isinstance(tool, Tool)


def test_has_a_name_and_a_description(tool):
    assert tool.name
    assert tool.description


def test_risk_is_valid(tool):
    assert tool.risk in typing.get_args(Risk)


def test_parameters_is_an_object_schema(tool):
    assert tool.parameters["type"] == "object"


def test_parameters_are_json_serializable(tool):
    json.dumps(tool.parameters)


async def test_run_returns_a_tool_result(tool):
    result = await tool.run({}, CTX)
    assert isinstance(result, ToolResult)

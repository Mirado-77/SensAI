import pytest

from sensai.core.fakes import FakeTool
from sensai.core.types import ToolSpec
from sensai.tools.registry import ToolRegistry


def test_get_returns_a_registered_tool_or_none():
    tool = FakeTool(name="read_file")
    registry = ToolRegistry([tool])
    assert registry.get("read_file") is tool
    assert registry.get("missing") is None


def test_specs_describe_the_tools_as_the_model_sees_them():
    parameters = {"type": "object", "properties": {"path": {"type": "string"}}}
    tool = FakeTool(name="read_file", description="Read.", parameters=parameters)
    registry = ToolRegistry([tool])
    assert registry.specs() == [ToolSpec("read_file", "Read.", parameters)]


def test_registering_a_duplicate_name_is_rejected():
    registry = ToolRegistry([FakeTool(name="read_file")])
    with pytest.raises(ValueError, match="read_file"):
        registry.register(FakeTool(name="read_file"))


def test_an_empty_registry_has_no_specs():
    registry = ToolRegistry()
    assert len(registry) == 0
    assert registry.specs() == []


@pytest.mark.feature("T5")
def test_a_tool_with_an_invalid_parameters_schema_is_rejected():
    with pytest.raises(ValueError, match="Schéma de paramètres invalide pour broken"):
        ToolRegistry([FakeTool(name="broken", parameters={"type": "nope"})])


@pytest.mark.feature("T5")
def test_tool_parameters_must_be_an_object_schema():
    with pytest.raises(ValueError, match="type object"):
        ToolRegistry([FakeTool(name="listy", parameters={"type": "array"})])

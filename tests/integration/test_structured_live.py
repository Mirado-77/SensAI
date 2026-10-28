"""Structured output (T5) against a real Ollama server. Run with `make test-integration`."""

import os

import httpx
import pytest
from pydantic import BaseModel, Field

from sensai.core.config import load_config
from sensai.core.errors import ModelNotFoundError
from sensai.core.types import Message
from sensai.model.ollama import OllamaGateway
from sensai.model.structured import generate_structured

pytestmark = [pytest.mark.integration, pytest.mark.feature("T5")]


class Step(BaseModel):
    description: str = Field(min_length=1)


class Plan(BaseModel):
    goal: str = Field(min_length=1)
    steps: list[Step] = Field(min_length=1, max_length=5)


@pytest.fixture
def gateway():
    base = load_config()
    try:
        httpx.get(f"{base.model.host}/api/version", timeout=2).raise_for_status()
    except httpx.HTTPError:
        pytest.skip(f"No Ollama server at {base.model.host}")
    model = os.environ.get("SENSAI_TEST_MODEL", base.model.name)
    return OllamaGateway(base.model.host, model, model.temperature, timeout_s=180)


async def test_the_model_returns_a_plan_that_conforms_to_the_schema(gateway):
    question = [Message("user", "Plan how to fix a failing unit test in a Python project.")]
    try:
        result = await generate_structured(gateway, question, Plan)
    except ModelNotFoundError as exc:
        pytest.skip(str(exc))
    assert isinstance(result.value, Plan)
    assert result.value.steps
    assert result.repairs <= 1

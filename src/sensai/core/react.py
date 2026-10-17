"""Bounded ReAct loop (A1): a short structured plan before the runtime acts.

The plan is asked with structured output (T5), so it is a JSON object and never free reasoning:
a goal and at most `max_plan_steps` short steps. It is then given to the model as a system
message for the rest of the turn; it is not saved in the session. The action / observation part
is the runtime's tool loop, bounded by `[runtime] max_steps`, which emits one event per phase.

Event payloads (the caller adds `run_id`):

- react.plan: goal, steps
- react.action: step, tools
- react.observation: step, results (list of {tool, is_error})
"""

from dataclasses import dataclass, field
from typing import Any

from sensai.core.contracts import ModelGateway
from sensai.core.types import Message, Usage
from sensai.model.structured import OnEvent, generate_structured

MAX_GOAL_CHARS = 200
MAX_STEP_CHARS = 160

_PLAN_INSTRUCTION = (
    "Before acting, plan. Give the goal of the user's last message in one sentence and at most "
    "{max_steps} short steps, one tool action each. Use an empty list of steps if no tool is "
    "needed. Available tools: {tools}."
)
_FOLLOW_INSTRUCTION = (
    "Plan for this turn:\n{steps}\n"
    "Do one step at a time: call the tool, read its result, then go on. If a result shows the "
    "plan is wrong (error, failing test), fix it and retry. Keep your text short."
)


@dataclass(frozen=True)
class Plan:
    goal: str
    steps: list[str]
    usage: list[Usage] = field(default_factory=list)

    def message(self) -> Message | None:
        """The system message that makes the model follow the plan, None if there is no step."""
        if not self.steps:
            return None
        listed = "\n".join(f"{index}. {step}" for index, step in enumerate(self.steps, 1))
        return Message("system", _FOLLOW_INSTRUCTION.format(steps=listed))


def plan_schema(max_steps: int) -> dict[str, Any]:
    text = {"type": "string", "minLength": 1}
    return {
        "type": "object",
        "properties": {
            "goal": {**text, "maxLength": MAX_GOAL_CHARS},
            "steps": {
                "type": "array",
                "maxItems": max_steps,
                "items": {**text, "maxLength": MAX_STEP_CHARS},
            },
        },
        "required": ["goal", "steps"],
        "additionalProperties": False,
    }


async def make_plan(
    gateway: ModelGateway,
    messages: list[Message],
    tool_names: list[str],
    *,
    max_steps: int,
    max_repairs: int,
    on_event: OnEvent | None = None,
) -> Plan:
    """Ask the model for a plan. Raises StructuredOutputError if it never conforms."""
    tools = ", ".join(tool_names) or "none"
    instruction = Message("system", _PLAN_INSTRUCTION.format(max_steps=max_steps, tools=tools))
    result = await generate_structured(
        gateway,
        [*messages, instruction],
        plan_schema(max_steps),
        max_repairs=max_repairs,
        on_event=on_event,
    )
    return Plan(result.value["goal"], result.value["steps"], result.usage)

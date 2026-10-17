"""Structured questions (X4): the model asks the user instead of guessing.

The runtime offers the model one extra tool, `ask_user`, and handles it itself because it owns
the UserChannel (as for approvals): the arguments are validated against `QUESTION_SCHEMA`, the
question goes to `UserChannel.ask`, and the answer comes back to the model as the tool result.
The exchange is saved in the session like any tool call.

Event payloads (all events also carry `run_id`):

- question.requested: call_id, question, options
- question.answered: call_id, answer
"""

from typing import Any

from sensai.core.types import ToolResult, ToolSpec

ASK_TOOL = "ask_user"
MAX_OPTIONS = 5

QUESTION_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "question": {"type": "string", "minLength": 1, "maxLength": 200},
        "options": {
            "type": "array",
            "maxItems": MAX_OPTIONS,
            "items": {"type": "string", "minLength": 1, "maxLength": 80},
        },
    },
    "required": ["question"],
    "additionalProperties": False,
}

ASK_SPEC = ToolSpec(
    ASK_TOOL,
    "Ask the user one short question when the request is ambiguous and a wrong guess would "
    "waste work (which file, which option, which behaviour). Give up to "
    f"{MAX_OPTIONS} options when the possible answers are known. Never ask what another tool "
    "can find out.",
    QUESTION_SCHEMA,
)


def answer_result(answer: dict[str, Any]) -> ToolResult:
    """The tool result the model reads. An empty answer tells it to go on with an assumption."""
    text = str(answer.get("answer", "")).strip()
    if not text:
        return ToolResult("The user did not answer. Go on with a reasonable assumption and say it.")
    # The user's own words: data for the model, never instructions from a third party.
    return ToolResult(f"The user answered: {text}")


def limit_result(max_questions: int) -> ToolResult:
    return ToolResult(
        f"No more questions this turn (maximum {max_questions}). "
        "Go on with a reasonable assumption and say it.",
        is_error=True,
    )

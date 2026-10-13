"""JSON Schema helpers shared by structured output (T5) and tool argument validation."""

from typing import Any

import jsonschema

MAX_ERRORS_SHOWN = 5


def check_schema(schema: dict[str, Any]) -> None:
    """Raise `jsonschema.SchemaError` if `schema` is not a valid JSON Schema."""
    jsonschema.Draft202012Validator.check_schema(schema)


def schema_errors(schema: dict[str, Any], value: Any) -> list[str]:
    """Readable JSON Schema violations of `value`, at most MAX_ERRORS_SHOWN."""
    validator = jsonschema.Draft202012Validator(schema)
    found = sorted(validator.iter_errors(value), key=lambda error: list(error.absolute_path))
    return [_describe(error) for error in found[:MAX_ERRORS_SHOWN]]


def _describe(error: jsonschema.ValidationError) -> str:
    path = "/".join(str(part) for part in error.absolute_path)
    return f"{path or '(root)'}: {error.message}"

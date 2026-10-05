"""Approval policy: what happens before a tool runs, by risk level."""

import json
import typing
from collections.abc import Mapping
from typing import Any

from sensai.core.config import PolicyConfig
from sensai.core.contracts import Tool
from sensai.core.types import Decision, Risk


class ApprovalPolicy:
    def __init__(self, rules: Mapping[Risk, Decision]) -> None:
        missing = set(typing.get_args(Risk)) - set(rules)
        if missing:
            raise ValueError(f"Règles manquantes pour les niveaux de risque : {sorted(missing)}")
        self._rules = dict(rules)

    @classmethod
    def from_config(cls, config: PolicyConfig) -> "ApprovalPolicy":
        return cls({"read": config.read, "write": config.write, "exec": config.exec})

    def decide(self, tool: Tool) -> Decision:
        """Return "allow", "ask" or "deny". A tool with an unknown risk level is always asked."""
        return self._rules.get(tool.risk, "ask")


def preview_call(tool_name: str, args: dict[str, Any]) -> str:
    """Default text shown to the user before a tool runs."""
    return f"{tool_name}({json.dumps(args, ensure_ascii=False, sort_keys=True)})"

"""Small typed tool-calling primitives shared by GEMS management plans."""

from __future__ import annotations

import json
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

from pydantic import BaseModel, ConfigDict, ValidationError


class ToolArguments(BaseModel):
    """Base class for strict function-tool arguments."""

    model_config = ConfigDict(extra="forbid")


@dataclass(frozen=True)
class FunctionTool:
    """A JSON-schema function tool with local argument validation."""

    name: str
    description: str
    arguments_model: type[ToolArguments]
    handler: Callable[[ToolArguments], Any]
    terminal: bool = False

    def openai_schema(self) -> dict[str, Any]:
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": self.arguments_model.model_json_schema(),
                "strict": True,
            },
        }

    def invoke(self, raw_arguments: str | dict[str, Any]) -> dict[str, Any]:
        """Validate, execute, and normalize one tool result for the model."""
        try:
            if isinstance(raw_arguments, str):
                arguments = self.arguments_model.model_validate_json(raw_arguments)
            else:
                arguments = self.arguments_model.model_validate(raw_arguments)
            value = self.handler(arguments)
            return {"ok": True, "result": value}
        except (ValidationError, ValueError, KeyError) as error:
            return {
                "ok": False,
                "error": f"{type(error).__name__}: {error}",
            }


@dataclass
class ToolLoopResult:
    """Transcript and termination state of a management tool interaction."""

    completed: bool
    termination: str
    content: str = ""
    tool_calls: list[dict[str, Any]] = field(default_factory=list)
    messages: list[dict[str, Any]] = field(default_factory=list)

    def trace_value(self) -> dict[str, Any]:
        return {
            "completed": self.completed,
            "termination": self.termination,
            "content": self.content,
            "tool_calls": self.tool_calls,
        }


def tool_result_content(value: dict[str, Any]) -> str:
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"))


__all__ = [
    "FunctionTool",
    "ToolArguments",
    "ToolLoopResult",
    "tool_result_content",
]

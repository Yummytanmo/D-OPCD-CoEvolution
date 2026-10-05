"""Failure-only retries for one-shot model-authored management operations."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class ManagementCallResult:
    raw_response: str
    parsed_response: dict[str, Any]
    normalized_output: Any
    attempts: list[dict[str, Any]]


class ManagementCallError(RuntimeError):
    def __init__(
        self,
        cause: Exception,
        *,
        attempts: list[dict[str, Any]],
        raw_response: str,
        parsed_response: dict[str, Any] | None,
    ) -> None:
        super().__init__(str(cause))
        self.cause = cause
        self.attempts = attempts
        self.raw_response = raw_response
        self.parsed_response = parsed_response


def _completion_metadata(think: Callable[[str], str]) -> dict[str, Any]:
    owner = getattr(think, "__self__", None)
    getter = getattr(owner, "get_last_completion_metadata", None)
    if not callable(getter):
        return {}
    value = getter()
    return dict(value) if isinstance(value, dict) else {}


def run_management_call(
    *,
    think: Callable[[str], str],
    prompt: str,
    max_attempts: int,
    stage_name: str,
    parse: Callable[[str], dict[str, Any] | None],
    execute: Callable[[dict[str, Any]], Any],
    retry_prompt: Callable[[str, Exception, int], str] | None = None,
) -> ManagementCallResult:
    """Run one normal call, retrying only when its result cannot be executed."""
    if int(max_attempts) <= 0:
        raise ValueError("management max_attempts must be positive")

    attempt_records: list[dict[str, Any]] = []
    last_response = ""
    last_value: dict[str, Any] | None = None
    next_prompt = prompt
    for attempt in range(1, int(max_attempts) + 1):
        attempt_prompt = next_prompt
        last_response = ""
        last_value = None
        try:
            last_response = str(think(attempt_prompt) or "").strip()
            last_value = parse(last_response)
            if last_value is None:
                raise ValueError(f"{stage_name} response is not a JSON object")
            normalized = execute(last_value)
        except Exception as error:
            attempt_records.append(
                {
                    "attempt": attempt,
                    "status": "failed",
                    "prompt": attempt_prompt,
                    "raw_response": last_response,
                    "parsed_response": last_value,
                    "completion_metadata": _completion_metadata(think),
                    "error": f"{type(error).__name__}: {error}",
                }
            )
            if attempt == int(max_attempts):
                raise ManagementCallError(
                    error,
                    attempts=attempt_records,
                    raw_response=last_response,
                    parsed_response=last_value,
                ) from error
            next_prompt = (
                str(retry_prompt(prompt, error, attempt))
                if retry_prompt is not None
                else prompt
            )
            continue

        attempt_records.append(
            {
                "attempt": attempt,
                "status": "completed",
                "prompt": attempt_prompt,
                "raw_response": last_response,
                "parsed_response": last_value,
                "completion_metadata": _completion_metadata(think),
            }
        )
        return ManagementCallResult(
            raw_response=last_response,
            parsed_response=last_value,
            normalized_output=normalized,
            attempts=attempt_records,
        )

    raise AssertionError("unreachable management retry state")


__all__ = [
    "ManagementCallError",
    "ManagementCallResult",
    "run_management_call",
]

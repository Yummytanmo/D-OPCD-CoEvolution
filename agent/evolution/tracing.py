"""Auditable prompt/response traces for model invocations that evolve harness state."""

from __future__ import annotations

import uuid
from typing import Any

from agent.evolution.storage import StateStore, utc_now


class EvolutionTraceStore:
    """Persist each evolution invocation and its parsed management response."""

    COLLECTION = "evolution_traces"

    def __init__(self, state: StateStore) -> None:
        self.state = state

    def start(
        self,
        *,
        stage: str,
        prompt: str,
        task_id: str | None = None,
        batch_id: int | None = None,
    ) -> str:
        trace_id = (
            f"{self.state.round_id}:{str(stage)}:"
            f"{task_id if task_id is not None else batch_id}:"
            f"{uuid.uuid4().hex}"
        )
        value: dict[str, Any] = {
            "schema_version": 1,
            "trace_id": trace_id,
            "stage": str(stage),
            "round_id": self.state.round_id,
            "status": "started",
            "prompt": str(prompt),
            "created_at": utc_now(),
        }
        if task_id is not None:
            value["task_id"] = str(task_id)
        if batch_id is not None:
            value["batch_id"] = int(batch_id)
        if not self.state.create_record(self.COLLECTION, trace_id, value):
            raise RuntimeError(f"duplicate evolution trace ID: {trace_id}")
        return trace_id

    def complete(
        self,
        trace_id: str,
        *,
        raw_response: str,
        parsed_response: Any,
        normalized_output: Any,
        model_attempts: list[dict[str, Any]] | None = None,
    ) -> None:
        updated = self.state.update_record(
            self.COLLECTION,
            trace_id,
            lambda value: {
                **value,
                "status": "completed",
                "raw_response": str(raw_response),
                "parsed_response": parsed_response,
                "normalized_output": normalized_output,
                **(
                    {"model_attempts": model_attempts}
                    if model_attempts is not None
                    else {}
                ),
                "completed_at": utc_now(),
            },
        )
        if updated is None:
            raise RuntimeError(f"evolution trace does not exist: {trace_id}")

    def fail(
        self,
        trace_id: str,
        error: BaseException,
        *,
        raw_response: str | None = None,
        parsed_response: Any = None,
        model_attempts: list[dict[str, Any]] | None = None,
    ) -> None:
        updated = self.state.update_record(
            self.COLLECTION,
            trace_id,
            lambda value: {
                **value,
                "status": "failed",
                "error": f"{type(error).__name__}: {error}",
                **(
                    {"raw_response": str(raw_response)}
                    if raw_response is not None
                    else {}
                ),
                **(
                    {"parsed_response": parsed_response}
                    if raw_response is not None
                    else {}
                ),
                **(
                    {"model_attempts": model_attempts}
                    if model_attempts is not None
                    else {}
                ),
                "completed_at": utc_now(),
            },
        )
        if updated is None:
            raise RuntimeError(f"evolution trace does not exist: {trace_id}")


__all__ = ["EvolutionTraceStore"]

"""One-call batch conversion from immutable Episodes to ordered Insight operations."""

from __future__ import annotations

import json
import re
from collections.abc import Callable
from typing import Any

from agent.evolution.management_call import (
    ManagementCallError,
    run_management_call,
)
from agent.evolution.memory.workspace import InsightWorkspace
from agent.evolution.tracing import EvolutionTraceStore
from agent.prompts import PromptManager


class InsightConsolidator:
    def __init__(
        self,
        think: Callable[[str], str],
        *,
        prompt_manager: PromptManager | None = None,
        trace_store: EvolutionTraceStore | None = None,
        max_attempts: int = 3,
    ) -> None:
        self.think = think
        self.prompt_manager = prompt_manager or PromptManager()
        self.trace_store = trace_store
        self.max_attempts = int(max_attempts)
        if self.max_attempts <= 0:
            raise ValueError("Insight consolidation max_attempts must be positive")

    def consolidate(
        self,
        *,
        batch_id: int,
        episodes: list[dict[str, Any]],
        current_insights: list[dict[str, Any]],
    ) -> list[dict[str, Any]]:
        """Generate, validate, and resolve one executable ordered operation chain."""
        if not episodes:
            return []
        episode_id_by_ref = {
            f"e{index}": str(item["episode_id"])
            for index, item in enumerate(episodes, 1)
        }
        insight_id_by_ref = {
            f"i{index}": str(item["insight_id"])
            for index, item in enumerate(current_insights, 1)
        }
        prompt = self.prompt_manager.insight_consolidation(
            episodes=[
                self._prompt_episode(item, episode_ref)
                for episode_ref, item in zip(episode_id_by_ref, episodes)
            ],
            current_insights=[
                self._prompt_insight(item, insight_ref)
                for insight_ref, item in zip(
                    insight_id_by_ref,
                    current_insights,
                )
            ],
        )
        trace_id = (
            self.trace_store.start(
                stage="insight_consolidation",
                prompt=prompt,
                batch_id=int(batch_id),
            )
            if self.trace_store is not None
            else None
        )
        def execute(value: dict[str, Any]) -> dict[str, Any]:
            workspace = InsightWorkspace(
                episode_id_by_ref=episode_id_by_ref,
                insight_id_by_ref=insight_id_by_ref,
            )
            workspace.apply_serialized_plan(value)
            return {"operations": workspace.operations()}

        try:
            result = run_management_call(
                think=self.think,
                prompt=prompt,
                max_attempts=self.max_attempts,
                stage_name="Insight",
                parse=self._json_object,
                execute=execute,
                retry_prompt=self._insight_retry_prompt,
            )
        except ManagementCallError as failure:
            if trace_id is not None:
                self.trace_store.fail(
                    trace_id,
                    failure.cause,
                    raw_response=failure.raw_response,
                    parsed_response=failure.parsed_response,
                    model_attempts=failure.attempts,
                )
            raise failure.cause from failure
        operations = result.normalized_output["operations"]
        if trace_id is not None:
            self.trace_store.complete(
                trace_id,
                raw_response=result.raw_response,
                parsed_response=result.parsed_response,
                normalized_output={"operations": operations},
                model_attempts=result.attempts,
            )
        return operations

    @staticmethod
    def _insight_retry_prompt(
        original_prompt: str,
        error: Exception,
        _: int,
    ) -> str:
        """Keep the full operation contract while exposing the validation failure."""
        return (
            f"{original_prompt}\n\n"
            "RETRY FEEDBACK:\n"
            f"- The previous operation plan failed validation: {error}.\n"
            "- Return a complete corrected JSON object whose operations are "
            "mutually consistent."
        )

    @staticmethod
    def retrieval_query(episode: dict[str, Any]) -> str:
        """Build a compact semantic query from task-local evidence only."""

        def clipped(value: Any, limit: int) -> str:
            normalized = " ".join(str(value or "").split())
            if len(normalized) <= limit:
                return normalized
            head = normalized[: limit - 1].rsplit(" ", 1)[0]
            return f"{head or normalized[: limit - 1]}…"

        def bullets(values: list[Any], *, limit: int = 220) -> str:
            unique = list(dict.fromkeys(clipped(item, limit) for item in values if item))
            return "\n".join(f"- {item}" for item in unique) or "- None"

        task = dict(episode.get("task") or {})
        evidence = dict(episode.get("evidence") or {})
        outcome = dict(episode.get("outcome") or {})
        attempts = list(evidence.get("attempts") or [])
        observations = [
            item.get("experience")
            for item in attempts[-3:]
            if isinstance(item, dict) and item.get("experience")
        ]
        transitions = []
        for delta in list(evidence.get("attempt_deltas") or [])[-2:]:
            if not isinstance(delta, dict):
                continue
            for label, key in (
                ("improved", "newly_passed"),
                ("regressed", "newly_failed"),
                ("still failed", "still_failed"),
            ):
                values = [str(item) for item in delta.get(key, []) if item]
                if values:
                    transitions.append(f"{label}: {', '.join(values)}")
        sections = [
            "Task:",
            clipped(task.get("original_prompt"), 520),
            "Final unresolved requirements:",
            bullets(list(outcome.get("failed") or [])),
            "Attempt observations:",
            bullets(observations, limit=300),
        ]
        if transitions:
            sections.extend(
                ["Observed check transitions:", bullets(transitions, limit=260)]
            )
        if outcome.get("feedback"):
            sections.extend(
                ["Post-task feedback:", clipped(outcome["feedback"], 320)]
            )
        return clipped("\n".join(sections), 1900)

    @staticmethod
    def _prompt_episode(
        value: dict[str, Any],
        episode_ref: str,
    ) -> dict[str, Any]:
        """Expose only semantic task evidence under a batch-local reference."""
        task = dict(value.get("task") or {})
        evidence = dict(value.get("evidence") or {})
        outcome = dict(value.get("outcome") or {})
        final_outcome = {
            key: outcome[key]
            for key in ("returned_attempt", "feedback", "reward")
            if outcome.get(key) is not None
        }
        projected = {
            "episode_ref": episode_ref,
            "original_prompt": str(task.get("original_prompt") or ""),
            "attempts": list(evidence.get("attempts") or []),
            "attempt_deltas": list(evidence.get("attempt_deltas") or []),
            "final_outcome": final_outcome,
        }
        if value.get("summary") is not None:
            projected["summary"] = value["summary"]
        return projected

    @staticmethod
    def _prompt_insight(
        value: dict[str, Any],
        insight_ref: str,
    ) -> dict[str, Any]:
        return {
            "insight_ref": insight_ref,
            "text": str(value.get("text") or ""),
            "support_count": len(value.get("support_episode_ids") or []),
            "contradiction_count": len(value.get("contradict_episode_ids") or []),
        }

    @staticmethod
    def _json_object(response: str) -> dict[str, Any] | None:
        try:
            value = json.loads(response)
        except json.JSONDecodeError:
            match = re.search(r"\{.*\}", response, re.DOTALL)
            if match is None:
                return None
            try:
                value = json.loads(match.group())
            except json.JSONDecodeError:
                return None
        return value if isinstance(value, dict) else None


__all__ = ["InsightConsolidator"]

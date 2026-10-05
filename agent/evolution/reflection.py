"""Generate one grounded, task-local episode summary after task completion."""

from __future__ import annotations

import json
import re
from collections.abc import Callable
from typing import Any

from agent.evolution.models import EpisodeSummary, LearningFeedback, RunResult
from agent.evolution.soft_constraints import warn_soft_limit
from agent.evolution.tracing import EvolutionTraceStore
from agent.prompts import PromptManager


_EFFECTS = {"helpful", "harmful", "no_clear_effect"}


class EpisodeGenerator:
    """Make exactly one model call and validate only the generated summary."""

    def __init__(
        self,
        think: Callable[[str], str],
        *,
        prompt_manager: PromptManager | None = None,
        trace_store: EvolutionTraceStore | None = None,
        max_observations: int = 5,
    ) -> None:
        self.think = think
        self.prompt_manager = prompt_manager or PromptManager()
        self.trace_store = trace_store
        self.max_observations = int(max_observations)
        if self.max_observations <= 0:
            raise ValueError("max_observations must be positive")

    def generate(
        self,
        result: RunResult,
        feedback: LearningFeedback | str,
        *,
        batch_id: int | None = None,
    ) -> EpisodeSummary:
        selected_feedback = LearningFeedback.coerce(feedback)
        attempts = result.episode_attempts()
        deltas = result.attempt_deltas()
        prompt = self.prompt_manager.episode_summary(
            original_prompt=result.original_prompt,
            returned_attempt=result.returned_attempt,
            attempts=attempts,
            attempt_deltas=deltas,
            feedback_text=selected_feedback.text,
            reward=selected_feedback.reward,
        )
        trace_id = (
            self.trace_store.start(
                stage="episode_summary",
                prompt=prompt,
                task_id=result.task_id,
                batch_id=batch_id,
            )
            if self.trace_store is not None
            else None
        )
        try:
            response = self.think(prompt).strip()
        except BaseException as error:
            if trace_id is not None:
                self.trace_store.fail(trace_id, error)
            raise
        value = self._json_object(response)
        summary = (
            self._validate(value, result) if value is not None else self._fallback(result)
        )
        if trace_id is not None:
            self.trace_store.complete(
                trace_id,
                raw_response=response,
                parsed_response=value,
                normalized_output=summary.to_dict(),
            )
        return summary

    @staticmethod
    def attempt_deltas(result: RunResult) -> list[dict[str, Any]]:
        """Compatibility wrapper for callers that inspect deterministic deltas."""
        return result.attempt_deltas()

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

    def _validate(self, value: dict[str, Any], result: RunResult) -> EpisodeSummary:
        valid_attempts = {item.iteration: item for item in result.attempts}
        observations = []
        for raw in value.get("observations") or []:
            if not isinstance(raw, dict):
                continue
            effect = str(raw.get("effect") or "no_clear_effect").casefold()
            effect = effect if effect in _EFFECTS else "no_clear_effect"
            change = str(raw.get("change") or "").strip()
            observed = str(raw.get("observed_result") or "").strip()
            evidence = []
            for attempt_id in raw.get("evidence_attempts") or []:
                try:
                    parsed = int(attempt_id)
                except (TypeError, ValueError):
                    continue
                if parsed in valid_attempts and parsed not in evidence:
                    evidence.append(parsed)
            if not change or not observed or not evidence:
                continue
            observations.append(
                {
                    "effect": effect,
                    "change": change,
                    "observed_result": observed,
                    "evidence_attempts": evidence,
                }
            )
        warn_soft_limit(
            f"Episode summary observations (task={result.task_id})",
            actual=len(observations),
            preferred_max=self.max_observations,
            unit="observations",
        )

        task_summary = str(value.get("task_summary") or "").strip()
        trajectory_summary = str(value.get("trajectory_summary") or "").strip()
        fallback = self._fallback(result)
        unresolved = list(
            dict.fromkeys(
                str(item).strip()
                for item in (value.get("unresolved") or [])
                if str(item).strip()
            )
        )
        returned = result.attempt(result.returned_attempt)
        if returned is not None:
            unresolved = list(dict.fromkeys([*unresolved, *returned.failed]))
        warn_soft_limit(
            f"Episode summary unresolved items (task={result.task_id})",
            actual=len(unresolved),
            preferred_max=8,
            unit="items",
        )
        return EpisodeSummary(
            task_summary=task_summary or fallback.task_summary,
            trajectory_summary=trajectory_summary or fallback.trajectory_summary,
            observations=observations,
            unresolved=unresolved,
        )

    @staticmethod
    def _fallback(result: RunResult) -> EpisodeSummary:
        ordered = sorted(result.attempts, key=lambda item: item.iteration)
        trajectory = []
        for item in ordered:
            trajectory.append(
                "Attempt {iteration} passed {passed} and failed {failed}.".format(
                    iteration=item.iteration,
                    passed=", ".join(item.passed) or "no recorded checks",
                    failed=", ".join(item.failed) or "no recorded checks",
                )
            )
        returned = result.attempt(result.returned_attempt)
        unresolved = list(returned.failed) if returned is not None else []
        if result.returned_attempt is not None:
            trajectory.append(f"Attempt {result.returned_attempt} was returned.")
        return EpisodeSummary(
            task_summary=result.original_prompt.strip(),
            trajectory_summary=" ".join(trajectory) or "No attempt checks were recorded.",
            observations=[],
            unresolved=unresolved,
        )


# Transitional import name for external callers. It now produces episodes and has no
# signature catalog or cross-task lesson behavior.
Reflector = EpisodeGenerator


__all__ = ["EpisodeGenerator", "Reflector"]

"""Immutable per-task episodes and deterministic execution envelopes."""

from __future__ import annotations

import hashlib
from typing import Any

from agent.evolution.models import EpisodeSummary, LearningFeedback, RunResult
from agent.evolution.storage import StateStore, utc_now


class EpisodeStore:
    def __init__(self, state: StateStore) -> None:
        self.state = state

    def get(self, task_id: str) -> dict[str, Any] | None:
        return self.state.read_record("episodes", self.state.episode_key(task_id))

    def get_by_episode_id(self, episode_id: str) -> dict[str, Any] | None:
        return next(
            (
                item
                for item in self.state.records("episodes")
                if item.get("episode_id") == episode_id
            ),
            None,
        )

    def for_batch(self, batch_id: int) -> list[dict[str, Any]]:
        return sorted(
            (
                item
                for item in self.state.records("episodes")
                if int(item.get("batch_id", -1)) == int(batch_id)
            ),
            key=lambda item: str(item.get("task_id") or ""),
        )

    def _episode_id(self, task_id: str) -> str:
        digest = hashlib.sha1(
            f"{self.state.run_id}|{self.state.round_id}|{task_id}".encode("utf-8")
        ).hexdigest()[:16]
        return f"ep_{digest}"

    def create(
        self,
        result: RunResult,
        feedback: LearningFeedback | str,
        summary: EpisodeSummary | None,
        *,
        batch_id: int,
    ) -> bool:
        """Create one immutable episode; an existing task is never regenerated."""
        selected_feedback = LearningFeedback.coerce(feedback)
        returned = result.attempt(result.returned_attempt)
        selected_skills = []
        for item in result.selected_skills:
            selected_skills.append(
                {
                    "skill_id": str(item["skill_id"]),
                    "version": int(item.get("version", 0)),
                    "execution_stage": "before_first_generation",
                }
            )
        now = utc_now()
        episode = {
            "schema_version": 1,
            "episode_id": self._episode_id(result.task_id),
            "task_id": result.task_id,
            "batch_id": int(batch_id),
            "task": {
                "original_prompt": result.original_prompt,
                "generator_model": result.generator_model,
                "round_id": result.round_id,
            },
            "evidence": {
                "attempts": result.episode_attempts(),
                "attempt_deltas": result.attempt_deltas(),
            },
            "memory_context": {
                "selected_skills": selected_skills,
                "initial_insight_ids": list(result.initial_insight_ids),
                "refinement_insight_ids": list(result.refinement_insight_ids),
            },
            "outcome": {
                **(
                    {"feedback": selected_feedback.text}
                    if selected_feedback.text is not None
                    else {}
                ),
                **(
                    {"reward": selected_feedback.reward}
                    if selected_feedback.reward is not None
                    else {}
                ),
                "returned_attempt": result.returned_attempt,
                "passed": list(returned.passed) if returned is not None else [],
                "failed": list(returned.failed) if returned is not None else [],
                "first_attempt_check_score": result.first_attempt_score(),
            },
            "summary": summary.to_dict() if summary is not None else None,
            "attempt_refs": [
                item.iteration
                for item in sorted(result.attempts, key=lambda item: item.iteration)
            ],
            "created_at": now,
        }
        return self.state.create_record(
            "episodes",
            self.state.episode_key(result.task_id),
            episode,
        )


__all__ = ["EpisodeStore"]

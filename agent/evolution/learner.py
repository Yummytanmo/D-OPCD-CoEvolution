"""Coordinate immutable Episode creation and serial batch learning commits."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable

from agent.evolution.memory.consolidator import InsightConsolidator
from agent.evolution.memory.episodes import EpisodeStore
from agent.evolution.memory.insights import InsightStore
from agent.evolution.models import LearningFeedback, RunResult
from agent.evolution.reflection import EpisodeGenerator
from agent.evolution.skills.evolver import SkillEvolver
from agent.evolution.skills.registry import SkillRegistry
from agent.evolution.storage import StateStore, utc_now


@dataclass
class EpisodeReport:
    task_id: str
    episode_id: str
    batch_id: int
    duplicate: bool = False


@dataclass
class BatchUpdateReport:
    batch_id: int
    duplicate: bool = False
    episode_ids: list[str] = field(default_factory=list)
    accepted_operations: list[dict] = field(default_factory=list)
    rejected_operations: list[dict] = field(default_factory=list)
    touched_insights: list[str] = field(default_factory=list)
    skill_changes: list[dict] = field(default_factory=list)


class HarnessLearner:
    """Keep task-parallel Episode work separate from the serial batch barrier."""

    def __init__(
        self,
        *,
        state: StateStore,
        episodes: EpisodeStore,
        insights: InsightStore,
        registry: SkillRegistry,
        consolidator: InsightConsolidator,
        evolver: SkillEvolver,
    ) -> None:
        self.state = state
        self.episodes = episodes
        self.insights = insights
        self.registry = registry
        self.consolidator = consolidator
        self.evolver = evolver

    def create_episode(
        self,
        result: RunResult,
        feedback: LearningFeedback | str,
        *,
        batch_id: int,
        generator: EpisodeGenerator | None = None,
        generate_summary: bool = False,
    ) -> EpisodeReport:
        """Persist once, optionally generating an LLM summary before the write."""
        if result.round_id != self.state.round_id:
            raise ValueError("run result round does not match the active harness state")
        existing = self.episodes.get(result.task_id)
        if existing is not None:
            if int(existing.get("batch_id", -1)) != int(batch_id):
                raise ValueError("task already belongs to a different evolution batch")
            return EpisodeReport(
                task_id=result.task_id,
                episode_id=str(existing["episode_id"]),
                batch_id=int(batch_id),
                duplicate=True,
            )

        if generate_summary and generator is None:
            raise ValueError("an EpisodeGenerator is required when summary is enabled")
        summary = (
            generator.generate(result, feedback, batch_id=int(batch_id))
            if generate_summary
            else None
        )
        created = self.episodes.create(
            result,
            feedback,
            summary,
            batch_id=int(batch_id),
        )
        episode = self.episodes.get(result.task_id)
        if episode is None:
            raise RuntimeError("episode was not persisted")
        return EpisodeReport(
            task_id=result.task_id,
            episode_id=str(episode["episode_id"]),
            batch_id=int(batch_id),
            duplicate=not created,
        )

    def commit_batch(
        self,
        *,
        batch_id: int,
        task_ids: Iterable[str] | None = None,
    ) -> BatchUpdateReport:
        """Consolidate once after the batch barrier, then update utility and Skills."""
        batch_id = int(batch_id)
        batch_key = f"{self.state.round_id}:{batch_id}"
        existing = self.state.read_record("batch_commits", batch_key)
        if existing is not None:
            return self._report(existing, duplicate=True)

        episodes = self.episodes.for_batch(batch_id)
        if task_ids is not None:
            allowed = {str(item) for item in task_ids}
            episodes = [item for item in episodes if str(item["task_id"]) in allowed]
        episode_ids = [str(item["episode_id"]) for item in episodes]
        # Management sees only the union of per-Episode semantic neighbours.  Unlike
        # inference retrieval, this includes immature hypotheses so a later Episode
        # can support, revise, contradict, merge, or archive them.
        frozen_insights = self.insights.consolidation_working_set(
            self.consolidator.retrieval_query(episode) for episode in episodes
        )
        operations = self.consolidator.consolidate(
            batch_id=batch_id,
            episodes=episodes,
            current_insights=frozen_insights,
        )
        insight_audit = self.insights.apply_operations(
            batch_id=batch_id,
            episode_ids=episode_ids,
            operations=operations,
        )

        # Utility is append-only and idempotent. It is updated only after all Insight
        # operations for the batch have committed.
        for episode in episodes:
            self.insights.record_episode_usage(episode)
            outcome = dict(episode.get("outcome") or {})
            score = outcome.get("first_attempt_check_score")
            reward = outcome.get("reward")
            self.registry.record_usage(
                task_id=str(episode["task_id"]),
                selections=list(
                    (episode.get("memory_context") or {}).get("selected_skills", [])
                ),
                first_attempt_score=(
                    float(score)
                    if isinstance(score, (int, float)) and not isinstance(score, bool)
                    else None
                ),
                final_reward=(
                    float(reward)
                    if isinstance(reward, (int, float))
                    and not isinstance(reward, bool)
                    else None
                ),
            )

        skill_changes = self.evolver.process_batch(
            batch_id=batch_id,
            insights=self.insights,
            registry=self.registry,
        )
        value = {
            "batch_id": batch_id,
            "episode_ids": episode_ids,
            "accepted_operations": list(
                insight_audit.get("accepted_operations", [])
            ),
            "rejected_operations": list(
                insight_audit.get("rejected_operations", [])
            ),
            "touched_insights": list(
                insight_audit.get("touched_insight_ids", [])
            ),
            "skill_changes": skill_changes,
            "created_at": utc_now(),
        }
        self.state.create_record("batch_commits", batch_key, value)
        return self._report(value)

    @staticmethod
    def _report(value: dict, *, duplicate: bool = False) -> BatchUpdateReport:
        return BatchUpdateReport(
            batch_id=int(value["batch_id"]),
            duplicate=duplicate,
            episode_ids=[str(item) for item in value.get("episode_ids", [])],
            accepted_operations=list(value.get("accepted_operations", [])),
            rejected_operations=list(value.get("rejected_operations", [])),
            touched_insights=[str(item) for item in value.get("touched_insights", [])],
            skill_changes=list(
                value.get("skill_changes", value.get("created_candidates", []))
            ),
        )


__all__ = ["BatchUpdateReport", "EpisodeReport", "HarnessLearner"]

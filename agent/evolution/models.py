"""Serializable values shared by task execution and batch evolution."""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any


@dataclass
class AttemptRecord:
    iteration: int
    prompt: str
    passed: list[str]
    failed: list[str]
    seed: int | None = None
    experience: str = ""
    image_path: str | None = None
    image_bytes: bytes | None = field(default=None, repr=False)

    def to_dict(self) -> dict[str, Any]:
        return {
            "iteration": self.iteration,
            "prompt": self.prompt,
            "passed": list(self.passed),
            "failed": list(self.failed),
            "seed": self.seed,
            "experience": self.experience,
            "image_path": self.image_path,
        }

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> "AttemptRecord":
        return cls(
            iteration=int(value["iteration"]),
            prompt=str(value["prompt"]),
            passed=[str(item) for item in value.get("passed", [])],
            failed=[str(item) for item in value.get("failed", [])],
            seed=value.get("seed"),
            experience=str(value.get("experience") or ""),
            image_path=value.get("image_path"),
        )


@dataclass
class PlanResult:
    prompt: str
    # Retain the singular projection for callers written before multi-Skill routing.
    selected_skill: dict[str, Any] | None = None
    retrieved_insights: list[dict[str, Any]] = field(default_factory=list)
    selected_skills: list[dict[str, Any]] = field(default_factory=list)

    def __post_init__(self) -> None:
        if self.selected_skills:
            self.selected_skills = list(self.selected_skills)
            self.selected_skill = self.selected_skills[0]
        elif self.selected_skill is not None:
            self.selected_skills = [self.selected_skill]


@dataclass
class FirstAttemptReplay:
    """Immutable first-attempt prefix loaded from a reference evaluation run."""

    task_id: str
    original_prompt: str
    selected_skills: list[dict[str, Any]]
    attempt: AttemptRecord
    source: str
    prefix_generator_calls: int = 1
    prefix_mllm_calls: int = 0


@dataclass
class RunResult:
    task_id: str
    original_prompt: str
    final_prompt: str
    final_image_bytes: bytes | None
    attempts: list[AttemptRecord]
    selected_skills: list[dict[str, Any]] = field(default_factory=list)
    initial_insight_ids: list[str] = field(default_factory=list)
    refinement_insight_ids: list[str] = field(default_factory=list)
    # Kept as a serialized compatibility projection. New code writes the two
    # stage-specific lists above and derives this union.
    retrieved_insight_ids: list[str] = field(default_factory=list)
    returned_attempt: int | None = None
    final_image_path: str | None = None
    generator_calls: int = 0
    mllm_calls: int = 0
    generator_model: str = "unknown"
    round_id: int = 0
    first_trajectory_source: str | None = None
    replayed_attempts: list[int] = field(default_factory=list)

    def __post_init__(self) -> None:
        if (
            self.retrieved_insight_ids
            and not self.initial_insight_ids
            and not self.refinement_insight_ids
        ):
            self.initial_insight_ids = list(self.retrieved_insight_ids)
        self.initial_insight_ids = list(
            dict.fromkeys(str(item) for item in self.initial_insight_ids)
        )
        self.refinement_insight_ids = list(
            dict.fromkeys(str(item) for item in self.refinement_insight_ids)
        )
        self.retrieved_insight_ids = list(
            dict.fromkeys(
                [
                    *self.initial_insight_ids,
                    *self.refinement_insight_ids,
                    *(str(item) for item in self.retrieved_insight_ids),
                ]
            )
        )
        if self.returned_attempt is None and self.attempts:
            matching = [
                item.iteration
                for item in self.attempts
                if item.prompt == self.final_prompt
            ]
            self.returned_attempt = (
                max(matching)
                if matching
                else max(item.iteration for item in self.attempts)
            )

    def attempt(self, iteration: int | None) -> AttemptRecord | None:
        if iteration is None:
            return None
        return next(
            (item for item in self.attempts if item.iteration == int(iteration)),
            None,
        )

    def first_attempt_score(self) -> float | None:
        """Return task-local first-generation check quality when checks exist."""
        first = min(self.attempts, key=lambda item: item.iteration, default=None)
        if first is None:
            return None
        total = len(first.passed) + len(first.failed)
        return len(first.passed) / total if total else None

    def episode_attempts(self) -> list[dict[str, Any]]:
        """Return the compact, seed-independent attempt evidence stored in Episodes."""
        return [
            {
                "iteration": item.iteration,
                "prompt": item.prompt,
                "passed": list(item.passed),
                "failed": list(item.failed),
                "experience": item.experience,
            }
            for item in sorted(self.attempts, key=lambda item: item.iteration)
        ]

    def attempt_deltas(self) -> list[dict[str, Any]]:
        """Calculate deterministic check transitions between adjacent attempts."""
        ordered = sorted(self.attempts, key=lambda item: item.iteration)
        deltas = []
        for left, right in zip(ordered, ordered[1:]):
            left_passed = set(left.passed)
            right_passed = set(right.passed)
            left_failed = set(left.failed)
            right_failed = set(right.failed)
            deltas.append(
                {
                    "from_attempt": left.iteration,
                    "to_attempt": right.iteration,
                    "newly_passed": sorted(left_failed & right_passed),
                    "newly_failed": sorted(left_passed & right_failed),
                    "still_failed": sorted(left_failed & right_failed),
                }
            )
        return deltas

    def to_dict(self) -> dict[str, Any]:
        return {
            "task_id": self.task_id,
            "original_prompt": self.original_prompt,
            "final_prompt": self.final_prompt,
            "final_image_path": self.final_image_path,
            "attempts": [attempt.to_dict() for attempt in self.attempts],
            "selected_skills": list(self.selected_skills),
            "initial_insight_ids": list(self.initial_insight_ids),
            "refinement_insight_ids": list(self.refinement_insight_ids),
            "retrieved_insight_ids": list(self.retrieved_insight_ids),
            "returned_attempt": self.returned_attempt,
            "generator_calls": self.generator_calls,
            "mllm_calls": self.mllm_calls,
            "generator_model": self.generator_model,
            "round_id": self.round_id,
            "first_trajectory_source": self.first_trajectory_source,
            "replayed_attempts": list(self.replayed_attempts),
        }

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> "RunResult":
        return cls(
            task_id=str(value["task_id"]),
            original_prompt=str(value["original_prompt"]),
            final_prompt=str(value["final_prompt"]),
            final_image_bytes=None,
            final_image_path=value.get("final_image_path"),
            attempts=[
                AttemptRecord.from_dict(item) for item in value.get("attempts", [])
            ],
            selected_skills=list(value.get("selected_skills", [])),
            initial_insight_ids=[
                str(item) for item in value.get("initial_insight_ids", [])
            ],
            refinement_insight_ids=[
                str(item) for item in value.get("refinement_insight_ids", [])
            ],
            retrieved_insight_ids=[
                str(item) for item in value.get("retrieved_insight_ids", [])
            ],
            returned_attempt=(
                int(value["returned_attempt"])
                if value.get("returned_attempt") is not None
                else None
            ),
            generator_calls=int(value.get("generator_calls", 0)),
            mllm_calls=int(value.get("mllm_calls", 0)),
            generator_model=str(
                value.get("generator_model")
                or value.get("generator_version")
                or "unknown"
            ),
            round_id=int(value.get("round_id", 0)),
            first_trajectory_source=value.get("first_trajectory_source"),
            replayed_attempts=[
                int(item) for item in value.get("replayed_attempts", [])
            ],
        )


@dataclass(frozen=True)
class LearningFeedback:
    """The post-task signal selected for learning: text, reward, or both."""

    text: str | None = None
    reward: float | None = None

    def __post_init__(self) -> None:
        text = str(self.text).strip() if self.text is not None else None
        text = text or None
        reward = float(self.reward) if self.reward is not None else None
        if reward is not None and (
            not math.isfinite(reward) or not 0.0 <= reward <= 1.0
        ):
            raise ValueError("feedback reward must be a finite number between 0 and 1")
        if text is None and reward is None:
            raise ValueError("learning feedback requires text, reward, or both")
        object.__setattr__(self, "text", text)
        object.__setattr__(self, "reward", reward)

    @classmethod
    def coerce(cls, value: "LearningFeedback | str") -> "LearningFeedback":
        return value if isinstance(value, cls) else cls(text=value)

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> "LearningFeedback":
        return cls(text=value.get("text"), reward=value.get("reward"))

    def to_dict(self) -> dict[str, Any]:
        value: dict[str, Any] = {}
        if self.text is not None:
            value["text"] = self.text
        if self.reward is not None:
            value["reward"] = self.reward
        return value


@dataclass
class EpisodeSummary:
    """The only model-generated portion of an immutable episode."""

    task_summary: str = ""
    trajectory_summary: str = ""
    observations: list[dict[str, Any]] = field(default_factory=list)
    unresolved: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "task_summary": self.task_summary,
            "trajectory_summary": self.trajectory_summary,
            "observations": list(self.observations),
            "unresolved": list(self.unresolved),
        }

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> "EpisodeSummary":
        return cls(
            task_summary=str(value.get("task_summary") or ""),
            trajectory_summary=str(value.get("trajectory_summary") or ""),
            observations=[
                dict(item)
                for item in value.get("observations", [])
                if isinstance(item, dict)
            ],
            unresolved=[str(item) for item in value.get("unresolved", [])],
        )


__all__ = [
    "AttemptRecord",
    "EpisodeSummary",
    "LearningFeedback",
    "PlanResult",
    "RunResult",
]

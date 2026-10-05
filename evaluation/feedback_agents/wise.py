"""Official WISE_Verified binary feedback for one submitted image."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from evaluation.evaluators.wise_verified import WiseVerifiedRuntime
from evaluation.feedback_agents.base import FeedbackAgent, FeedbackVerbalizer


class WiseFeedbackAgent(FeedbackAgent):
    benchmark = "wise"

    def __init__(
        self,
        *,
        evaluator_runtime: WiseVerifiedRuntime | None = None,
        verbalizer: FeedbackVerbalizer | None = None,
    ) -> None:
        super().__init__(verbalizer)
        self._runtime = evaluator_runtime or WiseVerifiedRuntime()

    def warmup(self) -> None:
        self._runtime.check_ready()

    def evaluate_image(
        self,
        *,
        task_prompt: str,
        image_path: Path,
        metadata: dict[str, Any] | None,
    ) -> dict[str, Any]:
        if metadata is None:
            raise ValueError("WISE feedback requires private benchmark metadata")
        explanation = metadata.get("explanation") or metadata.get("Explanation")
        if not isinstance(explanation, str) or not explanation.strip():
            raise ValueError("WISE metadata requires a non-empty explanation")
        return self._runtime.evaluate_image(
            image_path,
            prompt=task_prompt,
            explanation=explanation.strip(),
        )

    def evaluator_reward(self, private_evidence: dict[str, Any]) -> float:
        return float(private_evidence["score"])

    def fallback_feedback(
        self,
        *,
        task_prompt: str,
        private_evidence: dict[str, Any],
        metadata: dict[str, Any] | None,
    ) -> str:
        if bool(private_evidence.get("correct")):
            return "The image clearly realizes the intended knowledge-based meaning and remains visually usable."
        return "The image does not clearly realize the intended knowledge-based meaning or is too ambiguous to verify."


__all__ = ["WiseFeedbackAgent"]

"""GenEval2 Soft-TIFA feedback for one submitted image."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from evaluation.evaluator_pool import EvaluatorPool
from evaluation.evaluators.geneval2_runtime import GenEval2Runtime
from evaluation.feedback_agents.base import FeedbackAgent, FeedbackVerbalizer


class GenEval2FeedbackAgent(FeedbackAgent):
    benchmark = "geneval2"

    def __init__(
        self,
        *,
        model_path: str | Path | None = None,
        evaluator_runtime: GenEval2Runtime | None = None,
        evaluator_runtimes: list[GenEval2Runtime]
        | tuple[GenEval2Runtime, ...]
        | None = None,
        verbalizer: FeedbackVerbalizer | None = None,
    ) -> None:
        super().__init__(verbalizer)
        if evaluator_runtime is not None and evaluator_runtimes is not None:
            raise ValueError("Pass evaluator_runtime or evaluator_runtimes, not both")
        if evaluator_runtime is None and evaluator_runtimes is None and model_path is None:
            raise ValueError("GenEval2 requires model_path or an evaluator runtime")
        self.model_path = str(model_path) if model_path is not None else None
        self._runtime = evaluator_runtime
        self._runtime_pool = (
            EvaluatorPool(self.benchmark, evaluator_runtimes)
            if evaluator_runtimes is not None
            else None
        )

    @property
    def evaluator_concurrency(self) -> int:
        return self._runtime_pool.concurrency if self._runtime_pool is not None else 1

    def _get_runtime(self) -> GenEval2Runtime:
        if self._runtime is None:
            if self.model_path is None:
                raise RuntimeError("GenEval2 model path is unavailable")
            self._runtime = GenEval2Runtime(model_path=self.model_path)
        return self._runtime

    def warmup(self) -> None:
        if self._runtime_pool is None:
            self._get_runtime().load()

    def evaluate_image(
        self,
        *,
        task_prompt: str,
        image_path: Path,
        metadata: dict[str, Any] | None,
    ) -> dict[str, Any]:
        if metadata is None:
            raise ValueError("GenEval2 feedback requires benchmark metadata")
        if self._runtime_pool is not None:
            return self._runtime_pool.run(
                lambda runtime: runtime.evaluate_image(image_path, metadata)
            )
        return self._get_runtime().evaluate_image(image_path, metadata)

    def evaluator_reward(self, private_evidence: dict[str, Any]) -> float:
        return float(private_evidence["soft_tifa_gm"])

    def fallback_feedback(
        self,
        *,
        task_prompt: str,
        private_evidence: dict[str, Any],
        metadata: dict[str, Any] | None,
    ) -> str:
        return (
            "One or more requested objects, counts, attributes, actions, or spatial "
            "relationships are not clearly present in the image."
        )


__all__ = ["GenEval2FeedbackAgent"]

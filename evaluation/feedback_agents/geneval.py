"""GenEval-grounded feedback for one submitted image."""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any

from evaluation.assets import default_model_root
from evaluation.evaluators.geneval_runtime import GenEvalRuntime
from evaluation.evaluator_pool import EvaluatorPool
from evaluation.feedback_agents.base import FeedbackAgent, FeedbackVerbalizer


class GenEvalFeedbackAgent(FeedbackAgent):
    benchmark = "geneval"

    def __init__(
        self,
        *,
        model_path: str | Path | None = None,
        open_clip_cache_dir: str | Path | None = None,
        model_config: str | Path | None = None,
        evaluator_runtime: GenEvalRuntime | None = None,
        evaluator_runtimes: list[GenEvalRuntime] | tuple[GenEvalRuntime, ...] | None = None,
        verbalizer: FeedbackVerbalizer | None = None,
    ) -> None:
        super().__init__(verbalizer)
        if evaluator_runtime is not None and evaluator_runtimes is not None:
            raise ValueError("Pass evaluator_runtime or evaluator_runtimes, not both")
        model_root = default_model_root()
        self.model_path = Path(model_path or model_root / "geneval").expanduser().resolve()
        self.open_clip_cache_dir = Path(open_clip_cache_dir or model_root / "open_clip").expanduser().resolve()
        self.model_config = model_config
        self._runtime = evaluator_runtime
        self._runtime_pool = (
            EvaluatorPool(self.benchmark, evaluator_runtimes)
            if evaluator_runtimes is not None
            else None
        )

    def warmup(self) -> None:
        if self._runtime_pool is None:
            self._get_runtime().load()

    def _get_runtime(self) -> GenEvalRuntime:
        if self._runtime is None:
            self._runtime = GenEvalRuntime(
                model_path=self.model_path,
                open_clip_cache_dir=self.open_clip_cache_dir,
                model_config=self.model_config,
            )
        return self._runtime

    def evaluate_image(
        self,
        *,
        task_prompt: str,
        image_path: Path,
        metadata: dict[str, Any] | None,
    ) -> dict[str, Any]:
        if metadata is None:
            raise ValueError("GenEval feedback requires benchmark metadata")
        item_metadata = dict(metadata)
        item_metadata.setdefault("prompt", task_prompt)
        if self._runtime_pool is not None:
            row = self._runtime_pool.run(
                lambda runtime: runtime.evaluate_image(image_path, item_metadata)
            )
        else:
            row = self._get_runtime().evaluate_image(image_path, item_metadata)
        return {
            "correct": bool(row.get("correct")),
            "reason": str(row.get("reason") or ""),
            "tag": row.get("tag") or item_metadata.get("tag"),
            "requirements": {
                "include": item_metadata.get("include", []),
                "exclude": item_metadata.get("exclude", []),
            },
            "object_details": row.get("details"),
        }

    def fallback_feedback(
        self,
        *,
        task_prompt: str,
        private_evidence: dict[str, Any],
        metadata: dict[str, Any] | None,
    ) -> str:
        if bool(private_evidence.get("correct")):
            return "The image clearly presents all requested subjects, counts, attributes, and spatial relationships."

        reason = str(private_evidence.get("reason") or "")
        requirements = private_evidence.get("requirements") or {}
        includes = list(requirements.get("include") or [])
        messages: list[str] = []

        for requirement in includes:
            classname = str(requirement.get("class") or "requested object")
            count = int(requirement.get("count") or 1)
            color = requirement.get("color")
            count_match = re.search(
                rf"expected {re.escape(classname)}>={count}, found (\d+)",
                reason,
            )
            if count_match:
                observed = int(count_match.group(1))
                messages.append(
                    f"Only {observed} clearly visible instance(s) of {classname} appear in the image, while the task requires {count}."
                )
                continue
            if color and f"expected {color} {classname}>=" in reason:
                messages.append(
                    f"The {classname} in the image does not clearly have the requested {color} color."
                )
            if "position" in requirement and f"expected {classname} " in reason:
                relation, target_index = requirement["position"]
                target = (
                    includes[int(target_index)].get("class", "other object")
                    if int(target_index) < len(includes)
                    else "other object"
                )
                messages.append(
                    f"The image does not clearly place the {classname} {relation} the {target} as requested."
                )

        for requirement in list(requirements.get("exclude") or []):
            classname = str(requirement.get("class") or "excluded object")
            if f"expected {classname}<" in reason:
                messages.append(f"The image contains {classname}, which the task excludes.")

        if not messages:
            if "no target for" in reason:
                messages.append("A subject needed to form the requested spatial relationship is not clearly present in the image.")
            else:
                messages.append("The image does not fully present all visual requirements of the current task.")
        return " ".join(dict.fromkeys(messages))

    def evaluator_reward(self, private_evidence: dict[str, Any]) -> float:
        return float(bool(private_evidence.get("correct")))

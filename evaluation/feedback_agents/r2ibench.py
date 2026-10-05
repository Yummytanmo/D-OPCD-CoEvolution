"""Task-final feedback grounded in private R2I-Bench criteria."""

from __future__ import annotations

import re
from typing import Any

from evaluation.evaluators.r2ibench import R2IBenchRuntime
from evaluation.feedback_agents.base import FeedbackAgent


def _safe_requirement(value: str) -> str:
    text = " ".join(value.split()).strip().rstrip("?")
    replacements = {
        r"\bocr\b": "text reading",
        r"\bpaddleocr\b": "text reading",
        r"\bpickscore\b": "visual review",
        r"\bgeneval\b": "visual review",
        r"\bmask2former\b": "visual review",
        r"\bevaluation model\b": "visual reviewer",
        r"\bscore\b": "rating",
        r"\bevaluator\b": "reviewer",
        r"\bconfidence\b": "certainty",
        r"\bthreshold\b": "cutoff",
        r"\blevenshtein\b": "edit distance",
        r"\brecognized text\b": "visible writing",
        r"\bdetected object\b": "visible object",
        r"\bshould\b": "must",
        r"\brecommend\b": "favor",
        r"\bnext attempt\b": "current image",
        r"\bnext time\b": "current image",
        r"\bfuture task\b": "current task",
        r"\btry to\b": "aim to",
        r"\bimprove by\b": "strengthen through",
        r"\bcould be improved\b": "is weak",
    }
    for pattern, replacement in replacements.items():
        text = re.sub(pattern, replacement, text, flags=re.IGNORECASE)
    return text


class R2IBenchFeedbackAgent(FeedbackAgent):
    benchmark = "r2ibench"

    def __init__(self, *, evaluator_runtime: R2IBenchRuntime | None = None, verbalizer=None):
        super().__init__(verbalizer)
        self._runtime = evaluator_runtime or R2IBenchRuntime()

    def warmup(self) -> None:
        self._runtime.check_ready()

    def evaluate_image(self, *, task_prompt, image_path, metadata):
        checklist = (metadata or {}).get("checklist")
        if not checklist:
            raise ValueError("private official R2I checklist is required")
        result = self._runtime.evaluate_image(
            image_path, prompt=task_prompt, checklist=checklist
        )
        result["tag"] = str((metadata or {}).get("category") or "all")
        result["subcategory"] = str((metadata or {}).get("subcategory") or "unknown")
        return result

    def evaluator_reward(self, private_evidence):
        return float(private_evidence["score"])

    def fallback_feedback(self, *, task_prompt, private_evidence, metadata):
        failed = sorted(
            (
                item
                for item in private_evidence.get("checklist_results", [])
                if float(item.get("score", 0.0)) < 0.75
            ),
            key=lambda item: (-float(item.get("weight", 0.0)), float(item.get("score", 0.0))),
        )
        if not failed:
            return "The final image clearly realizes the requested concepts and relationships."
        requirements = [_safe_requirement(str(item["question"])) for item in failed[:3]]
        return ("Missing or unclear in the final image: " + "; ".join(requirements))[:800]

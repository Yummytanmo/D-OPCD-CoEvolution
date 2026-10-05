"""Shared inheritance and privacy boundary for feedback agents."""

from __future__ import annotations

import base64
import logging
import math
import re
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol

from evaluation.model_config import (
    feedback_model_settings,
    resolve_feedback_api_key,
)
from evaluation.image_transport import encode_transport_image
from evaluation.openai_compatible import create_chat_completion
from evaluation.prompt import SYSTEM_PROMPT, build_feedback_prompt


LOGGER = logging.getLogger("evaluation.feedback_service")


class FeedbackVerbalizer(Protocol):
    """Turns private evidence into a public visual assessment."""

    def generate(
        self,
        system_prompt: str,
        user_prompt: str,
        image_path: Path,
    ) -> str: ...


class OpenAIFeedbackVerbalizer:
    """OpenAI-compatible multimodal verbalizer with no evaluator logic."""

    def __init__(
        self,
        *,
        api_key: str | None = None,
        base_url: str | None = None,
        model: str | None = None,
        proxy_url: str | None = None,
        max_tokens: int = 512,
        compress_images: bool | None = None,
        image_max_side: int | None = None,
        image_quality: int | None = None,
        timeout: float = 600.0,
    ) -> None:
        settings = feedback_model_settings()
        resolved_key = resolve_feedback_api_key(api_key)
        if not resolved_key:
            raise ValueError("Missing feedback_mllm.api_key in evaluation/config.json")
        self.model = model or settings.model
        resolved_url = base_url or settings.base_url
        self.proxy_url = proxy_url if proxy_url is not None else settings.proxy_url
        self.api_key = resolved_key
        self.base_url = resolved_url
        self.max_tokens = max_tokens
        self.compress_images = (
            getattr(settings, "compress_images", True)
            if compress_images is None
            else bool(compress_images)
        )
        self.image_max_side = (
            settings.image_max_side if image_max_side is None else int(image_max_side)
        )
        self.image_quality = (
            settings.image_quality if image_quality is None else int(image_quality)
        )
        if self.image_max_side <= 0:
            raise ValueError("image_max_side must be positive")
        if not 1 <= self.image_quality <= 95:
            raise ValueError("image_quality must be between 1 and 95")
        self.timeout = timeout

    def generate(
        self,
        system_prompt: str,
        user_prompt: str,
        image_path: Path,
    ) -> str:
        transport = encode_transport_image(
            image_path,
            compress=self.compress_images,
            max_side=self.image_max_side,
            quality=self.image_quality,
        )
        encoded = base64.b64encode(transport.payload).decode("ascii")
        LOGGER.info(
            "verbalizer image prepared original_bytes=%d transmitted_bytes=%d "
            "base64_bytes=%d original_size=%sx%s transmitted_size=%sx%s "
            "mime_type=%s compression_enabled=%s",
            transport.original_bytes,
            len(transport.payload),
            len(encoded),
            *transport.original_size,
            *transport.transmitted_size,
            transport.mime_type,
            self.compress_images,
        )
        text_before, marker, text_after = user_prompt.partition("<image>")
        content: list[dict[str, Any]] = [{"type": "text", "text": text_before}]
        if marker:
            content.append(
                {
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:{transport.mime_type};base64,{encoded}"
                    },
                }
            )
            if text_after:
                content.append({"type": "text", "text": text_after})

        return create_chat_completion(
            api_key=self.api_key,
            base_url=self.base_url,
            model=self.model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": content},
            ],
            max_tokens=self.max_tokens,
            timeout=self.timeout,
            proxy_url=self.proxy_url,
        )


@dataclass(frozen=True)
class FeedbackResult:
    """Selected feedback signals plus the underlying evaluator evidence."""

    benchmark: str
    feedback: str | None
    reward: float | None
    private_evidence: dict[str, Any]
    sample_id: str | None = None
    used_verbalizer: bool = False
    verbalizer_model: str | None = None
    verbalizer_base_url: str | None = None
    fallback_reason: str | None = None

    def to_dict(self, *, include_private: bool = False) -> dict[str, Any]:
        result: dict[str, Any] = {
            "benchmark": self.benchmark,
            "feedback": self.feedback,
            "reward": self.reward,
            "sample_id": self.sample_id,
        }
        if include_private:
            result["private_evidence"] = self.private_evidence
            result["used_verbalizer"] = self.used_verbalizer
            result["verbalizer_model"] = self.verbalizer_model
            result["verbalizer_base_url"] = self.verbalizer_base_url
            result["fallback_reason"] = self.fallback_reason
        return result

    def to_response_dict(
        self,
        *,
        prompt: str,
        metadata: dict[str, Any] | None,
    ) -> dict[str, Any]:
        """Return the complete service/CLI record without echoing image bytes."""
        return {
            "benchmark": self.benchmark,
            "sample_id": self.sample_id,
            "prompt": prompt,
            "metadata": dict(metadata) if metadata is not None else None,
            "feedback": self.feedback,
            "reward": self.reward,
            "evaluator_result": self.private_evidence,
            "feedback_generation": {
                "used_verbalizer": self.used_verbalizer,
                "model": self.verbalizer_model,
                "base_url": self.verbalizer_base_url,
                "fallback_reason": self.fallback_reason,
            },
        }


_PRIVATE_TERMS = (
    "geneval",
    "mask2former",
    "evaluator",
    "evaluation model",
    "score",
    "confidence",
    "threshold",
    "detected object",
)

_ADVICE_TERMS = (
    "recommend",
    "should ",
    "next attempt",
    "next time",
    "future task",
    "try to",
    "improve by",
    "could be improved",
)


def _public_feedback_error(value: str) -> str | None:
    normalized = " ".join(str(value).split()).strip()
    if not normalized:
        return "empty feedback"
    lowered = normalized.casefold()
    for term in _PRIVATE_TERMS:
        if term.casefold() in lowered:
            return f"private evaluator term leaked: {term}"
    for term in _ADVICE_TERMS:
        if term.casefold() in lowered:
            return f"future-facing advice leaked: {term}"
    if len(normalized) > 800:
        return "feedback is not concise"
    return None


class FeedbackAgent(ABC):
    """Base class shared by all evaluator-grounded feedback agents."""

    benchmark: str

    def __init__(self, verbalizer: FeedbackVerbalizer | None = None) -> None:
        self.verbalizer = verbalizer

    def warmup(self) -> None:
        """Prepare evaluator assets and models before the service becomes ready."""

    @abstractmethod
    def evaluate_image(
        self,
        *,
        task_prompt: str,
        image_path: Path,
        metadata: dict[str, Any] | None,
    ) -> dict[str, Any]:
        """Run the original evaluator and return private grounding evidence."""

    @abstractmethod
    def evaluator_reward(self, private_evidence: dict[str, Any]) -> float:
        """Extract the evaluator's normalized 0-1 score."""

    @abstractmethod
    def fallback_feedback(
        self,
        *,
        task_prompt: str,
        private_evidence: dict[str, Any],
        metadata: dict[str, Any] | None,
    ) -> str:
        """Return safe natural language when no verbalizer is configured."""

    def generate_feedback(
        self,
        *,
        task_prompt: str,
        image_path: str | Path,
        metadata: dict[str, Any] | None = None,
        evaluator_result: dict[str, Any] | None = None,
        sample_id: str | None = None,
        feedback_mode: str = "both",
    ) -> FeedbackResult:
        """Evaluate once, then expose only the feedback channels the caller requested.

        Reward-only requests deliberately skip fallback text and the multimodal
        verbalizer. The evaluator score is already sufficient, so another model call
        would add latency and create an unnecessary evaluator-to-text privacy surface.
        """
        feedback_mode = feedback_mode.strip().casefold()
        if feedback_mode not in {"text", "reward", "both"}:
            raise ValueError("feedback_mode must be 'text', 'reward', or 'both'")
        resolved_image = Path(image_path).expanduser().resolve()
        if not resolved_image.is_file():
            raise FileNotFoundError(f"Submitted image is missing: {resolved_image}")

        evaluation_started = time.monotonic()
        LOGGER.info(
            "evaluator started benchmark=%s sample_id=%s",
            self.benchmark,
            sample_id,
        )
        private_evidence = (
            dict(evaluator_result)
            if evaluator_result is not None
            else dict(
                self.evaluate_image(
                    task_prompt=task_prompt,
                    image_path=resolved_image,
                    metadata=metadata,
                )
            )
        )
        LOGGER.info(
            "evaluator completed benchmark=%s sample_id=%s elapsed_seconds=%.3f",
            self.benchmark,
            sample_id,
            time.monotonic() - evaluation_started,
        )
        raw_reward = float(self.evaluator_reward(private_evidence))
        if not math.isfinite(raw_reward) or not 0.0 <= raw_reward <= 1.0:
            raise RuntimeError("evaluator reward must be between 0 and 1")
        reward = raw_reward if feedback_mode in {"reward", "both"} else None
        feedback: str | None = None
        used_verbalizer = False
        fallback_reason: str | None = None
        if feedback_mode in {"text", "both"}:
            fallback = self.fallback_feedback(
                task_prompt=task_prompt,
                private_evidence=private_evidence,
                metadata=metadata,
            )
            fallback_error = _public_feedback_error(fallback)
            if fallback_error:
                raise RuntimeError(f"Unsafe feedback fallback: {fallback_error}")
            feedback = fallback

            if self.verbalizer is not None:
                user_prompt = build_feedback_prompt(
                    self.benchmark,
                    task_prompt,
                    private_evidence,
                )
                verbalizer_started = time.monotonic()
                LOGGER.info(
                    "verbalizer started benchmark=%s sample_id=%s model=%s",
                    self.benchmark,
                    sample_id,
                    getattr(self.verbalizer, "model", None),
                )
                try:
                    candidate = self.verbalizer.generate(
                        SYSTEM_PROMPT,
                        user_prompt,
                        resolved_image,
                    )
                    candidate = re.sub(r"\s+", " ", candidate).strip()
                    validation_error = _public_feedback_error(candidate)
                    if validation_error:
                        fallback_reason = validation_error
                    else:
                        feedback = candidate
                        used_verbalizer = True
                except Exception as exc:  # evaluator reward remains usable
                    fallback_reason = f"verbalizer failed: {type(exc).__name__}"
                LOGGER.info(
                    "verbalizer completed benchmark=%s sample_id=%s used=%s "
                    "fallback_reason=%s elapsed_seconds=%.3f",
                    self.benchmark,
                    sample_id,
                    used_verbalizer,
                    fallback_reason,
                    time.monotonic() - verbalizer_started,
                )

        return FeedbackResult(
            benchmark=self.benchmark,
            feedback=feedback,
            reward=reward,
            private_evidence=private_evidence,
            sample_id=sample_id,
            used_verbalizer=used_verbalizer,
            verbalizer_model=getattr(self.verbalizer, "model", None),
            verbalizer_base_url=getattr(self.verbalizer, "base_url", None),
            fallback_reason=fallback_reason,
        )

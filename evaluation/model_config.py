"""Shared JSON configuration for evaluator and feedback-model services."""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any


PROJECT_CONFIG_PATH = Path(__file__).resolve().parent / "config.json"
DEFAULT_FEEDBACK_MLLM_MODEL = "Qwen/Qwen3-VL-8B-Instruct"
DEFAULT_FEEDBACK_MLLM_URL = "http://127.0.0.1:8000/v1"


def load_project_config(
    path: str | Path | None = None,
    *,
    required: bool = False,
) -> dict[str, Any]:
    """Load and validate the private evaluation JSON configuration."""
    resolved = Path(path or PROJECT_CONFIG_PATH).expanduser().resolve()
    if not resolved.is_file():
        if required:
            raise FileNotFoundError(f"Evaluation config does not exist: {resolved}")
        return {}
    value = json.loads(resolved.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"Evaluation config must be a JSON object: {resolved}")
    schema_version = value.get("schema_version", 1)
    if schema_version != 1:
        raise ValueError(
            f"Unsupported evaluation config schema_version={schema_version!r}: {resolved}"
        )
    return value


def config_section(
    name: str,
    *,
    path: str | Path | None = None,
) -> dict[str, Any]:
    value = load_project_config(path).get(name, {})
    if value is None:
        return {}
    if not isinstance(value, dict):
        raise ValueError(f"Evaluation config section {name!r} must be an object")
    return value


@dataclass(frozen=True)
class FeedbackModelSettings:
    api_key: str | None
    base_url: str
    model: str
    proxy_url: str | None
    max_tokens: int
    compress_images: bool
    image_max_side: int
    image_quality: int
    timeout_seconds: float
    preflight_timeout_seconds: float
    required: bool


def feedback_model_settings(
    path: str | Path | None = None,
) -> FeedbackModelSettings:
    section = config_section("feedback_mllm", path=path)
    raw_key = section.get("api_key")
    api_key = str(raw_key).strip() if raw_key is not None else ""
    base_url = str(section.get("base_url") or DEFAULT_FEEDBACK_MLLM_URL).strip()
    model = str(section.get("model") or DEFAULT_FEEDBACK_MLLM_MODEL).strip()
    raw_proxy_url = section.get("proxy_url")
    proxy_url = str(raw_proxy_url).strip() if raw_proxy_url is not None else ""
    max_tokens = int(section.get("max_tokens", 512))
    compress_images = bool(section.get("compress_images", True))
    image_max_side = int(section.get("image_max_side", 1024))
    image_quality = int(section.get("image_quality", 80))
    timeout_seconds = float(section.get("timeout_seconds", 60.0))
    preflight_timeout_seconds = float(
        section.get("preflight_timeout_seconds", 20.0)
    )
    required = bool(section.get("required", True))
    if max_tokens <= 0:
        raise ValueError("feedback_mllm.max_tokens must be positive")
    if image_max_side <= 0:
        raise ValueError("feedback_mllm.image_max_side must be positive")
    if not 1 <= image_quality <= 95:
        raise ValueError("feedback_mllm.image_quality must be between 1 and 95")
    if timeout_seconds <= 0:
        raise ValueError("feedback_mllm.timeout_seconds must be positive")
    if preflight_timeout_seconds <= 0:
        raise ValueError(
            "feedback_mllm.preflight_timeout_seconds must be positive"
        )
    return FeedbackModelSettings(
        api_key=api_key or None,
        base_url=base_url,
        model=model,
        proxy_url=proxy_url or None,
        max_tokens=max_tokens,
        compress_images=compress_images,
        image_max_side=image_max_side,
        image_quality=image_quality,
        timeout_seconds=timeout_seconds,
        preflight_timeout_seconds=preflight_timeout_seconds,
        required=required,
    )


def resolve_feedback_api_key(
    explicit: str | None = None,
    *,
    config_path: str | Path | None = None,
) -> str | None:
    """Resolve the API key from an explicit value or evaluation/config.json."""
    if explicit and explicit.strip():
        return explicit.strip()
    return feedback_model_settings(config_path).api_key

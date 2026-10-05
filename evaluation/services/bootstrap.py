"""Shared service bootstrap helpers; contains no feedback prompts."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import logging
import os
from pathlib import Path
import sys

from evaluation.feedback_agents.base import OpenAIFeedbackVerbalizer
from evaluation.model_config import (
    PROJECT_CONFIG_PATH,
    config_section,
    feedback_model_settings,
)


def evaluator_concurrency() -> int:
    """Return the configured number of isolated evaluator instances."""
    value = int(config_section("service").get("evaluator_concurrency", 2))
    if value <= 0:
        raise ValueError("service.evaluator_concurrency must be positive")
    return value


def configure_service_file_logging(
    benchmark: str,
    *,
    log_file: str | Path | None = None,
    record_file: str | Path | None = None,
) -> Path:
    """Write service logs to both a persistent file and process stdout."""
    service_config = config_section("service")
    configured = log_file or service_config.get("log_file")
    if configured:
        log_path = Path(str(configured)).expanduser().resolve()
    else:
        timestamp = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
        run_directory = (
            Path(__file__).resolve().parents[1]
            / "logs"
            / benchmark
            / f"feedback-{benchmark}-{timestamp}-{os.getpid()}"
        )
        log_path = run_directory / "service.log"
    log_path.parent.mkdir(parents=True, exist_ok=True)

    configured_record = record_file or service_config.get("record_file")
    record_path = Path(
        str(configured_record) if configured_record else log_path.parent / "feedback.jsonl"
    ).expanduser().resolve()
    record_path.parent.mkdir(parents=True, exist_ok=True)
    record_path.touch(exist_ok=True)
    os.environ["FEEDBACK_RECORD_FILE"] = str(record_path)

    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(line_buffering=True, write_through=True)
    if hasattr(sys.stderr, "reconfigure"):
        sys.stderr.reconfigure(line_buffering=True, write_through=True)

    log_format = "%(asctime)s | %(levelname)s | %(name)s | %(message)s"
    file_handler = logging.FileHandler(log_path, mode="a", encoding="utf-8")
    stdout_handler = logging.StreamHandler(sys.stdout)
    logging.basicConfig(
        level=str(service_config.get("log_level", "INFO")).upper(),
        format=log_format,
        handlers=[file_handler, stdout_handler],
        force=True,
    )
    logging.getLogger("evaluation.feedback_service").info(
        "%s service logs initialized service_log=%s feedback_records=%s",
        benchmark,
        log_path,
        record_path,
    )
    return log_path


def add_server_arguments(
    parser: argparse.ArgumentParser,
    *,
    benchmark: str,
    default_port: int,
) -> None:
    service_config = config_section("service")
    configured_ports = service_config.get("ports", {})
    if configured_ports is None:
        configured_ports = {}
    if not isinstance(configured_ports, dict):
        raise ValueError("service.ports must be an object in evaluation/config.json")
    parser.add_argument("--host", default=str(service_config.get("host", "127.0.0.1")))
    parser.add_argument(
        "--port",
        type=int,
        default=int(configured_ports.get(benchmark, default_port)),
    )
    parser.add_argument(
        "--service-log-file",
        type=Path,
        help="Explicit service log path for this submitted job.",
    )
    parser.add_argument(
        "--feedback-record-file",
        type=Path,
        help="Explicit feedback JSONL path for this submitted job.",
    )


def optional_verbalizer(
    config_path: str | Path | None = None,
) -> OpenAIFeedbackVerbalizer | None:
    settings = feedback_model_settings(config_path)
    if not settings.api_key:
        if settings.required:
            raise ValueError(
                "feedback_mllm.required is true but feedback_mllm.api_key is "
                f"missing from {Path(config_path or PROJECT_CONFIG_PATH).resolve()}"
            )
        return None
    return OpenAIFeedbackVerbalizer(
        api_key=settings.api_key,
        base_url=settings.base_url,
        model=settings.model,
        proxy_url=settings.proxy_url,
        max_tokens=settings.max_tokens,
        compress_images=settings.compress_images,
        image_max_side=settings.image_max_side,
        image_quality=settings.image_quality,
        timeout=settings.timeout_seconds,
    )

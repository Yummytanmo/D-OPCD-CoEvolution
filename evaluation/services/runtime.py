"""Dependency-free HTTP transport shared by all feedback-agent services."""

from __future__ import annotations

import base64
import binascii
import json
import logging
import os
import tempfile
import threading
import time
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from ipaddress import ip_address
from pathlib import Path
from typing import Any

from evaluation.feedback_agents.base import FeedbackAgent
from evaluation.model_config import config_section


LOGGER = logging.getLogger("evaluation.feedback_service")


class ServiceInputError(ValueError):
    pass


class ServiceBusyError(RuntimeError):
    pass


def _decode_image(value: Any, maximum_bytes: int) -> tuple[bytes, str]:
    if not isinstance(value, str) or not value.strip():
        raise ServiceInputError("image_base64 must be a non-empty string")
    encoded = value.strip()
    suffix = ".png"
    if encoded.startswith("data:"):
        header, separator, encoded = encoded.partition(",")
        if not separator or ";base64" not in header:
            raise ServiceInputError("image_base64 data URI is invalid")
        mime = header[5:].split(";", 1)[0].lower()
        suffix = {
            "image/jpeg": ".jpg",
            "image/webp": ".webp",
            "image/png": ".png",
        }.get(mime, ".png")
    try:
        payload = base64.b64decode(encoded, validate=True)
    except (binascii.Error, ValueError) as exc:
        raise ServiceInputError("image_base64 is not valid base64") from exc
    if not payload:
        raise ServiceInputError("decoded image is empty")
    if len(payload) > maximum_bytes:
        raise ServiceInputError(
            f"decoded image exceeds the {maximum_bytes}-byte service limit"
        )
    return payload, suffix


class FeedbackHTTPService:
    """Transport adapter returning feedback and raw evaluator output separately."""

    def __init__(
        self,
        agent: FeedbackAgent,
        *,
        maximum_image_bytes: int = 25 * 1024 * 1024,
        max_concurrent_requests: int | None = None,
        max_queued_requests: int | None = None,
        queue_timeout_seconds: float | None = None,
        feedback_modes: set[str] | None = None,
    ) -> None:
        self.agent = agent
        self.feedback_modes = frozenset(
            feedback_modes or {"text", "reward", "both"}
        )
        if not self.feedback_modes or not self.feedback_modes <= {
            "text",
            "reward",
            "both",
        }:
            raise ValueError("feedback_modes must contain text, reward, or both")
        self.maximum_image_bytes = maximum_image_bytes
        service_config = config_section("service")
        self.evaluator_concurrency = int(
            getattr(agent, "evaluator_concurrency", 0)
            or service_config.get("evaluator_concurrency", 2)
        )
        self.max_concurrent_requests = int(
            max_concurrent_requests
            if max_concurrent_requests is not None
            else service_config.get("max_concurrent_requests", 16)
        )
        self.max_queued_requests = int(
            max_queued_requests
            if max_queued_requests is not None
            else service_config.get("max_queued_requests", 32)
        )
        self.queue_timeout_seconds = float(
            queue_timeout_seconds
            if queue_timeout_seconds is not None
            else service_config.get("queue_timeout_seconds", 120.0)
        )
        if self.evaluator_concurrency <= 0:
            raise ValueError("service.evaluator_concurrency must be positive")
        if self.max_concurrent_requests <= 0:
            raise ValueError("service.max_concurrent_requests must be positive")
        if self.max_queued_requests < 0:
            raise ValueError("service.max_queued_requests cannot be negative")
        if self.queue_timeout_seconds <= 0:
            raise ValueError("service.queue_timeout_seconds must be positive")
        self._admission_slots = threading.BoundedSemaphore(
            self.max_concurrent_requests + self.max_queued_requests
        )
        self._request_slots = threading.BoundedSemaphore(
            self.max_concurrent_requests
        )
        self._state_lock = threading.Lock()
        self._active_requests = 0
        self._active_feedback_requests = 0
        self._active_calibration_requests = 0
        self._queued_requests = 0
        configured_record = os.environ.get("FEEDBACK_RECORD_FILE")
        self.feedback_record_file = (
            Path(configured_record).expanduser().resolve()
            if configured_record
            else None
        )
        self._feedback_record_lock = threading.Lock()

    @property
    def benchmark(self) -> str:
        return self.agent.benchmark

    def process(
        self,
        payload: dict[str, Any],
        *,
        request_kind: str = "feedback",
        record_response: bool = True,
    ) -> dict[str, Any]:
        if request_kind not in {"feedback", "calibration"}:
            raise ValueError(f"Unknown feedback request kind: {request_kind}")
        if not self._admission_slots.acquire(blocking=False):
            raise ServiceBusyError("feedback request queue is full")
        acquired_request_slot = False
        with self._state_lock:
            self._queued_requests += 1
        try:
            acquired_request_slot = self._request_slots.acquire(
                timeout=self.queue_timeout_seconds
            )
            if not acquired_request_slot:
                raise ServiceBusyError("feedback request queue wait timed out")
            with self._state_lock:
                self._queued_requests -= 1
                self._active_requests += 1
                if request_kind == "calibration":
                    self._active_calibration_requests += 1
                else:
                    self._active_feedback_requests += 1
            return self._process(
                payload,
                request_kind=request_kind,
                record_response=record_response,
            )
        finally:
            with self._state_lock:
                if acquired_request_slot:
                    self._active_requests -= 1
                    if request_kind == "calibration":
                        self._active_calibration_requests -= 1
                    else:
                        self._active_feedback_requests -= 1
                else:
                    self._queued_requests -= 1
            if acquired_request_slot:
                self._request_slots.release()
            self._admission_slots.release()

    def concurrency_status(self) -> dict[str, int | float]:
        with self._state_lock:
            active = self._active_requests
            active_feedback = self._active_feedback_requests
            active_calibration = self._active_calibration_requests
            queued = self._queued_requests
        return {
            "active_requests": active,
            "active_feedback_requests": active_feedback,
            "active_calibration_requests": active_calibration,
            "queued_requests": queued,
            "max_concurrent_requests": self.max_concurrent_requests,
            "max_queued_requests": self.max_queued_requests,
            "queue_timeout_seconds": self.queue_timeout_seconds,
            "evaluator_concurrency": self.evaluator_concurrency,
        }

    def _process(
        self,
        payload: dict[str, Any],
        *,
        request_kind: str,
        record_response: bool,
    ) -> dict[str, Any]:
        """Keep evaluator execution shared while allowing callers to skip text output."""
        if not isinstance(payload, dict):
            raise ServiceInputError("request body must be a JSON object")
        task_prompt = payload.get("prompt")
        if not isinstance(task_prompt, str) or not task_prompt.strip():
            raise ServiceInputError("prompt must be a non-empty string")
        metadata = payload.get("metadata")
        if metadata is not None and not isinstance(metadata, dict):
            raise ServiceInputError("metadata must be a JSON object")
        if self.benchmark in {"geneval", "geneval2", "wise", "r2ibench"} and metadata is None:
            raise ServiceInputError(f"{self.benchmark} requests require metadata")
        if self.benchmark == "r2ibench":
            from evaluation.evaluators.r2ibench import (
                build_r2ibench_messages,
                unique_checklist_ids,
                validate_r2i_checklist,
            )
            try:
                validate_r2i_checklist(metadata.get("checklist"))
                build_r2ibench_messages(
                    prompt=task_prompt,
                    checklist=unique_checklist_ids(metadata["checklist"]),
                    image_data_uri="request-validation",
                )
            except (TypeError, ValueError, KeyError) as error:
                raise ServiceInputError(str(error)) from error
        sample_id = payload.get("sample_id")
        if sample_id is not None and not isinstance(sample_id, str):
            raise ServiceInputError("sample_id must be a string")
        feedback_mode = str(payload.get("feedback_mode") or "both").casefold()
        if feedback_mode not in {"text", "reward", "both"}:
            raise ServiceInputError(
                "feedback_mode must be 'text', 'reward', or 'both'"
            )
        if feedback_mode not in self.feedback_modes:
            raise ServiceInputError(
                f"feedback_mode {feedback_mode!r} is unavailable; service supports "
                + ", ".join(sorted(self.feedback_modes))
            )

        image_bytes, suffix = _decode_image(
            payload.get("image_base64"),
            self.maximum_image_bytes,
        )
        normalized_prompt = task_prompt.strip()
        request_started = time.monotonic()
        LOGGER.info(
            "feedback request started benchmark=%s request_kind=%s "
            "sample_id=%s image_bytes=%d",
            self.benchmark,
            request_kind,
            sample_id,
            len(image_bytes),
        )
        with tempfile.TemporaryDirectory(prefix=f"{self.benchmark}-request-") as temp:
            image_path = Path(temp) / f"submitted{suffix}"
            image_path.write_bytes(image_bytes)
            result = self.agent.generate_feedback(
                task_prompt=normalized_prompt,
                image_path=image_path,
                metadata=metadata,
                sample_id=sample_id,
                feedback_mode=feedback_mode,
            )
        response = result.to_response_dict(
            prompt=normalized_prompt,
            metadata=metadata,
        )
        if record_response:
            self._append_feedback_record(response)
        LOGGER.info(
            "feedback request completed benchmark=%s request_kind=%s sample_id=%s "
            "used_verbalizer=%s elapsed_seconds=%.3f",
            self.benchmark,
            request_kind,
            sample_id,
            result.used_verbalizer,
            time.monotonic() - request_started,
        )
        return response

    def _append_feedback_record(self, response: dict[str, Any]) -> None:
        """Append one complete, image-free feedback response as JSONL."""
        if self.feedback_record_file is None:
            return
        line = json.dumps(response, ensure_ascii=False, separators=(",", ":"))
        with self._feedback_record_lock:
            with self.feedback_record_file.open("a", encoding="utf-8") as stream:
                stream.write(line)
                stream.write("\n")


class _FeedbackRequestHandler(BaseHTTPRequestHandler):
    server: "FeedbackHTTPServer"

    def _write_json(self, status: HTTPStatus, payload: dict[str, Any]) -> None:
        body = json.dumps(payload, ensure_ascii=False, separators=(",", ":")).encode(
            "utf-8"
        )
        self.send_response(status.value)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self) -> None:
        if self.path in ("/health", "/ready"):
            runtime = getattr(self.server.feedback_service.agent, "_runtime", None)
            self._write_json(
                HTTPStatus.OK,
                {
                    "status": "ready",
                    "judge_model": getattr(runtime, "model", None),
                    "benchmark": self.server.feedback_service.benchmark,
                    "feedback_modes": sorted(
                        self.server.feedback_service.feedback_modes
                    ),
                    "concurrency": self.server.feedback_service.concurrency_status(),
                },
            )
            return
        self._write_json(
            HTTPStatus.NOT_FOUND,
            {"error": "not_found"},
        )

    def do_POST(self) -> None:
        if self.path not in ("/feedback", "/calibration"):
            self._write_json(HTTPStatus.NOT_FOUND, {"error": "not_found"})
            return
        is_calibration = self.path == "/calibration"
        if is_calibration and not ip_address(self.client_address[0]).is_loopback:
            self._write_json(HTTPStatus.FORBIDDEN, {"error": "local_only"})
            return
        try:
            content_length = int(self.headers.get("Content-Length", "0"))
        except ValueError:
            self._write_json(HTTPStatus.BAD_REQUEST, {"error": "invalid_content_length"})
            return
        maximum_body = self.server.feedback_service.maximum_image_bytes * 2
        if content_length <= 0 or content_length > maximum_body:
            self._write_json(HTTPStatus.REQUEST_ENTITY_TOO_LARGE, {"error": "request_too_large"})
            return
        try:
            payload = json.loads(self.rfile.read(content_length))
            result = self.server.feedback_service.process(
                payload,
                request_kind="calibration" if is_calibration else "feedback",
                record_response=not is_calibration,
            )
        except (json.JSONDecodeError, UnicodeDecodeError, ServiceInputError) as exc:
            self._write_json(
                HTTPStatus.BAD_REQUEST,
                {"error": "invalid_request", "message": str(exc)},
            )
            return
        except ServiceBusyError as exc:
            self._write_json(
                HTTPStatus.TOO_MANY_REQUESTS,
                {"error": "service_busy", "message": str(exc)},
            )
            return
        except Exception:
            LOGGER.exception(
                "feedback request failed benchmark=%s",
                self.server.feedback_service.benchmark,
            )
            self._write_json(
                HTTPStatus.INTERNAL_SERVER_ERROR,
                {"error": "feedback_generation_failed"},
            )
            return
        self._write_json(HTTPStatus.OK, result)

    def log_message(self, message: str, *values: Any) -> None:
        LOGGER.info("%s - %s", self.address_string(), message % values)


class FeedbackHTTPServer(ThreadingHTTPServer):
    daemon_threads = True
    request_queue_size = 128

    def __init__(self, address: tuple[str, int], service: FeedbackHTTPService):
        self.feedback_service = service
        super().__init__(address, _FeedbackRequestHandler)


def serve_feedback_agent(
    agent: FeedbackAgent,
    *,
    host: str,
    port: int,
    warmup: bool = True,
    feedback_modes: set[str] | None = None,
) -> None:
    service_config = config_section("service")
    logging.basicConfig(
        level=str(service_config.get("log_level", "INFO")).upper(),
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )
    if warmup:
        LOGGER.info("deploying and loading %s evaluator assets", agent.benchmark)
        agent.warmup()
        LOGGER.info("%s evaluator assets are ready", agent.benchmark)
    service = FeedbackHTTPService(agent, feedback_modes=feedback_modes)
    LOGGER.info(
        "%s concurrency configured evaluator_workers=%d request_workers=%d "
        "queue_capacity=%d queue_timeout_seconds=%.1f",
        agent.benchmark,
        service.evaluator_concurrency,
        service.max_concurrent_requests,
        service.max_queued_requests,
        service.queue_timeout_seconds,
    )
    server = FeedbackHTTPServer((host, port), service)
    LOGGER.info(
        "%s feedback agent is serving on http://%s:%d/feedback",
        agent.benchmark,
        host,
        port,
    )
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        LOGGER.info("shutting down %s feedback service", agent.benchmark)
    finally:
        server.server_close()

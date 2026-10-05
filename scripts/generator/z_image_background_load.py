#!/usr/bin/env python3
"""Continuously replay real prompts against a loopback Z-Image service."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from datetime import datetime, timezone
from ipaddress import ip_address
import json
from pathlib import Path
import random
import signal
import threading
import time
from typing import Any
import urllib.error
import urllib.parse
import urllib.request


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG_PATH = PROJECT_ROOT / "configs" / "generator.z-image.json"


@dataclass(frozen=True)
class LoadSettings:
    enabled: bool
    source_jsonl: Path
    max_prompts: int
    shuffle_seed: int
    request_interval_seconds: float
    error_backoff_seconds: float
    ready_timeout_seconds: float
    request_timeout_seconds: float


@dataclass(frozen=True)
class CalibrationPrompt:
    sample_id: str
    prompt: str
    seed: int


def _number(value: Any, name: str, *, allow_zero: bool = False) -> float:
    number = float(value)
    if number < 0 or (number == 0 and not allow_zero):
        qualifier = "non-negative" if allow_zero else "positive"
        raise ValueError(f"background_load.{name} must be {qualifier}")
    return number


def require_loopback_service(service_url: str) -> None:
    parsed = urllib.parse.urlsplit(service_url)
    if parsed.scheme not in {"http", "https"} or not parsed.hostname:
        raise ValueError(f"Invalid Z-Image background-load URL: {service_url!r}")
    try:
        is_loopback = ip_address(parsed.hostname).is_loopback
    except ValueError:
        is_loopback = parsed.hostname.casefold() == "localhost"
    if not is_loopback:
        raise ValueError(
            "Z-Image background load may only call a loopback service; "
            f"received {service_url!r}"
        )


def load_settings(config_path: str | Path) -> LoadSettings:
    path = Path(config_path).expanduser().resolve()
    value = json.loads(path.read_text(encoding="utf-8"))
    if value.get("schema_version", 1) != 1:
        raise ValueError("Unsupported Z-Image service config schema_version")
    configured = value.get("background_load", {})
    if not isinstance(configured, dict):
        raise ValueError("background_load must be a JSON object")
    enabled = bool(configured.get("enabled", False))
    source = Path(str(configured.get("source_jsonl", ""))).expanduser()
    if not source.is_absolute():
        source = PROJECT_ROOT / source
    max_prompts = int(configured.get("max_prompts", 50))
    if max_prompts <= 0:
        raise ValueError("background_load.max_prompts must be positive")
    if enabled and not source.is_file():
        raise FileNotFoundError(f"Z-Image calibration prompts are missing: {source}")
    return LoadSettings(
        enabled=enabled,
        source_jsonl=source.resolve(),
        max_prompts=max_prompts,
        shuffle_seed=int(configured.get("shuffle_seed", 42)),
        request_interval_seconds=_number(
            configured.get("request_interval_seconds", 1.0),
            "request_interval_seconds",
            allow_zero=True,
        ),
        error_backoff_seconds=_number(
            configured.get("error_backoff_seconds", 5.0),
            "error_backoff_seconds",
        ),
        ready_timeout_seconds=_number(
            configured.get("ready_timeout_seconds", 900.0),
            "ready_timeout_seconds",
        ),
        request_timeout_seconds=_number(
            configured.get("request_timeout_seconds", 600.0),
            "request_timeout_seconds",
        ),
    )


def load_prompts(settings: LoadSettings) -> list[CalibrationPrompt]:
    prompts: list[CalibrationPrompt] = []
    with settings.source_jsonl.open("r", encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, 1):
            if not line.strip():
                continue
            try:
                item = json.loads(line)
                prompt = str(item["prompt"]).strip()
            except (KeyError, TypeError, json.JSONDecodeError) as exc:
                raise ValueError(
                    f"Invalid prompt record at {settings.source_jsonl}:{line_number}"
                ) from exc
            if not prompt:
                continue
            prompts.append(
                CalibrationPrompt(
                    sample_id=str(item.get("sample_id") or f"line:{line_number}"),
                    prompt=prompt,
                    seed=int(item.get("inference_seed", settings.shuffle_seed)),
                )
            )
    if not prompts:
        raise RuntimeError(f"No prompts found in {settings.source_jsonl}")
    random.Random(settings.shuffle_seed).shuffle(prompts)
    return prompts[: settings.max_prompts]


def _open_direct(request: urllib.request.Request, timeout: float):
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
    return opener.open(request, timeout=timeout)


def wait_until_ready(service_url: str, timeout: float) -> None:
    deadline = time.monotonic() + timeout
    last_error: Exception | None = None
    while time.monotonic() < deadline:
        try:
            request = urllib.request.Request(
                f"{service_url.rstrip('/')}/ready",
                headers={"Accept": "application/json"},
            )
            with _open_direct(request, 5.0) as response:
                value = json.loads(response.read().decode("utf-8"))
            if value.get("status") == "ready" and value.get("generator") == "z-image":
                return
        except (
            OSError,
            ValueError,
            urllib.error.HTTPError,
            urllib.error.URLError,
        ) as exc:
            last_error = exc
        time.sleep(2.0)
    raise RuntimeError(
        f"Z-Image did not become reachable in {timeout:.0f}s: {last_error}"
    )


class JsonlLog:
    def __init__(self, path: str | Path) -> None:
        self.path = Path(path).expanduser().resolve()
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()

    def write(self, **values: Any) -> None:
        record = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            **values,
        }
        line = json.dumps(record, ensure_ascii=False, separators=(",", ":"))
        with self._lock:
            with self.path.open("a", encoding="utf-8") as stream:
                stream.write(line)
                stream.write("\n")
        print(line, flush=True)


def run_load(
    *,
    service_url: str,
    settings: LoadSettings,
    prompts: list[CalibrationPrompt],
    log_path: str | Path,
) -> None:
    stop = threading.Event()

    def request_stop(_signum: int, _frame: object) -> None:
        stop.set()

    signal.signal(signal.SIGINT, request_stop)
    signal.signal(signal.SIGTERM, request_stop)
    wait_until_ready(service_url, settings.ready_timeout_seconds)
    log = JsonlLog(log_path)
    log.write(
        event="sidecar_started",
        prompts=len(prompts),
        request_interval_seconds=settings.request_interval_seconds,
    )
    sequence = 0
    while not stop.is_set():
        item = prompts[sequence % len(prompts)]
        sequence += 1
        query = urllib.parse.urlencode({"prompt": item.prompt, "seed": item.seed})
        request = urllib.request.Request(
            f"{service_url.rstrip('/')}/calibration?{query}",
            data=b"",
            headers={"Accept": "application/json"},
            method="POST",
        )
        started = time.monotonic()
        try:
            with _open_direct(request, settings.request_timeout_seconds) as response:
                result = json.loads(response.read().decode("utf-8"))
            if result.get("status") != "completed":
                raise RuntimeError(f"Invalid Z-Image calibration response: {result}")
            log.write(
                event="request_completed",
                sequence=sequence,
                source_sample_id=item.sample_id,
                image_bytes=int(result["image_bytes"]),
                image_sha256=str(result["image_sha256"]),
                elapsed_seconds=round(time.monotonic() - started, 3),
            )
            stop.wait(settings.request_interval_seconds)
        except Exception as exc:  # Keep calibration alive across transient errors.
            log.write(
                event="request_failed",
                sequence=sequence,
                source_sample_id=item.sample_id,
                error=str(exc),
            )
            stop.wait(settings.error_backoff_seconds)
    log.write(event="sidecar_stopped")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG_PATH)
    parser.add_argument("--service-url", default="http://127.0.0.1:8001")
    parser.add_argument("--log-file", type=Path, required=True)
    parser.add_argument(
        "--check",
        action="store_true",
        help="Validate configuration and prompt data without sending requests.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    require_loopback_service(args.service_url)
    settings = load_settings(args.config)
    if not settings.enabled:
        print("Z-Image background load is disabled", flush=True)
        return 3 if args.check else 0
    prompts = load_prompts(settings)
    print(
        f"Z-Image background load prompts={len(prompts)} "
        f"source={settings.source_jsonl}",
        flush=True,
    )
    if args.check:
        return 0
    run_load(
        service_url=args.service_url,
        settings=settings,
        prompts=prompts,
        log_path=args.log_file,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

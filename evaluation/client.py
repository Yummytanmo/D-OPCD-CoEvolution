#!/usr/bin/env python3
"""CLI client for a running evaluator-grounded feedback service."""

from __future__ import annotations

import argparse
import base64
import json
import math
import sys
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any


DEFAULT_SERVICE_URLS = {
    "geneval": "http://127.0.0.1:8101",
    "geneval2": "http://127.0.0.1:8104",
    "wise": "http://127.0.0.1:8105",
    "r2ibench": "http://127.0.0.1:8108",
}
DEFAULT_CLIENT_CONFIG_PATH = Path(__file__).resolve().parent / "client_config.json"
FEEDBACK_MODES = {"text", "reward", "both"}


def _evaluator_reward(value: dict[str, Any]) -> float | None:
    evidence = value.get("evaluator_result")
    if not isinstance(evidence, dict):
        return None
    if "correct" in evidence:
        return float(bool(evidence["correct"]))
    for key in ("soft_tifa_gm", "score", "alignment_value", "normalized_accuracy", "reward", "value"):
        if key in evidence:
            return float(evidence[key])
    return None


def normalize_feedback_result(
    value: dict[str, Any],
    *,
    feedback_mode: str = "both",
) -> dict[str, Any]:
    """Select learning signals and upgrade older persisted service responses.

    Older runs stored only ``evaluator_result``. Deriving the same top-level reward while
    loading them avoids forcing a new evaluation just because the wire format evolved.
    """
    mode = feedback_mode.strip().casefold()
    if mode not in FEEDBACK_MODES:
        raise ValueError("feedback_mode must be 'text', 'reward', or 'both'")
    if not isinstance(value, dict):
        raise RuntimeError("Feedback service returned an invalid response")

    result = dict(value)
    raw_text = result.get("feedback")
    text = raw_text.strip() if isinstance(raw_text, str) and raw_text.strip() else None
    raw_reward = result.get("reward")
    reward = (
        float(raw_reward)
        if isinstance(raw_reward, (int, float)) and not isinstance(raw_reward, bool)
        else None
    )
    # Prefer the service's explicit reward, but retain resumability for cached responses
    # written before feedback agents exposed that field.
    if reward is None and mode in {"reward", "both"}:
        reward = _evaluator_reward(result)
    if reward is not None and (
        not math.isfinite(reward) or not 0.0 <= reward <= 1.0
    ):
        raise RuntimeError("Feedback reward must be between 0 and 1")

    result["feedback"] = text if mode in {"text", "both"} else None
    result["reward"] = reward if mode in {"reward", "both"} else None
    if result["feedback"] is None and result["reward"] is None:
        raise RuntimeError("Feedback service returned no selected feedback signal")
    return result


def load_client_config(path: str | Path | None = None) -> dict[str, Any]:
    """Load the optional client-only service routing configuration."""
    resolved = Path(path or DEFAULT_CLIENT_CONFIG_PATH).expanduser().resolve()
    if not resolved.is_file():
        if path is not None:
            raise FileNotFoundError(f"Client config does not exist: {resolved}")
        return {}
    value = json.loads(resolved.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"Client config must be a JSON object: {resolved}")
    if value.get("schema_version", 1) != 1:
        raise ValueError("Unsupported client config schema_version")
    services = value.get("services", {})
    if not isinstance(services, dict):
        raise ValueError("Client config services must be a JSON object")
    for benchmark, service_url in services.items():
        if benchmark not in DEFAULT_SERVICE_URLS:
            raise ValueError(f"Unknown configured benchmark: {benchmark}")
        if not isinstance(service_url, str) or not service_url.strip():
            raise ValueError(f"Invalid service URL for benchmark: {benchmark}")
    default_benchmark = value.get("default_benchmark")
    if default_benchmark is not None and default_benchmark not in DEFAULT_SERVICE_URLS:
        raise ValueError(f"Unknown default benchmark: {default_benchmark}")
    timeout = float(value.get("timeout_seconds", 600.0))
    if timeout <= 0:
        raise ValueError("Client config timeout_seconds must be positive")
    bypass_proxy = value.get("bypass_proxy", False)
    if not isinstance(bypass_proxy, bool):
        raise ValueError("Client config bypass_proxy must be a boolean")
    return value


def _read_json_object(path: Path | None) -> dict[str, Any] | None:
    if path is None:
        return None
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"Expected a JSON object: {path}")
    return value


def _service_url(
    benchmark: str | None,
    explicit: str | None,
    configured: dict[str, Any] | None = None,
) -> str:
    if explicit:
        return explicit.rstrip("/")
    if benchmark:
        configured_url = (configured or {}).get(benchmark)
        if configured_url:
            return str(configured_url).rstrip("/")
        return DEFAULT_SERVICE_URLS[benchmark]
    raise ValueError(
        "Provide --service-url/--benchmark or set default_benchmark in client config"
    )


def request_feedback_result(
    *,
    service_url: str,
    prompt: str,
    image_path: str | Path,
    metadata: dict[str, Any] | None = None,
    sample_id: str | None = None,
    timeout: float = 600.0,
    bypass_proxy: bool = False,
    feedback_mode: str = "both",
) -> dict[str, Any]:
    """Call one feedback service and return its complete JSON result."""
    image = Path(image_path).expanduser().resolve()
    if not image.is_file():
        raise FileNotFoundError(f"Image is missing: {image}")
    payload: dict[str, Any] = {
        "prompt": prompt,
        "image_base64": base64.b64encode(image.read_bytes()).decode("ascii"),
    }
    if sample_id is not None:
        payload["sample_id"] = sample_id
    if metadata is not None:
        payload["metadata"] = metadata
    payload["feedback_mode"] = feedback_mode

    request = urllib.request.Request(
        f"{service_url.rstrip('/')}/feedback",
        data=json.dumps(payload, ensure_ascii=False).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    open_request = urllib.request.urlopen
    if bypass_proxy:
        open_request = urllib.request.build_opener(
            urllib.request.ProxyHandler({})
        ).open
    try:
        with open_request(request, timeout=timeout) as response:
            result = json.loads(response.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        body = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError(
            f"Feedback service returned HTTP {exc.code}: {body}"
        ) from exc
    except urllib.error.URLError as exc:
        raise RuntimeError(f"Cannot reach feedback service: {exc.reason}") from exc

    return normalize_feedback_result(result, feedback_mode=feedback_mode)


def request_feedback(
    *,
    service_url: str,
    prompt: str,
    image_path: str | Path,
    metadata: dict[str, Any] | None = None,
    sample_id: str | None = None,
    timeout: float = 600.0,
    bypass_proxy: bool = False,
) -> str:
    """Call one feedback service and extract only its feedback string."""
    result = request_feedback_result(
        service_url=service_url,
        prompt=prompt,
        image_path=image_path,
        metadata=metadata,
        sample_id=sample_id,
        timeout=timeout,
        bypass_proxy=bypass_proxy,
        feedback_mode="text",
    )
    return str(result["feedback"])


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "-b",
        "--benchmark",
        choices=sorted(DEFAULT_SERVICE_URLS),
        help="Select the default local service URL for this benchmark.",
    )
    parser.add_argument(
        "--service-url",
        help="Explicit service base URL, such as http://10.0.0.8:8103.",
    )
    parser.add_argument(
        "--config",
        type=Path,
        help=f"Client config path (default: {DEFAULT_CLIENT_CONFIG_PATH}).",
    )
    parser.add_argument("-p", "--prompt", required=True)
    parser.add_argument("-i", "--image", type=Path, required=True)
    parser.add_argument("--metadata-json", type=Path)
    parser.add_argument("--sample-id")
    parser.add_argument("--timeout", type=float)
    parser.add_argument(
        "--feedback-mode",
        choices=sorted(FEEDBACK_MODES),
        default="both",
        help="Return natural-language text, evaluator reward, or both.",
    )
    proxy_group = parser.add_mutually_exclusive_group()
    proxy_group.add_argument(
        "--bypass-proxy",
        action="store_true",
        dest="bypass_proxy",
        default=None,
        help="Ignore environment proxies when connecting to the feedback service.",
    )
    proxy_group.add_argument(
        "--use-environment-proxy",
        action="store_false",
        dest="bypass_proxy",
        help="Allow urllib to use environment proxy variables.",
    )
    parser.add_argument(
        "-j",
        "--json",
        action="store_true",
        help="Print a JSON object instead of the feedback text alone.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    try:
        client_config = load_client_config(args.config)
        benchmark = args.benchmark or client_config.get("default_benchmark")
        services = client_config.get("services", {})
        url = _service_url(benchmark, args.service_url, services)
        timeout = (
            args.timeout
            if args.timeout is not None
            else float(client_config.get("timeout_seconds", 600.0))
        )
        bypass_proxy = (
            args.bypass_proxy
            if args.bypass_proxy is not None
            else bool(client_config.get("bypass_proxy", False))
        )
        result = request_feedback_result(
            service_url=url,
            prompt=args.prompt,
            image_path=args.image,
            metadata=_read_json_object(args.metadata_json),
            sample_id=args.sample_id,
            timeout=timeout,
            bypass_proxy=bypass_proxy,
            feedback_mode=args.feedback_mode,
        )
    except (OSError, ValueError, RuntimeError, json.JSONDecodeError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1

    if args.json:
        print(json.dumps(result, ensure_ascii=False, indent=2))
    elif args.feedback_mode == "reward":
        print(result["reward"])
    else:
        print(result["feedback"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

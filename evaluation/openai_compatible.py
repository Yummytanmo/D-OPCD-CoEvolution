"""Small OpenAI-compatible chat client shared by preflight and feedback."""

from __future__ import annotations

import json
from ipaddress import ip_address
import urllib.error
import urllib.request
from urllib.parse import urlsplit
from typing import Any


def create_chat_completion(
    *,
    api_key: str,
    base_url: str,
    model: str,
    messages: list[dict[str, Any]],
    max_tokens: int,
    timeout: float,
    proxy_url: str | None = None,
) -> str:
    """Return one non-streaming chat completion without automatic retries."""
    request = urllib.request.Request(
        f"{base_url.rstrip('/')}/chat/completions",
        data=json.dumps(
            {
                "model": model,
                "messages": messages,
                "max_tokens": max_tokens,
                "stream": False,
            },
            ensure_ascii=False,
        ).encode("utf-8"),
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        },
        method="POST",
    )
    parsed = urlsplit(base_url)
    try:
        is_loopback = bool(parsed.hostname) and ip_address(parsed.hostname).is_loopback
    except ValueError:
        is_loopback = (parsed.hostname or "").casefold() == "localhost"
    open_request = urllib.request.urlopen
    if is_loopback:
        # A local vLLM endpoint must never be routed through an environment or
        # explicitly configured remote proxy.
        open_request = urllib.request.build_opener(
            urllib.request.ProxyHandler({})
        ).open
    elif proxy_url:
        proxy_handler = urllib.request.ProxyHandler(
            {"http": proxy_url, "https": proxy_url}
        )
        open_request = urllib.request.build_opener(proxy_handler).open
    try:
        with open_request(request, timeout=timeout) as response:
            result = json.loads(response.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        body = exc.read().decode("utf-8", errors="replace")[:1000]
        raise RuntimeError(f"Qwen API returned HTTP {exc.code}: {body}") from exc
    except urllib.error.URLError as exc:
        raise RuntimeError(f"Cannot reach Qwen API: {exc.reason}") from exc

    try:
        content = result["choices"][0]["message"]["content"]
    except (KeyError, IndexError, TypeError) as exc:
        raise RuntimeError("Qwen API returned an invalid chat-completion response") from exc
    if not isinstance(content, str) or not content.strip():
        raise RuntimeError("Qwen API returned empty chat-completion content")
    return content.strip()

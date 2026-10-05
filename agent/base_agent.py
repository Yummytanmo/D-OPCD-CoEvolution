import abc
import base64
import hashlib
import io
import os
import threading
import time
from collections import OrderedDict
from typing import Any

import requests
import httpx
from openai import OpenAI

from agent.tooling import (
    FunctionTool,
    ToolLoopResult,
    tool_result_content,
)
from PIL import Image, ImageOps

from agent.skill_manager import SkillManager


def _env_bool(name: str, default: bool) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    return value.strip().lower() not in {"0", "false", "no", "off"}


class BaseAgent(abc.ABC):
    def __init__(
        self,
        gen_url: str,
        mllm_url: str,
        *,
        skill_manager: SkillManager | None = None,
        generator_config: dict | None = None,
        mllm_config: dict | None = None,
    ):
        generator_config = dict(generator_config or {})
        mllm_config = dict(mllm_config or {})
        self.gen_url = gen_url
        self.mllm_url = mllm_url
        self.mllm_model = str(
            mllm_config.get("model")
            or os.getenv("GEMS_MLLM_MODEL", "YOUR_MULTIMODAL_MODEL")
        )
        api_key_env = str(mllm_config.get("api_key_env") or "").strip()
        self.mllm_api_key = (
            (os.getenv(api_key_env) if api_key_env else None)
            or os.getenv("GEMS_MLLM_API_KEY")
            or os.getenv("OPENAI_API_KEY")
            or "none"
        )
        self.mllm_timeout_seconds = float(
            mllm_config.get("timeout_seconds")
            or os.getenv("GEMS_MLLM_TIMEOUT_SECONDS", "180")
        )
        self.mllm_max_tokens = int(
            mllm_config.get("max_tokens")
            or os.getenv("GEMS_MLLM_MAX_TOKENS", "16384")
        )
        raw_extra_body = mllm_config.get("extra_body")
        if raw_extra_body is None:
            self.mllm_extra_body: dict[str, Any] = {}
        elif not isinstance(raw_extra_body, dict):
            raise ValueError("mllm.extra_body must be a JSON object")
        else:
            self.mllm_extra_body = dict(raw_extra_body)
        self.mllm_max_retries = int(
            mllm_config.get("max_retries")
            if mllm_config.get("max_retries") is not None
            else os.getenv("GEMS_MLLM_MAX_RETRIES", "2")
        )
        self.mllm_compress_images = bool(
            mllm_config.get("compress_images")
            if "compress_images" in mllm_config
            else _env_bool("GEMS_MLLM_COMPRESS_IMAGES", True)
        )
        self.mllm_image_format = str(
            mllm_config.get("image_format")
            or os.getenv("GEMS_MLLM_IMAGE_FORMAT", "webp")
        ).lower()
        if self.mllm_image_format not in {"jpeg", "webp"}:
            raise ValueError("mllm.image_format must be 'jpeg' or 'webp'")
        self.mllm_image_total_max_bytes = int(
            mllm_config.get("image_total_max_bytes")
            or os.getenv("GEMS_MLLM_IMAGE_TOTAL_MAX_BYTES", "800000")
        )
        if self.mllm_image_total_max_bytes <= 0:
            raise ValueError("mllm.image_total_max_bytes must be positive")
        if self.mllm_max_retries < 0:
            raise ValueError("mllm.max_retries must be non-negative")

        self.mllm_trust_env = bool(
            mllm_config.get("trust_env")
            if "trust_env" in mllm_config
            else _env_bool("GEMS_MLLM_TRUST_ENV", True)
        )
        self._mllm_http_client = httpx.Client(trust_env=self.mllm_trust_env)
        self.client = OpenAI(
            api_key=self.mllm_api_key,
            base_url=self.mllm_url,
            timeout=self.mllm_timeout_seconds,
            max_retries=self.mllm_max_retries,
            http_client=self._mllm_http_client,
        )
        self.generator_timeout_seconds = float(
            generator_config.get("timeout_seconds", 600)
        )
        self.generator_max_attempts = int(
            generator_config.get("max_attempts", 2)
        )
        self.generator_retry_backoff_seconds = float(
            generator_config.get("retry_backoff_seconds", 2)
        )
        if self.generator_timeout_seconds <= 0:
            raise ValueError("generator.timeout_seconds must be positive")
        if self.generator_max_attempts <= 0:
            raise ValueError("generator.max_attempts must be positive")
        if self.generator_retry_backoff_seconds < 0:
            raise ValueError("generator.retry_backoff_seconds must be non-negative")
        self._generator_session = requests.Session()
        self._generator_session.trust_env = bool(
            generator_config.get("trust_env")
            if "trust_env" in generator_config
            else _env_bool("GEMS_GENERATOR_TRUST_ENV", True)
        )
        self.skill_manager = skill_manager or SkillManager()
        self._counter_lock = threading.Lock()
        self._completion_local = threading.local()
        self._generator_calls = 0
        self._mllm_calls = 0
        self._image_cache_lock = threading.Lock()
        self._image_cache: OrderedDict[str, tuple[bytes, str]] = OrderedDict()
        self._image_cache_max_entries = 16

    def run(self, prompt: str):
        pass

    def get_generator_calls(self):
        with self._counter_lock:
            return self._generator_calls

    @property
    def generator_calls(self):
        return self.get_generator_calls()

    @generator_calls.setter
    def generator_calls(self, value):
        with self._counter_lock:
            self._generator_calls = int(value)

    def get_mllm_calls(self):
        with self._counter_lock:
            return self._mllm_calls

    @property
    def mllm_calls(self):
        return self.get_mllm_calls()

    @mllm_calls.setter
    def mllm_calls(self, value):
        with self._counter_lock:
            self._mllm_calls = int(value)

    def reset_counters(self):
        with self._counter_lock:
            self._generator_calls = 0
            self._mllm_calls = 0

    def reset_call_counters(self):
        """Compatibility name used by the structured evolution runner."""
        self.reset_counters()

    def generate(self, prompt: str, seed: int | None = None):
        params = {"prompt": prompt}
        if seed is not None:
            params["seed"] = seed
        with self._counter_lock:
            self._generator_calls += 1
        for attempt in range(1, self.generator_max_attempts + 1):
            try:
                response = self._generator_session.post(
                    self.gen_url,
                    params=params,
                    timeout=self.generator_timeout_seconds,
                )
                if (
                    response.status_code not in {429, 500, 502, 503, 504}
                    or attempt >= self.generator_max_attempts
                ):
                    response.raise_for_status()
                    return response.content
                response.close()
            except (requests.ConnectionError, requests.Timeout):
                if attempt >= self.generator_max_attempts:
                    raise
            time.sleep(self.generator_retry_backoff_seconds * (2 ** (attempt - 1)))
        raise RuntimeError("generator request exhausted all attempts")

    @staticmethod
    def _source_mime_type(image_bytes: bytes) -> str | None:
        if image_bytes.startswith(b"\x89PNG\r\n\x1a\n"):
            return "image/png"
        if image_bytes.startswith(b"\xff\xd8\xff"):
            return "image/jpeg"
        if image_bytes.startswith((b"GIF87a", b"GIF89a")):
            return "image/gif"
        if image_bytes.startswith(b"RIFF") and image_bytes[8:12] == b"WEBP":
            return "image/webp"
        return None

    @staticmethod
    def _rgb_image(image: Image.Image) -> Image.Image:
        image = ImageOps.exif_transpose(image)
        if image.mode in {"RGBA", "LA"} or (
            image.mode == "P" and "transparency" in image.info
        ):
            rgba = image.convert("RGBA")
            background = Image.new("RGBA", rgba.size, "white")
            return Image.alpha_composite(background, rgba).convert("RGB")
        return image.convert("RGB")

    def _encode_mllm_image(
        self,
        image: Image.Image,
        *,
        quality: int | None = None,
        lossless: bool = False,
    ) -> tuple[bytes, str]:
        buffer = io.BytesIO()
        if self.mllm_image_format == "webp":
            kwargs = {"format": "WEBP", "method": 6}
            if lossless:
                kwargs["lossless"] = True
            else:
                kwargs["quality"] = quality if quality is not None else 90
            image.save(buffer, **kwargs)
            return buffer.getvalue(), "image/webp"

        image.save(
            buffer,
            format="JPEG",
            quality=quality if quality is not None else 90,
            optimize=True,
            progressive=True,
            subsampling=0,
        )
        return buffer.getvalue(), "image/jpeg"

    def _compress_image_for_mllm(
        self,
        image_bytes: bytes,
        target_bytes: int,
    ) -> tuple[bytes, str]:
        source_mime = self._source_mime_type(image_bytes)
        if source_mime is not None and len(image_bytes) <= target_bytes:
            return image_bytes, source_mime

        cache_key = hashlib.sha256(
            image_bytes
            + str(target_bytes).encode("ascii")
            + self.mllm_image_format.encode("ascii")
        ).hexdigest()
        with self._image_cache_lock:
            cached = self._image_cache.get(cache_key)
            if cached is not None:
                self._image_cache.move_to_end(cache_key)
                return cached

            with Image.open(io.BytesIO(image_bytes)) as opened:
                image = self._rgb_image(opened)

            best: tuple[bytes, str] | None = None

            # Preserve every pixel when lossless WebP already fits the request budget.
            if self.mllm_image_format == "webp":
                encoded = self._encode_mllm_image(image, lossless=True)
                best = encoded
                if len(encoded[0]) <= target_bytes:
                    return self._cache_image(cache_key, encoded)

            # Keep the original resolution first and lower only encoding quality.
            for quality in (95, 92, 90, 88, 85, 82, 80):
                encoded = self._encode_mllm_image(image, quality=quality)
                if best is None or len(encoded[0]) < len(best[0]):
                    best = encoded
                if len(encoded[0]) <= target_bytes:
                    return self._cache_image(cache_key, encoded)

            # Only resize if encoding at the original resolution still exceeds budget.
            working = image
            while min(working.size) > 512:
                new_size = (
                    max(256, int(working.width * 0.85)),
                    max(256, int(working.height * 0.85)),
                )
                working = working.resize(new_size, Image.Resampling.LANCZOS)
                for quality in (92, 88, 85, 82, 80):
                    encoded = self._encode_mllm_image(working, quality=quality)
                    if best is None or len(encoded[0]) < len(best[0]):
                        best = encoded
                    if len(encoded[0]) <= target_bytes:
                        return self._cache_image(cache_key, encoded)

            for quality in (76, 72, 68, 60, 50, 40, 30):
                encoded = self._encode_mllm_image(working, quality=quality)
                if best is None or len(encoded[0]) < len(best[0]):
                    best = encoded
                if len(encoded[0]) <= target_bytes:
                    return self._cache_image(cache_key, encoded)

            assert best is not None
            return self._cache_image(cache_key, best)

    @classmethod
    def _uncompressed_image_for_mllm(
        cls,
        image_bytes: bytes,
    ) -> tuple[bytes, str]:
        mime_type = cls._source_mime_type(image_bytes)
        if mime_type is not None:
            return image_bytes, mime_type

        buffer = io.BytesIO()
        with Image.open(io.BytesIO(image_bytes)) as opened:
            image = cls._rgb_image(opened)
            image.save(buffer, format="PNG")
        return buffer.getvalue(), "image/png"

    def _cache_image(
        self,
        cache_key: str,
        value: tuple[bytes, str],
    ) -> tuple[bytes, str]:
        self._image_cache[cache_key] = value
        self._image_cache.move_to_end(cache_key)
        while len(self._image_cache) > self._image_cache_max_entries:
            self._image_cache.popitem(last=False)
        return value

    def _build_multimodal_content(
        self,
        text: str,
        images: list[bytes] | None,
    ) -> list[dict]:
        content = [{"type": "text", "text": text}]
        if not images:
            return content

        target_per_image = max(1, self.mllm_image_total_max_bytes // len(images))
        for image_bytes in images:
            if self.mllm_compress_images:
                transported, mime_type = self._compress_image_for_mllm(
                    image_bytes,
                    target_per_image,
                )
            else:
                transported, mime_type = self._uncompressed_image_for_mllm(
                    image_bytes
                )
            image_base64 = base64.b64encode(transported).decode("ascii")
            content.append(
                {
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:{mime_type};base64,{image_base64}"
                    },
                }
            )
        return content

    @staticmethod
    def _reasoning_content(delta) -> str:
        reasoning = getattr(delta, "reasoning_content", None)
        if reasoning:
            return reasoning
        model_extra = getattr(delta, "model_extra", None) or {}
        return model_extra.get("reasoning_content") or ""

    def _stream_chat_completion(self, messages: list[dict]) -> tuple[str, str]:
        with self._counter_lock:
            self._mllm_calls += 1
        content_parts: list[str] = []
        reasoning_parts: list[str] = []
        finish_reasons: list[str] = []
        usage: dict[str, Any] | None = None
        stream = None
        try:
            stream = self.client.chat.completions.create(
                model=self.mllm_model,
                messages=messages,
                max_tokens=self.mllm_max_tokens,
                stream=True,
                stream_options={"include_usage": True},
                **(
                    {"extra_body": self.mllm_extra_body}
                    if self.mllm_extra_body
                    else {}
                ),
            )
            for chunk in stream:
                chunk_usage = getattr(chunk, "usage", None)
                if chunk_usage is not None:
                    usage = (
                        chunk_usage.model_dump()
                        if hasattr(chunk_usage, "model_dump")
                        else dict(chunk_usage)
                    )
                if not chunk.choices:
                    continue
                choice = chunk.choices[0]
                delta = choice.delta
                if delta.content:
                    content_parts.append(delta.content)
                reasoning = self._reasoning_content(delta)
                if reasoning:
                    reasoning_parts.append(reasoning)
                if choice.finish_reason:
                    finish_reasons.append(str(choice.finish_reason))
        except BaseException as error:
            self._completion_local.metadata = {
                "content_chars": len("".join(content_parts)),
                "reasoning_chars": len("".join(reasoning_parts)),
                "finish_reasons": finish_reasons,
                "usage": usage,
                "error": f"{type(error).__name__}: {error}",
            }
            raise
        finally:
            close = getattr(stream, "close", None) if stream is not None else None
            if callable(close):
                close()
        self._completion_local.metadata = {
            "content_chars": len("".join(content_parts)),
            "reasoning_chars": len("".join(reasoning_parts)),
            "finish_reasons": finish_reasons,
            "usage": usage,
        }
        return "".join(content_parts), "".join(reasoning_parts)

    def get_last_completion_metadata(self) -> dict[str, Any]:
        """Return metadata for this thread's latest completion, without reasoning text."""
        value = getattr(self._completion_local, "metadata", {})
        return dict(value) if isinstance(value, dict) else {}

    def think(self, prompt: str, images: list[bytes] | None = None):
        content = self._build_multimodal_content(prompt, images)
        response, _ = self._stream_chat_completion(
            [{"role": "user", "content": content}]
        )
        return response

    def run_tool_loop(
        self,
        prompt: str,
        tools: list[FunctionTool],
        *,
        max_rounds: int = 8,
        require_terminal: bool = True,
    ) -> ToolLoopResult:
        """Let the configured model manage state through validated local tools."""
        if max_rounds <= 0:
            raise ValueError("max_rounds must be positive")
        if not tools:
            raise ValueError("at least one tool is required")
        by_name = {tool.name: tool for tool in tools}
        if len(by_name) != len(tools):
            raise ValueError("tool names must be unique")

        messages: list[dict] = [{"role": "user", "content": str(prompt)}]
        call_log: list[dict] = []
        last_content = ""
        for round_index in range(1, int(max_rounds) + 1):
            with self._counter_lock:
                self._mllm_calls += 1
            completion = self.client.chat.completions.create(
                model=self.mllm_model,
                messages=messages,
                tools=[tool.openai_schema() for tool in tools],
                tool_choice="auto",
                max_tokens=self.mllm_max_tokens,
                stream=False,
                **(
                    {"extra_body": getattr(self, "mllm_extra_body", {})}
                    if getattr(self, "mllm_extra_body", {})
                    else {}
                ),
            )
            if not completion.choices:
                return ToolLoopResult(
                    completed=False,
                    termination="no_choice",
                    content=last_content,
                    tool_calls=call_log,
                    messages=messages,
                )
            message = completion.choices[0].message
            last_content = str(message.content or "")
            raw_calls = list(message.tool_calls or [])
            assistant_message: dict[str, Any] = {
                "role": "assistant",
                "content": last_content,
            }
            if raw_calls:
                assistant_message["tool_calls"] = [
                    {
                        "id": str(call.id),
                        "type": "function",
                        "function": {
                            "name": str(call.function.name),
                            "arguments": str(call.function.arguments or "{}"),
                        },
                    }
                    for call in raw_calls
                ]
            messages.append(assistant_message)

            if not raw_calls:
                return ToolLoopResult(
                    completed=not require_terminal,
                    termination=(
                        "assistant_final" if not require_terminal else "missing_finish"
                    ),
                    content=last_content,
                    tool_calls=call_log,
                    messages=messages,
                )

            terminal_succeeded = False
            round_failed = False
            for call_index, call in enumerate(raw_calls):
                name = str(call.function.name)
                arguments = str(call.function.arguments or "{}")
                tool = by_name.get(name)
                if tool is None:
                    result = {"ok": False, "error": f"unknown tool: {name}"}
                elif tool.terminal and call_index != len(raw_calls) - 1:
                    result = {
                        "ok": False,
                        "error": "the terminal tool must be the final tool call",
                    }
                elif tool.terminal and round_failed:
                    result = {
                        "ok": False,
                        "error": (
                            "cannot finish in a turn where an earlier tool call failed; "
                            "inspect the error and retry"
                        ),
                    }
                else:
                    result = tool.invoke(arguments)
                entry = {
                    "round": round_index,
                    "tool_call_id": str(call.id),
                    "name": name,
                    "arguments": arguments,
                    "result": result,
                }
                call_log.append(entry)
                if not result.get("ok"):
                    round_failed = True
                messages.append(
                    {
                        "role": "tool",
                        "tool_call_id": str(call.id),
                        "content": tool_result_content(result),
                    }
                )
                if tool is not None and tool.terminal and result.get("ok"):
                    terminal_succeeded = True
            if terminal_succeeded:
                return ToolLoopResult(
                    completed=True,
                    termination="terminal_tool",
                    content=last_content,
                    tool_calls=call_log,
                    messages=messages,
                )

        return ToolLoopResult(
            completed=False,
            termination="max_rounds",
            content=last_content,
            tool_calls=call_log,
            messages=messages,
        )

    def think_with_thought(self, prompt: str, images: list[bytes] | None = None):
        content = self._build_multimodal_content(prompt, images)
        response, reasoning = self._stream_chat_completion(
            [{"role": "user", "content": content}]
        )
        return response, reasoning

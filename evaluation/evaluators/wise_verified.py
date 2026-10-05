"""WISE_Verified judge client matching the official binary Qwen protocol."""

from __future__ import annotations

import base64
import hashlib
import inspect
import json
import mimetypes
import re
import threading
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any


SCORE_PATTERN = re.compile(
    r"\*{0,2}Score\*{0,2}\s*[::]?\s*([01])\b",
    re.IGNORECASE,
)


def extract_wise_score(text: str) -> float | None:
    """Accept one complete binary answer; never truncate decimals or conflicts."""
    match = re.fullmatch(r'\s*(?:\*{0,2}Score\*{0,2}\s*:\s*)?([01])\s*', text, re.IGNORECASE)
    if match:
        return float(match.group(1))
    return None


def build_wise_messages(
    *,
    prompt: str,
    explanation: str,
    image_data_uri: str,
) -> list[dict[str, Any]]:
    """Build the same judge instructions and inputs as official vllm_eval.py."""
    if not all(isinstance(x, str) and x.strip() for x in (prompt, explanation, image_data_uri)):
        raise ValueError('WISE judge requires prompt, explanation and image')
    protocol = f"""Please evaluate this generated image for the WISE benchmark and return ONLY one binary score.

# WISE Text-to-Image Evaluation Protocol

## What WISE Is Evaluating
WISE is a knowledge-intensive text-to-image benchmark. Many prompts do not directly state the final visual answer. Instead, the model must use commonsense, cultural, scientific, spatial, or temporal knowledge to infer what should appear in the image.

Your job is not to judge whether the image is beautiful. Your job is to judge whether the generated image correctly realizes the knowledge-based meaning of the prompt and is visually usable.

## Input Fields

**PROMPT**
The original text-to-image prompt given to the image generation model. It may contain an implicit clue rather than the explicit final answer.

**EXPLANATION**
The reference interpretation used for judging. It explains the intended answer, the required knowledge reasoning chain, and the visual evidence that should appear in a correct image. Treat EXPLANATION as the ground-truth judging guide.

For example:
- If PROMPT says "the round pastry commonly shared during Mid-Autumn Festival family gatherings", EXPLANATION may specify mooncakes. A correct image should show mooncakes, not just any festival food.
- If PROMPT says "a plant kept for many days beside a bright one-sided window", EXPLANATION may specify phototropism. A correct image should show the plant bending toward the light source.
- If PROMPT says "a street in New York when it is midnight in Beijing", EXPLANATION may specify the corresponding local time and expected lighting/activity. A correct image should reflect that inferred local time, not simply show Beijing or generic night.

## How To Judge

Evaluate the image using these checks:
1. Does the image contain the main objects or scene required by the PROMPT?
2. Does it satisfy the intended knowledge-based answer described in the EXPLANATION?
3. Are important relations correct, such as spatial layout, temporal state, physical effect, biological behavior, cultural object, or scientific phenomenon?
4. Is the image visually usable for judging, without obvious collapse, severe deformation, unreadable main objects, or major artifacts?

## Binary Score

**Score: 1**
Give 1 only when both conditions are met:
- The image is semantically correct according to both PROMPT and EXPLANATION.
- The image has no obvious generation failure that prevents reliable judging.

Minor aesthetic weakness, ordinary composition, non-photorealistic style, or lack of artistic beauty should not by itself cause rejection if the semantic target is correct and the image is clear.

**Score: 0**
Give 0 if any of the following applies:
- The image misses the intended answer in EXPLANATION.
- The image only follows surface words in PROMPT but fails the required knowledge inference.
- Key objects, attributes, states, behaviors, or relations are missing or wrong.
- The image contradicts the prompt or explanation.
- The main visual evidence is ambiguous enough that a human judge could not confidently verify correctness.
- The image has obvious visual collapse, severe deformation, garbled main objects, impossible structure, or artifacts that interfere with evaluation.

If there is serious doubt, return 0.

## Output Format

Return exactly one line and nothing else:

Score: 0

or

Score: 1

---

PROMPT: "{prompt}"
EXPLANATION: "{explanation}"

Return only `Score: 0` or `Score: 1`."""
    return [
        {
            "role": "system",
            "content": [
                {
                    "type": "text",
                    "text": (
                        "You are a professional text-to-image quality auditor. "
                        "Evaluate the image strictly according to the protocol."
                    ),
                }
            ],
        },
        {
            "role": "user",
            "content": [
                {"type": "text", "text": protocol},
                {
                    "type": "image_url",
                    "image_url": {"url": image_data_uri},
                },
            ],
        },
    ]


class WiseVerifiedRuntime:
    """Call a local OpenAI-compatible Qwen3.5 WISE judge."""

    def __init__(
        self,
        *,
        api_base: str = "http://127.0.0.1:8205/v1",
        model: str = "Qwen3.5-35B-A3B",
        api_key: str = "EMPTY",
        timeout_seconds: float = 300.0,
        max_extract_attempts: int = 3,
    ) -> None:
        self.api_base = api_base.rstrip("/")
        self.model = model
        self.api_key = api_key
        self.timeout_seconds = float(timeout_seconds)
        self.max_extract_attempts = int(max_extract_attempts)
        self._request_lock = threading.Lock()
        self.protocol_hash = hashlib.sha256((inspect.getsource(build_wise_messages) + inspect.getsource(extract_wise_score)).encode()).hexdigest()
        if self.timeout_seconds <= 0:
            raise ValueError("timeout_seconds must be positive")
        if self.max_extract_attempts <= 0:
            raise ValueError("max_extract_attempts must be positive")

    def _open_json(
        self,
        request: urllib.request.Request,
        *,
        timeout: float,
    ) -> dict[str, Any]:
        opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
        try:
            with opener.open(request, timeout=timeout) as response:
                value = json.loads(response.read().decode("utf-8"))
        except urllib.error.HTTPError as error:
            detail = error.read(1024).decode("utf-8", errors="replace")
            raise RuntimeError(
                f"WISE judge returned HTTP {error.code}: {detail}"
            ) from error
        except (urllib.error.URLError, TimeoutError, json.JSONDecodeError) as error:
            raise RuntimeError(f"WISE judge request failed: {error}") from error
        if not isinstance(value, dict):
            raise RuntimeError("WISE judge returned non-object JSON")
        return value

    def check_ready(self, *, timeout_seconds: float = 5.0) -> None:
        value = self._open_json(
            urllib.request.Request(
                f"{self.api_base}/models",
                headers={"Authorization": f"Bearer {self.api_key}"},
            ),
            timeout=timeout_seconds,
        )
        if not isinstance(value.get("data"), list):
            raise RuntimeError("WISE judge /models response has no model list")
        if self.model not in {item.get('id') for item in value['data'] if isinstance(item, dict)}:
            raise RuntimeError(f'Expected judge model {self.model} is not served')

    @staticmethod
    def _image_data_uri(image_path: Path) -> str:
        mime = mimetypes.guess_type(image_path.name)[0] or "image/png"
        encoded = base64.b64encode(image_path.read_bytes()).decode("ascii")
        return f"data:{mime};base64,{encoded}"

    def evaluate_image(
        self,
        image_path: str | Path,
        *,
        prompt: str,
        explanation: str,
    ) -> dict[str, Any]:
        resolved = Path(image_path).expanduser().resolve()
        if not resolved.is_file():
            raise FileNotFoundError(resolved)
        messages = build_wise_messages(
            prompt=prompt,
            explanation=explanation,
            image_data_uri=self._image_data_uri(resolved),
        )
        body = json.dumps(
            {
                "model": self.model,
                "messages": messages,
                "temperature": 0.0,
                "max_tokens": 500,
                "chat_template_kwargs": {"enable_thinking": False},
            },
            ensure_ascii=False,
        ).encode("utf-8")
        request = urllib.request.Request(
            f"{self.api_base}/chat/completions",
            data=body,
            headers={
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
            },
            method="POST",
        )
        last_output = ""
        last_error: BaseException | None = None
        # Single-flight keeps the BF16 judge inside its one-sequence VRAM budget.
        with self._request_lock:
            for _ in range(self.max_extract_attempts):
                try:
                    value = self._open_json(
                        request,
                        timeout=self.timeout_seconds,
                    )
                    choices = value.get("choices")
                    if not isinstance(choices, list) or not choices:
                        raise RuntimeError("WISE judge returned no choices")
                    message = choices[0].get("message") or {}
                    last_output = str(message.get("content") or "").strip()
                    last_output = re.sub(
                        r"<think>.*?</think>\s*",
                        "",
                        last_output,
                        flags=re.DOTALL,
                    )
                    last_output = re.sub(r"</think>\s*", "", last_output).strip()
                    score = extract_wise_score(last_output)
                    if score is not None:
                        return {
                            "correct": bool(score),
                            "score": score,
                            "judge_output": last_output,
                            "judge_model": self.model,
                            "thinking_enabled": False,
                            "protocol_hash": self.protocol_hash,
                            "image_sha256": hashlib.sha256(resolved.read_bytes()).hexdigest(),
                            "judge_request": {'prompt': messages[1]['content'][0]['text'],
                                              'body_sha256': hashlib.sha256(body).hexdigest(),
                                              'model': self.model, 'temperature': 0.0, 'max_tokens': 500},
                        }
                    last_error = RuntimeError(
                        f"could not extract WISE score from: {last_output!r}"
                    )
                except Exception as error:
                    last_error = error
        raise RuntimeError(
            f"WISE judge failed after {self.max_extract_attempts} attempts"
        ) from last_error


__all__ = [
    "WiseVerifiedRuntime",
    "build_wise_messages",
    "extract_wise_score",
]

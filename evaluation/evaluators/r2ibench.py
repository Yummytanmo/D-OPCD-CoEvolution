"""R2I-Bench weighted checklist feedback through a judge API."""

from __future__ import annotations

import hashlib
import json
import math
import re
import threading
import urllib.request
from pathlib import Path
from typing import Any

from evaluation.evaluators.wise_verified import WiseVerifiedRuntime


OFFICIAL_PROMPT = """
# Text-to-Image Quality Evaluation Protocol 
# ## System Instruction 
You are an AI quality auditor for text-to-image generation. Answer these questions with ABSOLUTE RUTHLESSNESS. 
Only images meeting the HIGHEST standards should receive top scores.

## Task Overview
The image is prompt by the prompt: 
[PROMPT]

## Question List
[QUESTION LIST]

## Output Format
You can analyze in your output, but ensure the final line of your output is formatted as follows:

## Important Enforcement
- As long as the answer to the question of other confirmed valid permutation combinations has a score not equal to 1, the scores for the two questions of whether there are repeated permutations and whether there are invalid permutations will both be 0.

```json
{
    "id": score,
    ...
}
``` 
]
"""


def r2i_protocol_hash() -> str:
    # Pin executable adapter as well as source prompt; template-only hashes
    # cannot distinguish a broken renderer from a repaired one.
    return hashlib.sha256(Path(__file__).read_bytes() + official_prompt_template().encode()).hexdigest()


def validate_r2i_checklist(checklist: Any) -> None:
    if not isinstance(checklist, list) or not checklist:
        raise ValueError('a non-empty official R2I checklist is required')
    for item in checklist:
        if (not isinstance(item, dict) or isinstance(item.get('id'), bool)
                or not isinstance(item.get('id'), (str, int)) or not str(item['id']).strip()
                or not isinstance(item.get('question'), str) or not item['question'].strip()
                or not isinstance(item.get('criteria'), str) or not item['criteria'].strip()
                or isinstance(item.get('weight'), bool) or not isinstance(item.get('weight'), (int, float))
                or not math.isfinite(float(item['weight'])) or item['weight'] < 0):
            raise ValueError('R2I checklist requires id, question, criteria and finite non-negative weight')
    total = sum(float(item['weight']) for item in checklist)
    if not math.isfinite(total) or total <= 0:
        raise ValueError('R2I checklist requires positive finite total weight')


def unique_checklist_ids(checklist: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Disambiguate repeated source IDs without dropping or changing questions."""
    reserved = {str(item['id']) for item in checklist}
    seen: set[str] = set()
    result = []
    candidate = 1
    for item in checklist:
        identifier = str(item['id'])
        copied = dict(item)
        if identifier in seen:
            while str(candidate) in reserved:
                candidate += 1
            copied['source_id'] = item['id']
            copied['id'] = str(candidate)
            reserved.add(str(candidate))
        seen.add(identifier)
        result.append(copied)
    return result


def official_prompt_template() -> str:
    return OFFICIAL_PROMPT


def build_r2ibench_messages(
    *, prompt: str, checklist: list[dict[str, Any]], image_data_uri: str
) -> list[dict[str, Any]]:
    validate_r2i_checklist(checklist)
    if not isinstance(prompt, str) or not prompt.strip():
        raise ValueError('R2I prompt must not be empty')
    questions = "\n".join(
        f"{item['id']}. {item['question']} \nCriteria: {item['criteria']}"
        for item in checklist
    )
    template = official_prompt_template()
    slots = re.findall(r'\[(?:PROMPT|QUESTION LIST|QUESTION_LIST)\]', template)
    if slots.count('[PROMPT]') != 1 or sum(s != '[PROMPT]' for s in slots) != 1:
        raise ValueError('R2I template requires exactly one prompt and one question-list placeholder')
    # Substitute once so placeholder-like text inside data is never rewritten.
    protocol = re.sub(r'\[(?:PROMPT|QUESTION LIST|QUESTION_LIST)\]',
                      lambda m: prompt if m.group() == '[PROMPT]' else questions, template)
    if prompt not in protocol or any(item['question'] not in protocol or item['criteria'] not in protocol for item in checklist):
        raise ValueError('R2I rendered request lost required judging inputs')
    required_keys = ", ".join(json.dumps(str(item["id"])) for item in checklist)
    protocol += f"""

Concrete output enforcement for this sample:
- The example key \"id\" above is a placeholder and is not a valid answer key.
- Return one JSON object with exactly these keys: {required_keys}.
- Every value must be a numeric value from 0.0 through 1.0 according to that question's criteria.
- Do not omit a key and do not add any other key.
"""
    return [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": protocol},
                {"type": "image_url", "image_url": {"url": image_data_uri}},
            ],
        }
    ]


def extract_r2ibench_scores(text: str, expected_ids: list[str]) -> dict[str, float]:
    if not expected_ids or len(set(expected_ids)) != len(expected_ids):
        raise ValueError('Expected checklist IDs must be nonempty and unique')
    def unique_object(pairs):
        result = {}
        for key, value in pairs:
            if key in result:
                raise ValueError(f'duplicate JSON score key: {key}')
            result[key] = value
        return result
    fenced = re.search(r"```json\s*(\{.*?\})\s*```", text, flags=re.DOTALL | re.IGNORECASE)
    candidates = [fenced.group(1)] if fenced else []
    candidates.append(text)
    value: Any = None
    decoder = json.JSONDecoder(object_pairs_hook=unique_object)
    for candidate in candidates:
        try:
            value = json.loads(candidate, object_pairs_hook=unique_object)
        except json.JSONDecodeError:
            for index, character in enumerate(candidate):
                if character != "{":
                    continue
                try:
                    value, _ = decoder.raw_decode(candidate[index:])
                    break
                except json.JSONDecodeError:
                    continue
        if isinstance(value, dict):
            break
    if not isinstance(value, dict):
        raise ValueError("judge output does not contain a JSON object")
    normalized = {str(key): raw for key, raw in value.items()}
    if set(normalized) != set(expected_ids):
        missing = sorted(set(expected_ids) - set(normalized))
        extra = sorted(set(normalized) - set(expected_ids))
        raise ValueError(f"judge output checklist coverage mismatch missing={missing} extra={extra}")
    scores: dict[str, float] = {}
    for question_id in expected_ids:
        raw = normalized[question_id]
        if isinstance(raw, bool) or not isinstance(raw, (int, float)):
            raise ValueError(f"non-numeric value for checklist id {question_id}")
        score = float(raw)
        if not 0.0 <= score <= 1.0:
            raise ValueError(f"out-of-range value for checklist id {question_id}")
        scores[question_id] = score
    return scores


class R2IBenchRuntime(WiseVerifiedRuntime):
    """Evaluate one image with every official question for its R2I sample."""

    def __init__(
        self,
        *,
        api_base: str = "http://127.0.0.1:8208/v1",
        model: str = "Qwen3-VL-32B-Instruct",
        cache_dir: str | Path = "runs/cache/r2ibench-judge",
        max_concurrent_requests: int = 1,
        **kwargs: Any,
    ) -> None:
        super().__init__(api_base=api_base, model=model, **kwargs)
        self.max_concurrent_requests = int(max_concurrent_requests)
        if self.max_concurrent_requests < 1:
            raise ValueError("max_concurrent_requests must be positive")
        self._request_slots = threading.BoundedSemaphore(self.max_concurrent_requests)
        self._cache_guard = threading.Lock()
        self._cache_locks: dict[str, threading.Lock] = {}
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.protocol_hash = r2i_protocol_hash()

    def evaluate_image(
        self,
        image_path: str | Path,
        *,
        prompt: str,
        checklist: list[dict[str, Any]],
    ) -> dict[str, Any]:
        validate_r2i_checklist(checklist)

        checklist = unique_checklist_ids(checklist)
        expected_ids = [str(item['id']) for item in checklist]

        path = Path(image_path)
        image_hash = hashlib.sha256(path.read_bytes()).hexdigest()
        key_payload = {
            "image_sha256": image_hash,
            "prompt": prompt,
            "checklist": checklist,
            "model": self.model,
            "protocol_hash": self.protocol_hash,
            "temperature": 0.1,
            "thinking_enabled": False,
            "parser_version": 1,
        }
        key = hashlib.sha256(
            json.dumps(key_payload, ensure_ascii=False, sort_keys=True).encode("utf-8")
        ).hexdigest()
        cache_path = self.cache_dir / f"{key}.json"
        with self._cache_guard:
            cache_lock = self._cache_locks.setdefault(key, threading.Lock())
        with cache_lock:
            if cache_path.is_file():
                return json.loads(cache_path.read_text(encoding="utf-8"))
            messages = build_r2ibench_messages(
                prompt=prompt,
                checklist=checklist,
                image_data_uri=self._image_data_uri(path),
            )
            last_error: Exception | None = None
            with self._request_slots:
                for max_tokens in (2048, 4096, 8192):
                    body = {
                        "model": self.model,
                        "messages": messages,
                        "temperature": 0.1,
                        "max_tokens": max_tokens,
                        "chat_template_kwargs": {"enable_thinking": False},
                    }
                    try:
                        response = self._open_json(
                            urllib.request.Request(
                                f"{self.api_base}/chat/completions",
                                data=json.dumps(body, ensure_ascii=False).encode("utf-8"),
                                headers={
                                    "Content-Type": "application/json",
                                    "Authorization": f"Bearer {self.api_key}",
                                },
                                method="POST",
                            ),
                            timeout=self.timeout_seconds,
                        )
                        choice = response["choices"][0]
                        if choice.get("finish_reason") == "length":
                            raise ValueError("judge output reached the token limit")
                        output = str((choice.get("message") or {}).get("content") or "")
                        scores = extract_r2ibench_scores(output, expected_ids)
                    except (KeyError, IndexError, TypeError, ValueError, RuntimeError) as exc:
                        last_error = exc
                        continue
                    weighted = sum(
                        float(item["weight"]) * scores[str(item["id"])]
                        for item in checklist
                    )
                    total_weight = sum(float(item["weight"]) for item in checklist)
                    raw_score = weighted / total_weight
                    result = {
                        "score": round(raw_score, 2),
                        "raw_score": raw_score,
                        "question_coverage": 1.0,
                        "valid_questions": len(checklist),
                        "total_questions": len(checklist),
                        "checklist_results": [
                            {
                                "id": str(item["id"]),
                                **({'source_id': item['source_id']} if 'source_id' in item else {}),
                                "question": item["question"],
                                "weight": float(item["weight"]),
                                "score": scores[str(item["id"])],
                            }
                            for item in checklist
                        ],
                        "judge_model": self.model,
                        "judge_output": output,
                        "judge_usage": response.get("usage"),
                        "thinking_enabled": False,
                        "temperature": 0.1,
                        "protocol_hash": self.protocol_hash,
                        "image_sha256": image_hash,
                        "judge_request": {
                            "prompt": messages[0]['content'][0]['text'],
                            "model": self.model, "temperature": body['temperature'],
                            "max_tokens": body['max_tokens'],
                            "thinking_enabled": False,
                            "body_sha256": hashlib.sha256(json.dumps(body, ensure_ascii=False).encode()).hexdigest(),
                        },
                    }
                    temporary = cache_path.with_suffix(".tmp")
                    temporary.write_text(json.dumps(result, ensure_ascii=False), encoding="utf-8")
                    temporary.replace(cache_path)
                    return result
        raise RuntimeError(f"R2I-Bench judge failed after retry schedule: {last_error}")

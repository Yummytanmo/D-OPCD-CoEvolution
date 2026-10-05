#!/usr/bin/env python3
"""Evaluate generated images with the R2I-Bench weighted checklist protocol."""

from __future__ import annotations

import argparse
import base64
import concurrent.futures
import hashlib
import json
import math
import os
from pathlib import Path
import re
import sys
import threading
import time

from openai import OpenAI
from PIL import Image

if not __package__:
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from evaluation.benchmarks.common import ROOT, clean_env, run_logged, task_main


OFFICIAL_PROMPT = """
# Text-to-Image Quality Evaluation Protocol
## System Instruction
You are an AI quality auditor for text-to-image generation. Answer these questions with ABSOLUTE RUTHLESSNESS.
Only images meeting the HIGHEST standards should receive top scores.

## Task Overview
The image is prompt by the prompt:
[PROMPT]

## Question List
[QUESTION_LIST]

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
""".strip()

OFFICIAL_REPOSITORY = "https://github.com/PLUM-Lab/R2I-Bench"
OFFICIAL_COMMIT = "874e8a1b20a246fe390152c767534a8abcfeae82"
WRITE_LOCK = threading.Lock()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def atomic_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def parse_scores(text: str, expected_ids: list[str]) -> dict[str, float]:
    candidates = re.findall(r"```(?:json)?\s*(\{.*?\})\s*```", text, flags=re.DOTALL | re.IGNORECASE)
    if not candidates:
        left, right = text.find("{"), text.rfind("}")
        if left >= 0 and right > left:
            candidates = [text[left : right + 1]]
    if not candidates:
        raise ValueError("Judge response contains no JSON object")
    parsed = json.loads(re.sub(r"[\x00-\x1f\x7f]", "", candidates[-1]))
    if not isinstance(parsed, dict):
        raise ValueError("Judge JSON is not an object")
    missing = [question_id for question_id in expected_ids if question_id not in parsed]
    if missing:
        raise ValueError("Judge response is missing checklist IDs: " + ",".join(missing))
    result: dict[str, float] = {}
    for question_id in expected_ids:
        value = float(parsed[question_id])
        if not math.isfinite(value) or not 0 <= value <= 1:
            raise ValueError(f"Invalid score for checklist ID {question_id}")
        result[question_id] = value
    return result


def prompt_for(row: dict) -> str:
    questions = "\n".join(
        f"{question['id']}. {question['question']} \nCriteria: {question['criteria']}"
        for question in row["checklist"]
    )
    return OFFICIAL_PROMPT.replace("[PROMPT]", row["prompt"]).replace("[QUESTION_LIST]", questions)


def image_path(image_dir: Path, index: int) -> Path:
    return image_dir / f"{index:05d}.png"


def validate_record(record: dict, identity: dict, expected_ids: list[str]) -> dict:
    if record.get("identity") != identity:
        raise ValueError("Existing R2I score record identity mismatch")
    scores = record.get("checklist_scores")
    if not isinstance(scores, dict) or set(scores) != set(expected_ids):
        raise ValueError("Existing R2I score record coverage mismatch")
    value = float(record.get("score"))
    if not math.isfinite(value) or not 0 <= value <= 1:
        raise ValueError("Existing R2I score is invalid")
    return record


def evaluate_one(
    index: int,
    row: dict,
    *,
    image_dir: Path,
    records_dir: Path,
    api_base: str,
    model: str,
    max_tokens: int | None,
    reasoning_effort: str | None,
    retries: int,
    manifest_sha256: str,
    completed: list[int],
    progress_path: Path,
    expected_count: int,
) -> dict:
    path = image_path(image_dir, index)
    if not path.is_file():
        raise FileNotFoundError(path)
    Image.open(path).verify()
    checklist = row.get("checklist")
    if not isinstance(checklist, list) or not checklist:
        raise ValueError(f"Missing checklist for row {index}")
    question_ids = [str(question["id"]) for question in checklist]
    if len(question_ids) != len(set(question_ids)):
        raise ValueError(f"Duplicate checklist ID for row {index}")
    identity = {
        "manifest_sha256": manifest_sha256,
        "index": index,
        "sample_id": row["sample_id"],
        "image_sha256": sha256(path),
        "prompt_sha256": hashlib.sha256(row["prompt"].encode("utf-8")).hexdigest(),
        "judge_api_base": api_base,
        "judge_model": model,
        "reasoning_effort": reasoning_effort,
        "max_tokens": max_tokens,
        "official_prompt_sha256": hashlib.sha256(OFFICIAL_PROMPT.encode("utf-8")).hexdigest(),
        "official_commit": OFFICIAL_COMMIT,
    }
    record_path = records_dir / f"{index:05d}.json"
    if record_path.exists():
        record = validate_record(json.loads(record_path.read_text(encoding="utf-8")), identity, question_ids)
    else:
        api_key = os.environ.get("R2I_JUDGE_API_KEY")
        if not api_key:
            raise RuntimeError("R2I_JUDGE_API_KEY is unavailable")
        encoded = base64.b64encode(path.read_bytes()).decode("ascii")
        request_prompt = prompt_for(row)
        last_error: BaseException | None = None
        response_text = ""
        scores: dict[str, float] | None = None
        for attempt in range(1, retries + 1):
            try:
                client = OpenAI(api_key=api_key, base_url=api_base, timeout=180, max_retries=0)
                request = {
                    "model": model,
                    "temperature": 0.1,
                    "messages": [
                        {
                            "role": "user",
                            "content": [
                                {"type": "text", "text": request_prompt},
                                {
                                    "type": "image_url",
                                    "image_url": {"url": f"data:image/png;base64,{encoded}"},
                                },
                            ],
                        }
                    ],
                }
                # The upstream GPT-4o evaluator sets neither reasoning effort nor
                # an output-token cap. Keep both optional so that exact request
                # shape can be used through an OpenAI-compatible endpoint.
                if max_tokens is not None:
                    request["max_tokens"] = max_tokens
                if reasoning_effort is not None:
                    request["extra_body"] = {"reasoning_effort": reasoning_effort}
                response = client.chat.completions.create(**request)
                response_text = response.choices[0].message.content or ""
                scores = parse_scores(response_text, question_ids)
                break
            except BaseException as exc:
                last_error = exc
                if attempt < retries:
                    time.sleep(2 * attempt)
        if scores is None:
            raise RuntimeError(f"R2I judge failed after {retries} attempts: {type(last_error).__name__}") from last_error
        numerator = sum(float(question["weight"]) * scores[str(question["id"])] for question in checklist)
        denominator = sum(float(question["weight"]) for question in checklist)
        if not math.isfinite(denominator) or denominator <= 0:
            raise ValueError(f"Invalid checklist weights for row {index}")
        record = {
            "identity": identity,
            "category": row["category"],
            "subcategory": row["subcategory"],
            "checklist_scores": scores,
            "score": round(numerator / denominator, 2),
            "response_text": response_text,
        }
        atomic_json(record_path, record)
    with WRITE_LOCK:
        completed[0] += 1
        atomic_json(
            progress_path,
            {"status": "running", "completed": completed[0], "expected": expected_count},
        )
    return record


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--manifest-sha256", required=True)
    parser.add_argument("--image-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--expected-count", type=int, required=True)
    parser.add_argument("--api-base", default=os.environ.get("R2I_JUDGE_API_BASE"))
    parser.add_argument("--model", default="gpt-4o")
    parser.add_argument(
        "--reasoning-effort",
        default="none",
        help="Use 'none' to reproduce the upstream GPT-4o request.",
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=0,
        help="Use 0 to leave max_tokens unset as in the upstream evaluator.",
    )
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--retries", type=int, default=3)
    parser.add_argument("--limit", type=int)
    args = parser.parse_args()
    if not args.api_base:
        parser.error("Set --api-base or R2I_JUDGE_API_BASE")
    if sha256(args.manifest) != args.manifest_sha256:
        raise ValueError("Manifest SHA256 mismatch")
    all_rows = [json.loads(line) for line in args.manifest.read_text(encoding="utf-8").splitlines() if line.strip()]
    if len(all_rows) != args.expected_count:
        raise ValueError(f"Manifest count={len(all_rows)}, expected={args.expected_count}")
    if any(row.get("benchmark") != "r2ibench" for row in all_rows):
        raise ValueError("Non-R2I row in manifest")
    rows = all_rows[: args.limit] if args.limit is not None else all_rows
    expected = len(rows)
    if expected <= 0 or args.workers <= 0 or args.retries <= 0 or args.max_tokens < 0:
        raise ValueError("Expected count, workers, and retries must be positive")
    reasoning_effort = None if args.reasoning_effort.lower() == "none" else args.reasoning_effort
    max_tokens = None if args.max_tokens == 0 else args.max_tokens
    records_dir = args.output_dir / "records"
    progress_path = args.output_dir / "progress.json"
    completed = [0]
    kwargs = {
        "image_dir": args.image_dir,
        "records_dir": records_dir,
        "api_base": args.api_base,
        "model": args.model,
        "max_tokens": max_tokens,
        "reasoning_effort": reasoning_effort,
        "retries": args.retries,
        "manifest_sha256": args.manifest_sha256,
        "completed": completed,
        "progress_path": progress_path,
        "expected_count": expected,
    }
    with concurrent.futures.ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = [pool.submit(evaluate_one, index, row, **kwargs) for index, row in enumerate(rows)]
        records = [future.result() for future in futures]
    scores_path = args.output_dir / "scores.jsonl"
    scores_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = scores_path.with_suffix(".jsonl.tmp")
    temporary.write_text("".join(json.dumps(record, ensure_ascii=False) + "\n" for record in records), encoding="utf-8")
    temporary.replace(scores_path)
    categories: dict[str, list[float]] = {}
    for record in records:
        categories.setdefault(record["category"], []).append(float(record["score"]))
    metrics = {
        "status": "completed",
        "benchmark": "r2ibench",
        "sample_count": expected,
        "r2i_score": sum(float(record["score"]) for record in records) / expected,
        "category_scores": {
            category: sum(values) / len(values) for category, values in sorted(categories.items())
        },
        "judge": {
            "api_base": args.api_base,
            "model": args.model,
            "reasoning_effort": reasoning_effort,
            "max_tokens": max_tokens,
            "temperature": 0.1,
            "official_gpt4o_request_shape": (
                args.model == "gpt-4o" and reasoning_effort is None and max_tokens is None
            ),
            "comparability": "Compare only with the same judge model and request protocol.",
        },
        "protocol": {
            "official_repository": OFFICIAL_REPOSITORY,
            "official_commit": OFFICIAL_COMMIT,
            "official_prompt_sha256": hashlib.sha256(OFFICIAL_PROMPT.encode("utf-8")).hexdigest(),
            "per_instance_rounding": 2,
        },
    }
    atomic_json(args.output_dir / "metrics.json", metrics)
    atomic_json(progress_path, {"status": "completed", "completed": expected, "expected": expected})
    print(json.dumps(metrics, ensure_ascii=False))
    return 0


def run(task: dict) -> None:
    dest = Path(task["evaluation_dir"])
    judge = task["judge"]
    command = [
        sys.executable, str(Path(__file__).resolve()),
        "--manifest", str(task["manifest"]),
        "--manifest-sha256", str(task["manifest_sha256"]),
        "--image-dir", str(dest / "images"),
        "--output-dir", str(dest / "scores"),
        "--expected-count", str(task["expected_count"]),
        "--api-base", str(judge["api_base"]),
        "--model", str(judge["model"]),
        "--workers", str(judge.get("workers", 1)),
        "--retries", str(judge.get("retries", 3)),
        "--max-tokens", str(judge.get("max_tokens", 0)),
    ]
    if judge.get("reasoning_effort"):
        command += ["--reasoning-effort", str(judge["reasoning_effort"])]
    run_logged(command, cwd=ROOT, log=dest / "metadata" / "evaluator.log",
               env=clean_env(keep_proxy=True))


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "--task":
        task_main("r2ibench", run)
    else:
        raise SystemExit(main())

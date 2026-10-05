"""GenEval scoring with an external upstream evaluator."""

from __future__ import annotations

import json
from pathlib import Path
import sys

if not __package__:
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from evaluation.benchmarks.common import (ROOT, clean_env, evaluator_root, project_path, run_logged,
                                          stage_agent_inputs, task_main, write_json)


def summarize(raw: Path, expected_count: int) -> dict:
    rows = [json.loads(line) for line in raw.read_text(encoding="utf-8").splitlines()
            if line.strip()]
    if len(rows) != expected_count:
        raise ValueError(f"GenEval rows={len(rows)}, expected={expected_count}")
    if not all(isinstance(row.get("correct"), bool) for row in rows):
        raise ValueError("GenEval results contain a non-boolean correct value")
    tags: dict[str, list[bool]] = {}
    for row in rows:
        tags.setdefault(str(row["tag"]), []).append(row["correct"])
    task_scores = {tag: sum(values) / len(values) for tag, values in tags.items()}
    return {
        "schema_version": 1, "status": "completed", "benchmark": "geneval",
        "sample_count": len(rows),
        "image_accuracy": sum(row["correct"] for row in rows) / len(rows),
        "overall_score": sum(task_scores.values()) / len(task_scores),
        "task_scores": task_scores,
    }


def run(task: dict) -> None:
    if task.get("agent_run_dir"):
        stage_agent_inputs(task)
    dest = Path(task["evaluation_dir"])
    upstream = evaluator_root(task, "geneval")
    inputs = dest / "inputs" / "images"
    scores = dest / "scores"
    scores.mkdir(parents=True, exist_ok=True)
    raw = scores / "results.jsonl"
    log = dest / "metadata" / "evaluator.log"
    env = clean_env()
    run_logged([
        sys.executable, str(upstream / "evaluation/evaluate_images.py"), str(inputs),
        "--outfile", str(raw),
        "--model-path", str(project_path(task["detector_model"])),
        "--clip-cache-dir", str(project_path(task["clip_cache"])),
    ], cwd=upstream, log=log, env=env)
    write_json(scores / "metrics.json", summarize(raw, task["expected_count"]))


if __name__ == "__main__":
    task_main("geneval", run)

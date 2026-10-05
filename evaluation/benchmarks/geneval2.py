"""GenEval2 Soft-TIFA-AM scoring with its isolated upstream environment."""

from __future__ import annotations

import json
import math
from pathlib import Path
import sys

if not __package__:
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from evaluation.benchmarks.common import (ROOT, clean_env, evaluator_root, project_path, run_logged,
                                          stage_agent_inputs, task_main, write_json)


GENEVAL2_PROBABILITY_EPSILON = 1e-6


def summarize(raw: Path, benchmark_data: Path, expected_count: int) -> dict:
    scores = json.loads(raw.read_text(encoding="utf-8"))
    benchmark_rows = [json.loads(line) for line in benchmark_data.read_text(encoding="utf-8").splitlines()
                      if line.strip()]
    if len(scores) != expected_count or len(benchmark_rows) != expected_count:
        raise ValueError(
            f"GenEval2 score/benchmark counts={len(scores)}/{len(benchmark_rows)}, "
            f"expected={expected_count}"
        )
    skill_values: dict[str, list[float]] = {}
    prompt_am: list[float] = []
    prompt_gm: list[float] = []
    clipped_atom_count = 0
    maximum_excess = 0.0
    for score_list, row in zip(scores, benchmark_rows, strict=True):
        skills = row["skills"]
        if len(score_list) != len(skills) or not score_list:
            raise ValueError("GenEval2 atom score/skill cardinality mismatch")
        numeric = [float(value) for value in score_list]
        if not all(
            math.isfinite(value) and 0 <= value <= 1 + GENEVAL2_PROBABILITY_EPSILON
            for value in numeric
        ):
            raise ValueError("GenEval2 atom score is outside [0, 1]")
        # Float32 probabilities can exceed one by a few ulps.
        clipped_atom_count += sum(value > 1 for value in numeric)
        maximum_excess = max(maximum_excess, max(numeric) - 1)
        numeric = [min(value, 1.0) for value in numeric]
        prompt_am.append(sum(numeric) / len(numeric))
        prompt_gm.append(math.prod(numeric) ** (1 / len(numeric)))
        for skill, value in zip(skills, numeric, strict=True):
            skill_values.setdefault(str(skill), []).append(value)
    return {
        "schema_version": 1, "status": "completed", "benchmark": "geneval2",
        "sample_count": len(scores),
        "soft_tifa_am": 100 * sum(prompt_am) / len(prompt_am),
        "soft_tifa_gm": 100 * sum(prompt_gm) / len(prompt_gm),
        "skill_scores": {
            skill: 100 * sum(values) / len(values)
            for skill, values in sorted(skill_values.items())
        },
        "numerical_validation": {
            "upper_bound_epsilon": GENEVAL2_PROBABILITY_EPSILON,
            "clipped_atom_count": clipped_atom_count,
            "maximum_excess_above_one": maximum_excess,
            "raw_scores_modified": False,
        },
    }


def run(task: dict) -> None:
    if task.get("agent_run_dir"):
        stage_agent_inputs(task)
    dest = Path(task["evaluation_dir"])
    upstream = evaluator_root(task, "geneval2")
    inputs = dest / "inputs"
    scores = dest / "scores"
    scores.mkdir(parents=True, exist_ok=True)
    raw = scores / "soft-tifa-atom-scores.json"
    data = inputs / "benchmark_data.jsonl"
    log = dest / "metadata" / "evaluator.log"
    env = clean_env()
    run_logged([
        sys.executable, str(upstream / "evaluation.py"),
        "--benchmark_data", str(data),
        "--image_filepath_data", str(inputs / "image_paths.json"),
        "--method", "soft_tifa_am",
        "--model_path", str(project_path(task["judge_model"])),
        "--output_file", str(raw),
    ], cwd=upstream, log=log, env=env)
    write_json(scores / "metrics.json", summarize(raw, data, task["expected_count"]))


if __name__ == "__main__":
    task_main("geneval2", run)

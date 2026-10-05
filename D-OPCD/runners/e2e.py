"""Run one D-OPCD train → validation selection → held-out test attempt.

``python -m runners.e2e --config ...`` preflights without launching. Add
``--execute`` for a new automatically numbered attempt, or ``--resume RUN_ID``
to continue the exact same frozen attempt. Never writes Markdown reports.
"""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import json
import math
import os
from pathlib import Path
import sys
import traceback

if not __package__:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from runners import generate
from runners.common import (RESULTS, ROOT, RUNS, digest, executable, now, path,
                            read_json, read_jsonl, reserve_run, run_logged,
                            validate_config, write_json)
from train import dopcd


METRICS = {
    "geneval": ("overall_score", 1.0, "results.jsonl"),
    "geneval2": ("soft_tifa_am", 100.0, "soft-tifa-atom-scores.json"),
    "wise": ("wiscore", 1.0, "scores.jsonl"),
    "r2ibench": ("r2i_score", 1.0, "scores.jsonl"),
}



def checkpoint(run_root: Path, step: int) -> Path:
    return run_root / "training" / f"checkpoint-{step}" / "pytorch_lora_weights.safetensors"


def checkpoint_complete(run_root: Path, step: int) -> bool:
    weights = checkpoint(run_root, step)
    marker = weights.parent / "checkpoint_complete.json"
    return (weights.is_file() and marker.is_file() and
            read_json(marker).get("status") == "complete" and
            read_json(marker).get("global_step") == step)


def evaluation_dir(run_root: Path, step: int, split: str) -> Path:
    return run_root / "evaluations" / "validation-selected-test" / f"checkpoint-{step}" / split


def evaluation_result(config: dict, run_root: Path, step: int, split: str) -> dict | None:
    benchmark = config["benchmark"]
    key, scale, raw_name = METRICS[benchmark]
    dest = evaluation_dir(run_root, step, split)
    metrics_path = dest / "scores/metrics.json"
    raw_path = dest / "scores" / raw_name
    lifecycle = dest / "metadata/evaluator-lifecycle.json"
    stage = dest / "metadata/generation-stage.json"
    if not all(file.is_file() for file in (metrics_path, raw_path, lifecycle, stage)):
        return None
    metrics = read_json(metrics_path)
    stage_record = read_json(stage)
    raw = read_jsonl(raw_path) if raw_name.endswith(".jsonl") else json.loads(
        raw_path.read_text(encoding="utf-8"))
    count = config["splits"][split]["count"]
    score = metrics.get(key)
    if (metrics.get("status") != "completed" or metrics.get("sample_count") != count or
            read_json(lifecycle).get("status") != "completed" or
            stage_record.get("status") != "completed" or len(raw) != count or
            not isinstance(score, (int, float)) or not math.isfinite(score) or
            not 0 <= score <= scale):
        return None
    images = stage_record.get("images", {})
    if (stage_record.get("sample_count") != count or len(images) != count or
            stage_record.get("manifest_sha256") != digest(path(config["splits"][split]["path"])) or
            stage_record.get("checkpoint_sha256") != digest(checkpoint(run_root, step))):
        return None
    for record in images.values():
        image = Path(record["path"])
        if not image.is_file() or digest(image) != record["sha256"]:
            return None
    if benchmark in {"wise", "r2ibench"} and len(list((dest / "images").glob("*.png"))) != count:
        return None
    return {"step": step, "split": split, "score": score / scale, "metric": key,
            "sample_count": count, "metrics_path": str(metrics_path),
            "metrics_sha256": digest(metrics_path)}


def make_task(config: dict, run_id: str, run_root: Path, step: int,
              split: str, *, processes: int) -> tuple[Path, dict]:
    if not checkpoint_complete(run_root, step):
        raise ValueError(f"Incomplete checkpoint: c{step}")
    weights = checkpoint(run_root, step)
    manifest_spec = config["splits"][split]
    dest = evaluation_dir(run_root, step, split)
    settings = generate.DEFAULT_GENERATION | config.get("generation", {})
    settings.update(num_processes=processes)
    settings.pop("port", None)
    task = {
        "schema_version": 1, "benchmark": config["benchmark"],
        "run_id": run_id, "step": step, "split": split,
        "checkpoint": str(weights), "checkpoint_sha256": digest(weights),
        "manifest": str(path(manifest_spec["path"])),
        "manifest_sha256": digest(path(manifest_spec["path"])),
        "expected_count": manifest_spec["count"],
        "evaluation_dir": str(dest), "generation": settings,
    }
    task.update(config.get("evaluation", {}))
    for key in ("evaluator_root", "detector_model", "clip_cache", "judge_model"):
        if key in task:
            task[key] = str(path(task[key]))
    task_path = dest / "metadata/task.json"
    if task_path.is_file():
        if read_json(task_path) != task:
            raise ValueError(f"Evaluation task changed: {task_path}")
    else:
        write_json(task_path, task)
    return task_path, task


def run_evaluation(config: dict, run_id: str, run_root: Path, step: int,
                   split: str, *, slot: int = 0, processes: int = 1) -> dict:
    existing = evaluation_result(config, run_root, step, split)
    if existing is not None:
        return existing
    task_path, task = make_task(config, run_id, run_root, step, split,
                                processes=processes)
    environment = os.environ.copy()
    environment.update(DOPCD_RESULTS_ROOT=str(RESULTS),
                       CUDA_VISIBLE_DEVICES=str(slot) if processes == 1
                       else ",".join(str(item) for item in range(processes)))
    generate.run(task, env=environment, port=29610 + slot)
    interpreter = executable(config.get("evaluation", {}).get("python", sys.executable))
    if not interpreter.is_file():
        raise FileNotFoundError(f"Evaluator Python is unavailable: {interpreter}")
    for key in ("WORLD_SIZE", "RANK", "LOCAL_RANK", "LOCAL_WORLD_SIZE",
                "MASTER_ADDR", "MASTER_PORT"):
        environment.pop(key, None)
    if config["benchmark"] != "r2ibench":
        for key in list(environment):
            if key.lower() in {"http_proxy", "https_proxy", "all_proxy"}:
                environment.pop(key, None)
    run_logged([str(interpreter), str(ROOT.parent / "evaluation/benchmarks" / f"{config['benchmark']}.py"),
                "--task", str(task_path)],
               log=Path(task["evaluation_dir"]) / "metadata/evaluator-process.log",
               env=environment)
    result = evaluation_result(config, run_root, step, split)
    if result is None:
        raise ValueError(f"Incomplete {config['benchmark']} {split} c{step} evaluation")
    return result


def prepare_attempt(config: dict, config_path: Path, resume: str | None) -> tuple[str, Path, dict]:
    data_path = path(read_json(path(config["training"]["task_template"]))["data_path"])
    fingerprint = {"config_sha256": digest(config_path),
                   "training_data_sha256": digest(data_path)}
    if resume:
        if not resume.startswith(f"{config['experiment_id']}-r") or "/" in resume or ".." in resume:
            raise ValueError("Resume ID does not belong to this experiment")
        run_root = RUNS / resume
        frozen = read_json(run_root / "metadata/attempt.json")
        if any(frozen.get(key) != value for key, value in fingerprint.items()):
            raise ValueError("Attempt config or training data changed; create a new attempt")
        return resume, run_root, frozen
    run_id, run_root = reserve_run(config["experiment_id"])
    frozen = {"schema_version": 1, "run_id": run_id, "experiment_id": config["experiment_id"],
              "benchmark": config["benchmark"], "created_at": now(), **fingerprint}
    write_json(run_root / "metadata/attempt.json", frozen)
    template = read_json(path(config["training"]["task_template"]))
    template.update(run_id=run_id, task_name=run_id)
    args = template.setdefault("trainer_args", {})
    args.update(tracker_run_name=run_id, tracker_run_id=run_id)
    write_json(run_root / "metadata/training-task.json", template)
    return run_id, run_root, frozen


def run(config: dict, config_path: Path, *, resume: str | None = None) -> str:
    run_id, run_root, _ = prepare_attempt(config, config_path, resume)
    state_path = run_root / "metadata/e2e.json"
    state = read_json(state_path) if state_path.is_file() else {
        "run_id": run_id, "benchmark": config["benchmark"], "started_at": now()}
    state.update(status="running", updated_at=now())
    write_json(state_path, state)
    try:
        steps = sorted(config["checkpoint_steps"])
        if not all(checkpoint_complete(run_root, step) for step in steps):
            state.update(stage="training", updated_at=now())
            write_json(state_path, state)
            dopcd.run(run_root / "metadata/training-task.json",
                      path(config["training"]["runtime_config"]), execute=True)
        if not all(checkpoint_complete(run_root, step) for step in steps):
            raise ValueError("Training ended without all configured checkpoints")
        complete = read_json(run_root / "training/training_complete.json")
        if complete.get("global_step") != steps[-1] or complete.get("method") != "D-OPCD":
            raise ValueError("Training completion marker is inconsistent")
        state.update(stage="validation", training_complete=True, updated_at=now())
        write_json(state_path, state)
        candidates: list[dict] = []
        parallelism = config.get("generation", {}).get("validation_parallelism", 1)
        pending = [step for step in steps if evaluation_result(config, run_root, step, "validation") is None]
        candidates.extend(result for step in steps if
                          (result := evaluation_result(config, run_root, step, "validation")) is not None)
        for start in range(0, len(pending), parallelism):
            wave = pending[start:start + parallelism]
            with ThreadPoolExecutor(max_workers=len(wave)) as pool:
                futures = {pool.submit(run_evaluation, config, run_id, run_root, step,
                                       "validation", slot=slot): step
                           for slot, step in enumerate(wave)}
                for future in as_completed(futures):
                    candidates.append(future.result())
            state.update(validation_completed=len(candidates), updated_at=now())
            write_json(state_path, state)
        if {item["step"] for item in candidates} != set(steps):
            raise ValueError("Validation is incomplete; test remains gated")
        best = max(candidates, key=lambda item: (item["score"], -item["step"]))
        selection_path = run_root / "evaluations/validation-selected-test/selection.json"
        selection = {"status": "completed", "validation_only": True,
                     "selected_step": best["step"], "selected_score": best["score"],
                     "metric": best["metric"], "candidates": sorted(candidates, key=lambda x: x["step"])}
        if selection_path.is_file():
            if read_json(selection_path) != selection:
                raise ValueError("Existing checkpoint selection disagrees with validation")
        else:
            write_json(selection_path, selection)
        state.update(stage="test", selected_step=best["step"], updated_at=now())
        write_json(state_path, state)
        test = run_evaluation(config, run_id, run_root, best["step"], "test",
                              processes=config.get("generation", {}).get("test_processes", 1))
        state.update(status="completed", stage="completed", test_result=test,
                     finished_at=now(), updated_at=now())
        write_json(state_path, state)
        return run_id
    except BaseException as exc:
        state.update(status="failed", error_type=type(exc).__name__,
                     error_message=str(exc)[:2000], traceback=traceback.format_exc(),
                     finished_at=now(), updated_at=now())
        write_json(state_path, state)
        raise


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--execute", action="store_true", help="Create a new run; preflight by default")
    mode.add_argument("--resume", metavar="RUN_ID", help="Continue an exact existing attempt")
    args = parser.parse_args()
    config_path = args.config.expanduser().resolve()
    config = read_json(config_path)
    validate_config(config)
    summary = {"status": "configured", "experiment_id": config["experiment_id"],
               "benchmark": config["benchmark"], "method": config["method"],
               "training_template": str(path(config["training"]["task_template"])),
               "checkpoint_steps": config["checkpoint_steps"],
               "validation_count": config["splits"]["validation"]["count"],
               "test_count": config["splits"]["test"]["count"],
               "results_root": str(RESULTS)}
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)
    if args.execute or args.resume:
        print(json.dumps({"run_id": run(config, config_path, resume=args.resume)}), flush=True)


if __name__ == "__main__":
    main()

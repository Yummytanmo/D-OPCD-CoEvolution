"""Small, stable file and process helpers for script-based runners."""

from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
RESULTS = Path(os.environ.get("DOPCD_RESULTS_ROOT", ROOT / "results")).expanduser().resolve()
RUNS = RESULTS / "runs"
SAFE_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]*$")
BENCHMARKS = {"geneval", "geneval2", "wise", "r2ibench"}


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


def path(value: str | Path) -> Path:
    candidate = Path(value).expanduser()
    return (candidate if candidate.is_absolute() else ROOT / candidate).resolve()


def executable(value: str | Path) -> Path:
    candidate = Path(value).expanduser()
    return (candidate if candidate.is_absolute() else ROOT / candidate).absolute()


def read_json(source: Path) -> dict[str, Any]:
    value = json.loads(source.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"Expected JSON object: {source}")
    return value


def write_json(destination: Path, value: dict[str, Any]) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(f".{destination.name}.{os.getpid()}.tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    temporary.replace(destination)


def digest(source: Path) -> str:
    value = hashlib.sha256()
    with source.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            value.update(block)
    return value.hexdigest()


def read_jsonl(source: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in source.read_text(encoding="utf-8").splitlines()
            if line.strip()]


def validate_manifest(spec: dict[str, Any], split: str, benchmark: str) -> Path:
    manifest = path(spec["path"])
    rows = read_jsonl(manifest)
    if len(rows) != spec["count"] or any(row.get("split") != split for row in rows):
        raise ValueError(f"{split} manifest split/count mismatch: {manifest}")
    if any((row.get("benchmark") or row.get("dataset_id")) != benchmark for row in rows):
        raise ValueError(f"{split} benchmark mismatch: {manifest}")
    return manifest


def validate_config(config: dict[str, Any]) -> None:
    if config.get("schema_version") != 1 or config.get("method") != "dopcd":
        raise ValueError("The new E2E runner currently supports schema-1 D-OPCD experiments")
    identifier = config.get("experiment_id")
    if not isinstance(identifier, str) or not SAFE_ID.fullmatch(identifier):
        raise ValueError("experiment_id must be a safe filename component")
    benchmark = config.get("benchmark")
    if benchmark not in BENCHMARKS:
        raise ValueError(f"Unsupported benchmark: {benchmark}")
    training = config["training"]
    task_path = path(training["task_template"])
    runtime_path = path(training["runtime_config"])
    task = read_json(task_path)
    runtime = read_json(runtime_path)
    if task.get("schema_version") != 2 or task.get("experiment_id") != identifier:
        raise ValueError("Training template identity differs from E2E experiment")
    if task.get("run_id") or task.get("task_name"):
        raise ValueError("Reusable training template must not pin a run ID")
    if task.get("dataset") != benchmark:
        raise ValueError("Training task dataset differs from E2E benchmark")
    data = path(task["data_path"])
    if not data.is_relative_to((ROOT / "data").resolve()):
        raise ValueError("Training input must be under data")
    if task.get("expected_count") != len(read_jsonl(data)):
        raise ValueError("Training data count mismatch")
    if runtime.get("schema_version") != 1:
        raise ValueError("Unsupported training runtime config")
    for split in ("validation", "test"):
        validate_manifest(config["splits"][split], split, benchmark)
    steps = config.get("checkpoint_steps")
    if (not isinstance(steps, list) or not steps or
            any(not isinstance(step, int) or step < 1 for step in steps) or
            len(steps) != len(set(steps))):
        raise ValueError("checkpoint_steps must contain distinct positive integers")
    trainer_args = task.get("trainer_args", {})
    interval = trainer_args.get("checkpointing_steps")
    if (trainer_args.get("max_train_steps") != max(steps) or
            not isinstance(interval, int) or interval < 1 or
            any(step % interval for step in steps)):
        raise ValueError("Checkpoint list differs from the training schedule")
    generation = config.get("generation", {})
    parallelism = generation.get("validation_parallelism", 1)
    if not isinstance(parallelism, int) or parallelism < 1:
        raise ValueError("validation_parallelism must be positive")
    if benchmark in {"wise", "r2ibench"} and "judge" not in config.get("evaluation", {}):
        raise ValueError(f"{benchmark} requires judge configuration")


def reserve_run(identifier: str) -> tuple[str, Path]:
    RUNS.mkdir(parents=True, exist_ok=True)
    for number in range(1, 10000):
        run_id = f"{identifier}-r{number:03d}"
        root = RUNS / run_id
        try:
            root.mkdir()
        except FileExistsError:
            continue
        return run_id, root
    raise RuntimeError(f"No free run suffix for {identifier}")


def run_logged(command: list[str], *, log: Path, env: dict[str, str] | None = None) -> None:
    log.parent.mkdir(parents=True, exist_ok=True)
    with log.open("a", encoding="utf-8") as handle:
        subprocess.run(command, cwd=ROOT, env=env or os.environ.copy(),
                       stdout=handle, stderr=subprocess.STDOUT, check=True)

#!/usr/bin/env python3
"""Evaluate the raw image generator without GEMS, MLLM, Memory, or Skills."""

from __future__ import annotations

import argparse
import threading
import time
from pathlib import Path
from typing import Any

import requests

from agent.evolution.models import AttemptRecord, RunResult
from evaluation.client import normalize_feedback_result
from runners.config import EvolutionConfig
from runners.evaluate_frozen import _summary
from runners.evolve_stream import (
    _check_generator,
    _check_ready,
    _export_jsonl,
    _feedback_metadata,
    _read_json,
    _request_feedback_with_retry,
    _run_task_batch,
    _safe_name,
    _save_run_artifacts,
    _write_json,
)


VARIANT = "zimage_only"


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/evolution.example.json"),
    )
    parser.add_argument("--manifest", type=Path, action="append")
    parser.add_argument("--max-tasks", type=int)
    parser.add_argument("--workers", type=int, default=6)
    parser.add_argument(
        "--name",
        default="geneval-zimage-only",
        help="Evaluation suite name below runs/evaluations.",
    )
    parser.add_argument("--skip-health-check", action="store_true")
    return parser.parse_args(argv)


class _GeneratorClient:
    """Small thread-local client for exactly one raw generator call per sample."""

    def __init__(self, url: str, config: dict[str, Any]) -> None:
        self.url = str(url)
        self.timeout = float(config.get("timeout_seconds", 600))
        self.max_attempts = int(config.get("max_attempts", 2))
        self.retry_backoff = float(config.get("retry_backoff_seconds", 2))
        self.session = requests.Session()
        self.session.trust_env = bool(config.get("trust_env", True))
        if self.timeout <= 0 or self.max_attempts <= 0 or self.retry_backoff < 0:
            raise ValueError("Invalid raw generator retry configuration")

    def generate(self, prompt: str, *, seed: int | None) -> bytes:
        params: dict[str, Any] = {"prompt": str(prompt)}
        if seed is not None:
            params["seed"] = int(seed)
        for attempt in range(1, self.max_attempts + 1):
            try:
                response = self.session.post(
                    self.url,
                    params=params,
                    timeout=self.timeout,
                )
                if (
                    response.status_code not in {429, 500, 502, 503, 504}
                    or attempt >= self.max_attempts
                ):
                    response.raise_for_status()
                    return response.content
                response.close()
            except (requests.ConnectionError, requests.Timeout):
                if attempt >= self.max_attempts:
                    raise
            time.sleep(self.retry_backoff * (2 ** (attempt - 1)))
        raise RuntimeError("Raw generator request exhausted all attempts")

    def close(self) -> None:
        self.session.close()


def _run_dir(name: str) -> Path:
    normalized = str(name).strip()
    if not normalized or Path(normalized).name != normalized:
        raise ValueError("--name must be one non-empty directory name")
    return (Path("runs/evaluations") / normalized / VARIANT).resolve()


def _is_complete(run_dir: Path, task_id: str) -> bool:
    name = _safe_name(task_id)
    result_path = run_dir / "results" / f"{name}.json"
    feedback_path = run_dir / "private_feedback" / f"{name}.json"
    if not result_path.is_file() or not feedback_path.is_file():
        return False
    result = _read_json(result_path)
    image_path = result.get("final_image_path")
    return isinstance(image_path, str) and Path(image_path).is_file()


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    if args.workers <= 0:
        raise ValueError("--workers must be positive")

    config = EvolutionConfig.load(
        args.config,
        manifests=args.manifest,
        max_tasks=args.max_tasks,
        freeze_state=True,
        workers=args.workers,
        skip_health_check=args.skip_health_check,
    )
    config.configure_no_proxy()
    tasks = config.load_tasks()
    run_dir = _run_dir(args.name)
    run_dir.mkdir(parents=True, exist_ok=True)
    _write_json(
        run_dir / "effective_config.json",
        {
            **config.effective_dict(),
            "evaluation": {
                **config.evaluation,
                "variant": VARIANT,
                "mode": "original_prompt_single_generation",
                "workers": args.workers,
            },
            "uses_mllm": False,
            "uses_memory": False,
            "uses_skills": False,
            "uses_refinement": False,
        },
    )
    required_benchmarks = config.validate_feedback_services(tasks)

    if not config.skip_health_check:
        print("\n=== Z-Image-only service preflight ===", flush=True)
        _check_generator(
            config.generator_url,
            timeout=float(config.preflight.get("generator_timeout_seconds", 15)),
            health_path=str(
                config.preflight.get("generator_health_path", "/openapi.json")
            ),
            trust_env=bool(config.generator.get("trust_env", True)),
        )
        print("[preflight] generator: ready", flush=True)
        for benchmark in required_benchmarks:
            _check_ready(
                str(config.service_urls[benchmark]),
                timeout=float(config.preflight.get("feedback_timeout_seconds", 15)),
            )
            print(f"[preflight] feedback[{benchmark}]: ready", flush=True)

    local = threading.local()
    clients: list[_GeneratorClient] = []
    clients_lock = threading.Lock()

    def client() -> _GeneratorClient:
        value = getattr(local, "client", None)
        if value is None:
            value = _GeneratorClient(config.generator_url, config.generator)
            local.client = value
            with clients_lock:
                clients.append(value)
        return value

    def process(_index: int, task: dict[str, Any]) -> None:
        task_id = str(task["sample_id"])
        name = _safe_name(task_id)
        result_path = run_dir / "results" / f"{name}.json"
        if result_path.is_file():
            result = RunResult.from_dict(_read_json(result_path))
        else:
            prompt = str(task["prompt"])
            seed_value = task.get("inference_seed")
            seed = int(seed_value) if seed_value is not None else None
            image = client().generate(prompt, seed=seed)
            result = RunResult(
                task_id=task_id,
                original_prompt=prompt,
                final_prompt=prompt,
                final_image_bytes=image,
                attempts=[
                    AttemptRecord(
                        iteration=1,
                        prompt=prompt,
                        passed=[],
                        failed=[],
                        seed=seed,
                        experience="Direct Z-Image generation without agent processing.",
                        image_bytes=image,
                    )
                ],
                returned_attempt=1,
                generator_calls=1,
                mllm_calls=0,
                generator_model=config.generator_model,
                round_id=config.round_id,
            )
            result_path = _save_run_artifacts(result, run_dir)

        feedback_path = run_dir / "private_feedback" / f"{name}.json"
        if feedback_path.is_file():
            feedback = _read_json(feedback_path)
        else:
            feedback = _request_feedback_with_retry(
                max_attempts=config.feedback_max_attempts,
                retry_backoff_seconds=config.feedback_retry_backoff,
                service_url=str(config.service_urls[task["benchmark"]]),
                prompt=str(task["prompt"]),
                image_path=str(result.final_image_path),
                metadata=_feedback_metadata(task),
                sample_id=task_id,
                timeout=config.feedback_timeout,
                bypass_proxy=True,
                feedback_mode=config.feedback_mode,
            )
        feedback = normalize_feedback_result(
            feedback,
            feedback_mode=config.feedback_mode,
        )
        _write_json(feedback_path, feedback)
        _write_json(
            run_dir / "contexts" / f"{name}.json",
            {
                "sample_id": task_id,
                "q": result.original_prompt,
                "p": result.final_prompt,
                "image_path": result.final_image_path,
                "generator_model": result.generator_model,
                "variant": VARIANT,
            },
        )

    completed_before = sum(
        _is_complete(run_dir, str(task["sample_id"])) for task in tasks
    )
    try:
        indexed = list(enumerate(tasks, 1))
        batch_size = int(config.batch_size)
        total_batches = (len(indexed) + batch_size - 1) // batch_size
        for offset in range(0, len(indexed), batch_size):
            batch_id = offset // batch_size + 1
            batch = indexed[offset : offset + batch_size]
            pending = [
                item
                for item in batch
                if not _is_complete(run_dir, str(item[1]["sample_id"]))
            ]
            workers = min(args.workers, len(pending)) if pending else 0
            started = time.monotonic()
            print(
                f"[batch {batch_id}/{total_batches}] start tasks={len(batch)} "
                f"pending={len(pending)} workers={workers}",
                flush=True,
            )
            if pending:
                _run_task_batch(pending, workers=workers, process=process)
            print(
                f"[batch {batch_id}/{total_batches}] complete "
                f"generated={len(pending)} resumed={len(batch) - len(pending)} "
                f"elapsed={time.monotonic() - started:.1f}s",
                flush=True,
            )
    finally:
        for value in clients:
            value.close()

    _export_jsonl(run_dir / "private_feedback", run_dir / "private_evaluation.jsonl")
    _export_jsonl(run_dir / "contexts", run_dir / "contexts.jsonl")
    summary = _summary(run_dir, variant=VARIANT)
    print(
        f"Z-Image-only evaluation complete: tasks={summary['tasks']} "
        f"accuracy={summary['accuracy']:.4f} generated={len(tasks) - completed_before} "
        f"resumed={completed_before}",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

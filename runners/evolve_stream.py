#!/usr/bin/env python3
"""Run evaluator-preserving tasks with batched evolution or frozen evaluation."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import threading
import time
import urllib.error
import urllib.parse
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Callable

from agent.GEMS import GEMS
from agent.evolution.embedding import build_text_embedder
from agent.evolution.learner import HarnessLearner
from agent.evolution.memory import EpisodeStore, InsightConsolidator, InsightStore
from agent.evolution.models import (
    AttemptRecord,
    FirstAttemptReplay,
    LearningFeedback,
    RunResult,
)
from agent.evolution.reflection import EpisodeGenerator
from agent.evolution.skills import SkillEvolver, SkillRegistry
from agent.evolution.skills.workspace import DEFAULT_MAX_SKILL_CHARACTERS
from agent.evolution.storage import StateStore
from agent.evolution.tracing import EvolutionTraceStore
from agent.skill_manager import SkillManager
from evaluation.client import normalize_feedback_result, request_feedback_result
from runners.checkpointing import save_training_checkpoint_if_due
from runners.config import EvolutionConfig


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/evolution.example.json"),
    )
    parser.add_argument("--manifest", type=Path, action="append")
    parser.add_argument("--run-id")
    parser.add_argument("--gen-url")
    parser.add_argument("--mllm-url")
    parser.add_argument("--max-tasks", type=int)
    parser.add_argument("--shuffle-seed", type=int)
    parser.add_argument("--freeze-state", action="store_true")
    parser.add_argument(
        "--workers",
        type=int,
        help="Task workers used inside each fixed evolution/evaluation batch.",
    )
    parser.add_argument("--skip-health-check", action="store_true")
    parser.add_argument("--preflight-only", action="store_true")
    parser.add_argument(
        "--on-feedback-error",
        choices=["skip_update", "abort"],
    )
    return parser.parse_args(argv)


def _safe_name(task_id: str) -> str:
    prefix = "".join(character if character.isalnum() else "_" for character in task_id)
    digest = hashlib.sha1(task_id.encode("utf-8")).hexdigest()[:10]
    return f"{prefix[:80]}_{digest}"


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"Expected JSON object: {path}")
    return value


def _first_attempt_prefix_mllm_calls(result: RunResult) -> int:
    """Recover the logical MLLM calls consumed before first-attempt refinement."""
    attempts = sorted(result.attempts, key=lambda item: item.iteration)
    if not attempts:
        return 0

    # The saved total includes the Skill-only suffix. Remove every call after the
    # first experience summary: its first refinement, later verifications, and each
    # later summary/refinement pair. This keeps replayed call accounting comparable
    # without rerunning the shared prefix.
    suffix_calls = 0
    if len(attempts) > 1:
        suffix_calls += 1  # Refinement call after the saved first experience.
    for position, attempt in enumerate(attempts[1:], start=1):
        if attempt.passed or attempt.failed:
            suffix_calls += 1  # One batched verifier call for all checks.
        if position < len(attempts) - 1:
            suffix_calls += 2  # Experience summary plus the following refinement.
    if suffix_calls > result.mllm_calls:
        raise ValueError(
            f"cannot recover first-attempt MLLM calls for task {result.task_id}: "
            f"suffix={suffix_calls}, total={result.mllm_calls}"
        )
    return result.mllm_calls - suffix_calls


class FirstTrajectorySource:
    """Validated, read-only Skill-only first-attempt source for paired evaluation."""

    def __init__(self, run_dir: Path, tasks: list[dict[str, Any]]):
        self.run_dir = run_dir.expanduser().resolve()
        effective = _read_json(self.run_dir / "effective_config.json")
        evaluation = effective.get("evaluation") or {}
        if not isinstance(evaluation, dict):
            raise ValueError(
                f"reference evaluation config is invalid: {self.run_dir}"
            )
        if evaluation.get("enable_skills") is not True:
            raise ValueError("reference first trajectories must have Skills enabled")
        if evaluation.get("enable_memory") is not False:
            raise ValueError(
                "reference first trajectories must come from a Skill-only run"
            )

        self._records: dict[
            str,
            tuple[RunResult, AttemptRecord, Path, int],
        ] = {}
        for task in tasks:
            task_id = str(task["sample_id"])
            name = _safe_name(task_id)
            result_path = self.run_dir / "results" / f"{name}.json"
            result = RunResult.from_dict(_read_json(result_path))
            if result.task_id != task_id:
                raise ValueError(
                    f"reference result task mismatch: expected {task_id}, "
                    f"got {result.task_id}"
                )
            if result.original_prompt != str(task["prompt"]):
                raise ValueError(
                    f"reference result prompt mismatch for task {task_id}"
                )
            first = min(result.attempts, key=lambda item: item.iteration, default=None)
            if first is None or first.iteration != 1:
                raise ValueError(
                    f"reference result has no first attempt for task {task_id}"
                )
            expected_seed = task.get("inference_seed")
            if (
                expected_seed is not None
                and first.seed is not None
                and int(expected_seed) != int(first.seed)
            ):
                raise ValueError(
                    f"reference first-attempt seed mismatch for task {task_id}"
                )
            image_path = Path(first.image_path) if first.image_path else Path()
            if not first.image_path or not image_path.is_file():
                image_path = self.run_dir / "images" / name / "attempt_01.png"
            if not image_path.is_file():
                raise FileNotFoundError(
                    f"reference first-attempt image is missing: {image_path}"
                )
            if first.failed and not first.experience.strip():
                raise ValueError(
                    f"reference failed first attempt lacks experience: {task_id}"
                )
            self._records[task_id] = (
                result,
                first,
                image_path.resolve(),
                _first_attempt_prefix_mllm_calls(result),
            )

    def replay(self, task: dict[str, Any]) -> FirstAttemptReplay:
        task_id = str(task["sample_id"])
        try:
            result, first, image_path, prefix_mllm_calls = self._records[task_id]
        except KeyError as error:
            raise KeyError(f"reference first trajectory is missing: {task_id}") from error
        attempt = AttemptRecord(
            iteration=1,
            prompt=first.prompt,
            passed=list(first.passed),
            failed=list(first.failed),
            seed=first.seed,
            experience=first.experience,
            image_path=str(image_path),
            image_bytes=image_path.read_bytes(),
        )
        return FirstAttemptReplay(
            task_id=task_id,
            original_prompt=result.original_prompt,
            selected_skills=[dict(item) for item in result.selected_skills],
            attempt=attempt,
            source=str(self.run_dir),
            prefix_generator_calls=1,
            prefix_mllm_calls=prefix_mllm_calls,
        )


def _reject_incompatible_legacy_run(run_dir: Path) -> None:
    """Protect artifacts created by the former SQLite evolution store."""
    legacy_store = run_dir / "harness.sqlite"
    current_store = run_dir / "evolution" / "metadata.json"
    if legacy_store.exists() and not current_store.exists():
        raise RuntimeError(
            "run directory contains legacy harness.sqlite state but no current "
            f"Episode/Insight state: {run_dir}. Use a new run_id to preserve the "
            "legacy experiment."
        )


def _request_json(
    request: urllib.request.Request,
    *,
    timeout: float,
    service_name: str,
    trust_env: bool = True,
) -> dict[str, Any]:
    opener = (
        urllib.request.build_opener()
        if trust_env
        else urllib.request.build_opener(urllib.request.ProxyHandler({}))
    )
    try:
        with opener.open(request, timeout=timeout) as response:
            value = json.loads(response.read().decode("utf-8"))
    except urllib.error.HTTPError as error:
        detail = error.read(512).decode("utf-8", errors="replace")
        raise RuntimeError(
            f"{service_name} preflight failed: HTTP {error.code}: {detail}"
        ) from error
    except (urllib.error.URLError, TimeoutError, json.JSONDecodeError) as error:
        raise RuntimeError(f"{service_name} preflight failed: {error}") from error
    if not isinstance(value, dict):
        raise RuntimeError(f"{service_name} preflight returned non-object JSON")
    return value


def _check_generator(
    generator_url: str,
    *,
    timeout: float,
    health_path: str = "/openapi.json",
    trust_env: bool = True,
) -> None:
    parsed = urllib.parse.urlsplit(generator_url)
    if parsed.scheme not in {"http", "https"} or not parsed.netloc:
        raise RuntimeError(f"Invalid generator URL: {generator_url}")
    health_url = urllib.parse.urlunsplit(
        (parsed.scheme, parsed.netloc, health_path, "", "")
    )
    value = _request_json(
        urllib.request.Request(health_url, headers={"Accept": "application/json"}),
        timeout=timeout,
        service_name=f"Generator ({health_url})",
        trust_env=trust_env,
    )
    paths = value.get("paths")
    if health_path == "/openapi.json" and (
        not isinstance(paths, dict) or parsed.path not in paths
    ):
        raise RuntimeError(
            f"Generator preflight failed: {health_url} does not expose {parsed.path}"
        )


def _resolve_mllm_api_key(mllm_config: dict[str, Any]) -> str:
    configured_name = str(mllm_config.get("api_key_env") or "").strip()
    if configured_name:
        api_key = os.getenv(configured_name)
        if not api_key:
            raise RuntimeError(
                f"MLLM preflight failed: environment variable {configured_name} is empty"
            )
        return api_key
    for name in (
        "GEMS_MLLM_API_KEY",
        "OPENAI_API_KEY",
    ):
        if os.getenv(name):
            return str(os.environ[name])
    return "none"


def _check_mllm(
    mllm_url: str,
    *,
    model: str,
    api_key: str,
    timeout: float,
    max_tokens: int,
    extra_body: dict[str, Any] | None = None,
    trust_env: bool = True,
) -> None:
    endpoint = f"{mllm_url.rstrip('/')}/chat/completions"
    if extra_body is not None and not isinstance(extra_body, dict):
        raise ValueError("mllm.extra_body must be a JSON object")
    body = json.dumps(
        {
            "model": model,
            "messages": [{"role": "user", "content": "Reply with OK."}],
            "max_tokens": max_tokens,
            "stream": False,
            **dict(extra_body or {}),
        }
    ).encode("utf-8")
    value = _request_json(
        urllib.request.Request(
            endpoint,
            data=body,
            headers={
                "Accept": "application/json",
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json",
            },
            method="POST",
        ),
        timeout=timeout,
        service_name=f"MLLM ({endpoint})",
        trust_env=trust_env,
    )
    if not isinstance(value.get("choices"), list) or not value["choices"]:
        raise RuntimeError(f"MLLM preflight returned no choices: {endpoint}")


def _check_ready(service_url: str, timeout: float = 10.0) -> None:
    ready_url = f"{service_url.rstrip('/')}/ready"
    value = _request_json(
        urllib.request.Request(ready_url, headers={"Accept": "application/json"}),
        timeout=timeout,
        service_name=f"Feedback service ({ready_url})",
    )
    if value.get("status") != "ready":
        raise RuntimeError(f"Feedback service is not ready: {service_url}: {value}")


def _run_service_preflight(
    *,
    generator_url: str,
    generator_config: dict[str, Any],
    mllm_url: str,
    mllm_config: dict[str, Any],
    service_urls: dict[str, Any],
    required_benchmarks: list[str],
    preflight_config: dict[str, Any],
) -> None:
    generator_timeout = float(
        preflight_config.get("generator_timeout_seconds", 15.0)
    )
    mllm_timeout = float(preflight_config.get("mllm_timeout_seconds", 60.0))
    feedback_timeout = float(
        preflight_config.get("feedback_timeout_seconds", 15.0)
    )
    mllm_max_tokens = int(preflight_config.get("mllm_max_tokens", 8))
    if min(generator_timeout, mllm_timeout, feedback_timeout) <= 0:
        raise ValueError("preflight timeouts must be positive")
    if mllm_max_tokens <= 0:
        raise ValueError("preflight.mllm_max_tokens must be positive")

    print("\n=== Service preflight ===", flush=True)
    print(f"[preflight] generator: {generator_url}", flush=True)
    _check_generator(
        generator_url,
        timeout=generator_timeout,
        health_path=str(preflight_config.get("generator_health_path", "/openapi.json")),
        trust_env=bool(generator_config.get("trust_env", True)),
    )
    print("[preflight] generator: ready", flush=True)

    mllm_model = str(mllm_config.get("model") or "kimi-k2.5")
    print(f"[preflight] mllm: {mllm_url} model={mllm_model}", flush=True)
    _check_mllm(
        mllm_url,
        model=mllm_model,
        api_key=_resolve_mllm_api_key(mllm_config),
        timeout=mllm_timeout,
        max_tokens=mllm_max_tokens,
        extra_body=mllm_config.get("extra_body"),
        trust_env=bool(mllm_config.get("trust_env", True)),
    )
    print("[preflight] mllm: ready", flush=True)

    for benchmark in required_benchmarks:
        service_url = str(service_urls[benchmark])
        print(f"[preflight] feedback[{benchmark}]: {service_url}", flush=True)
        _check_ready(service_url, timeout=feedback_timeout)
        print(f"[preflight] feedback[{benchmark}]: ready", flush=True)
    print("Service preflight passed.", flush=True)


def _save_run_artifacts(result: RunResult, run_dir: Path) -> Path:
    name = _safe_name(result.task_id)
    image_dir = run_dir / "images" / name
    image_dir.mkdir(parents=True, exist_ok=True)
    for attempt in result.attempts:
        if attempt.image_bytes is None:
            continue
        attempt_path = image_dir / f"attempt_{attempt.iteration:02d}.png"
        attempt_path.write_bytes(attempt.image_bytes)
        attempt.image_path = str(attempt_path.resolve())
    final_path = image_dir / "final.png"
    if result.final_image_bytes is None:
        raise RuntimeError("run result does not contain final image bytes")
    final_path.write_bytes(result.final_image_bytes)
    result.final_image_path = str(final_path.resolve())
    result_path = run_dir / "results" / f"{name}.json"
    _write_json(result_path, result.to_dict())
    return result_path


def _export_jsonl(source_dir: Path, output: Path) -> None:
    records = []
    if source_dir.exists():
        for path in sorted(source_dir.glob("*.json")):
            value = json.loads(path.read_text(encoding="utf-8"))
            records.append(
                json.dumps(value, ensure_ascii=False, separators=(",", ":"))
            )
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(output.suffix + ".tmp")
    temporary.write_text(
        "\n".join(record for record in records if record) + ("\n" if records else ""),
        encoding="utf-8",
    )
    temporary.replace(output)


def _feedback_metadata(task: dict[str, Any]) -> dict[str, Any] | None:
    return dict(task) if task["benchmark"] in {"geneval", "geneval2", "wise", "r2ibench"} else None


def _request_feedback_with_retry(
    *,
    max_attempts: int,
    retry_backoff_seconds: float,
    **kwargs,
) -> dict[str, Any]:
    if max_attempts <= 0:
        raise ValueError("feedback.max_attempts must be positive")
    if retry_backoff_seconds < 0:
        raise ValueError("feedback.retry_backoff_seconds must be non-negative")
    for attempt in range(1, max_attempts + 1):
        try:
            return request_feedback_result(**kwargs)
        except Exception:
            if attempt >= max_attempts:
                raise
            time.sleep(retry_backoff_seconds * (2 ** (attempt - 1)))
    raise RuntimeError("feedback request exhausted all attempts")


def _run_task_batch(
    pending: list[tuple[int, dict[str, Any]]],
    *,
    workers: int,
    process: Callable[[int, dict[str, Any]], None],
) -> int:
    if workers == 1:
        for index, task in pending:
            process(index, task)
        return len(pending)

    executor = ThreadPoolExecutor(
        max_workers=workers,
        thread_name_prefix="gems-eval",
    )
    futures = {
        executor.submit(process, index, task): str(task["sample_id"])
        for index, task in pending
    }
    completed = 0
    try:
        for future in as_completed(futures):
            task_id = futures[future]
            try:
                future.result()
            except BaseException as error:
                raise RuntimeError(f"task failed: sample_id={task_id}") from error
            completed += 1
    except BaseException:
        for future in futures:
            future.cancel()
        raise
    finally:
        executor.shutdown(wait=True, cancel_futures=True)
    return completed


def _process_task(
    *,
    index: int,
    total: int,
    task: dict[str, Any],
    agent: GEMS,
    state: StateStore,
    learner: HarnessLearner | None,
    episode_generator: EpisodeGenerator | None,
    episode_summary_enabled: bool,
    batch_id: int,
    run_dir: Path,
    service_urls: dict[str, Any],
    feedback_timeout: float,
    feedback_max_attempts: int,
    feedback_retry_backoff: float,
    feedback_error_mode: str,
    feedback_mode: str,
    frozen: bool,
    first_trajectory_source: FirstTrajectorySource | None = None,
    defer_feedback: bool = False,
) -> None:
    """Finish one task before exposing its evaluator signal to evolution.

    Generated artifacts and feedback are cached independently. This ordering prevents
    evaluator information from leaking into the current task and lets interrupted runs
    resume without regenerating an image or requesting the same evaluation twice.
    """
    if defer_feedback and (not frozen or learner is not None):
        raise ValueError("Deferred feedback is only valid for frozen generation")
    task_id = str(task["sample_id"])
    name = _safe_name(task_id)
    progress = state.progress(task_id)
    result_path = run_dir / "results" / f"{name}.json"
    if (
        progress
        and progress["status"] in {"generated", "feedback_error", "episode_ready"}
        and result_path.is_file()
    ):
        result = RunResult.from_dict(_read_json(result_path))
    else:
        public_task = {
            "prompt": task["prompt"],
            "sample_id": task_id,
            "inference_seed": task.get("inference_seed"),
        }
        first_attempt_replay = (
            first_trajectory_source.replay(task)
            if first_trajectory_source is not None
            else None
        )
        result = (
            agent.run_with_result(
                public_task,
                first_attempt_replay=first_attempt_replay,
            )
            if first_attempt_replay is not None
            else agent.run_with_result(public_task)
        )
        result_path = _save_run_artifacts(result, run_dir)
        state.mark_progress(
            task_id,
            "generated",
            image_path=result.final_image_path,
            result_path=str(result_path.resolve()),
        )

    if defer_feedback:
        _write_json(run_dir / "contexts" / f"{name}.json", {
            "sample_id": task_id, "q": result.original_prompt,
            "p": result.final_prompt, "image_path": result.final_image_path,
            "generator_model": result.generator_model,
            "evaluation_status": "pending_official_offline_scoring",
        })
        state.mark_progress(task_id, "committed")
        return

    feedback_path = run_dir / "private_feedback" / f"{name}.json"
    try:
        if feedback_path.is_file():
            feedback_response = _read_json(feedback_path)
        else:
            feedback_response = _request_feedback_with_retry(
                max_attempts=feedback_max_attempts,
                retry_backoff_seconds=feedback_retry_backoff,
                service_url=str(service_urls[task["benchmark"]]),
                prompt=str(task["prompt"]),
                image_path=str(result.final_image_path),
                metadata=_feedback_metadata(task),
                sample_id=task_id,
                timeout=feedback_timeout,
                feedback_mode=feedback_mode,
            )
        feedback_response = normalize_feedback_result(
            feedback_response,
            feedback_mode=feedback_mode,
        )
        _write_json(feedback_path, feedback_response)
    except Exception as error:
        error_text = f"{type(error).__name__}: {error}"
        if feedback_error_mode == "abort":
            state.mark_progress(
                task_id,
                "feedback_error",
                feedback_error=error_text,
            )
            raise
        print(f"Feedback failed; state update skipped: {error}", file=sys.stderr)
        _write_json(
            run_dir / "contexts" / f"{name}.json",
            {
                "sample_id": task_id,
                "q": result.original_prompt,
                "p": result.final_prompt,
                "image_path": result.final_image_path,
                "generator_model": result.generator_model,
                "round_id": result.round_id,
                "feedback_error": error_text,
            },
        )
        state.mark_progress(
            task_id,
            "committed",
            feedback_error=error_text,
        )
        return

    # Frozen evaluation follows the identical evaluator path but stops at this boundary,
    # ensuring validation never mutates training memory or skills.
    if learner is not None:
        update = learner.create_episode(
            result,
            LearningFeedback(
                text=feedback_response.get("feedback"),
                reward=feedback_response.get("reward"),
            ),
            batch_id=batch_id,
            generator=episode_generator,
            generate_summary=episode_summary_enabled,
        )
        _write_json(
            run_dir / "updates" / f"{name}.json",
            {
                "task_id": update.task_id,
                "episode_id": update.episode_id,
                "batch_id": update.batch_id,
                "duplicate": update.duplicate,
            },
        )

    _write_json(
        run_dir / "contexts" / f"{name}.json",
        {
            "sample_id": task_id,
            "q": result.original_prompt,
            "p": result.final_prompt,
            "image_path": result.final_image_path,
            "generator_model": result.generator_model,
            "round_id": result.round_id,
        },
    )
    state.mark_progress(task_id, "episode_ready" if learner is not None else "committed")


def main(argv: list[str] | None = None) -> int:
    """Resolve inputs, validate services, then execute the resumable task stream."""
    args = parse_args(argv)
    config = EvolutionConfig.load(
        args.config,
        manifests=args.manifest,
        run_id=args.run_id,
        gen_url=args.gen_url,
        mllm_url=args.mllm_url,
        max_tasks=args.max_tasks,
        shuffle_seed=args.shuffle_seed,
        freeze_state=args.freeze_state,
        workers=args.workers,
        skip_health_check=args.skip_health_check,
        feedback_error_mode=args.on_feedback_error,
    )
    tasks = config.load_tasks()
    defer_feedback = bool(config.evaluation.get("defer_feedback", False))
    if defer_feedback and not config.freeze_state:
        raise ValueError("evaluation.defer_feedback requires freeze_state")
    first_trajectory_source = None
    configured_first_trajectory_run = config.evaluation.get(
        "first_trajectory_run"
    )
    if configured_first_trajectory_run is not None:
        first_trajectory_source = FirstTrajectorySource(
            Path(str(configured_first_trajectory_run)),
            tasks,
        )
        print(
            "[evaluation] replaying Skill-only first trajectories from "
            f"{first_trajectory_source.run_dir}",
            flush=True,
        )
    run_dir = config.run_dir
    run_dir.mkdir(parents=True, exist_ok=True)
    _reject_incompatible_legacy_run(run_dir)

    # Persist the fully resolved values so a resumed run is auditable even when CLI
    # overrides or legacy aliases were used at launch time.
    _write_json(run_dir / "effective_config.json", config.effective_dict())
    required_benchmarks = [] if defer_feedback else config.validate_feedback_services(tasks)

    # Fail before generation if any external dependency is unavailable; otherwise a
    # partial task could consume budget without ever receiving post-task feedback.
    if not config.skip_health_check:
        _run_service_preflight(
            generator_url=config.generator_url,
            generator_config=config.generator,
            mllm_url=config.mllm_url,
            mllm_config=config.mllm,
            service_urls=config.service_urls,
            required_benchmarks=[] if defer_feedback else required_benchmarks,
            preflight_config=config.preflight,
        )
    elif args.preflight_only:
        raise ValueError("--preflight-only cannot be combined with disabled preflight")
    if args.preflight_only:
        return 0

    # These objects share one file-backed state root. Constructing them together makes
    # the task commit the only boundary at which memory and skill state can advance.
    state = StateStore(
        run_dir / "evolution",
        run_id=config.run_id,
        generator_model=config.generator_model,
        round_id=config.round_id,
    )
    embedder = build_text_embedder(config.memory)
    embedding_config = config.memory.get("embedding") or {}
    if bool(embedding_config.get("preload", False)):
        print(
            f"[memory] embedding: loading {embedding_config.get('model')}",
            flush=True,
        )
        embedder.embed_queries(["Insight retrieval readiness check."])
        print("[memory] embedding: ready", flush=True)
    insights = InsightStore(
        state,
        embedder=embedder,
        top_k=int(config.memory.get("top_k", 4)),
        consolidation_top_k=int(
            config.memory.get("consolidation_top_k", 4)
        ),
        semantic_threshold=float(config.memory.get("semantic_threshold", 0.35)),
        duplicate_threshold=float(config.memory.get("duplicate_threshold", 0.92)),
        retrieval_strategy=str(
            config.memory.get("retrieval_strategy") or "embedding"
        ),
    )
    episodes = EpisodeStore(state)
    traces = EvolutionTraceStore(state)
    configured_hard_max = config.skill.get("hard_max_skills")
    registry = SkillRegistry(
        state,
        max_skills=(
            int(configured_hard_max)
            if configured_hard_max is not None
            else None
        ),
    )
    registry.initialize(
        str(config.skill.get("initialization") or "empty"),
        str(config.skill.get("seed_dir") or "agent/skills"),
    )

    def build_agent() -> GEMS:
        return GEMS(
            gen_url=config.generator_url,
            mllm_url=config.mllm_url,
            max_iterations=int(
                config.agent.get("max_iterations", config.data.get("max_iterations", 5))
            ),
            skill_manager=SkillManager(registry=registry),
            insight_store=insights,
            generator_model=config.generator_model,
            round_id=config.round_id,
            verification_max_questions=int(
                config.verification.get("max_questions", 10)
            ),
            generator_config=config.generator,
            mllm_config=config.mllm,
            enable_skills=config.enable_skills,
            enable_memory=config.enable_memory,
            prompt_adapter=bool(config.agent.get("prompt_adapter", False)),
            prompt_adapter_seed=int(config.agent.get("prompt_adapter_seed", 20260920)),
            verbose=False,
        )

    worker_agents: list[GEMS] = []
    worker_agents_lock = threading.Lock()
    worker_local = threading.local()

    def remember_agent(value: GEMS) -> GEMS:
        with worker_agents_lock:
            worker_agents.append(value)
        return value

    learner = None
    episode_summary_enabled = bool(config.memory.get("episode_summary", False))
    if not config.freeze_state:
        learning_agent = remember_agent(build_agent())
        learner = HarnessLearner(
            state=state,
            episodes=episodes,
            insights=insights,
            registry=registry,
            consolidator=InsightConsolidator(
                learning_agent.think,
                trace_store=traces,
                max_attempts=int(
                    config.memory.get("consolidation_max_attempts", 3)
                ),
            ),
            evolver=SkillEvolver(
                learning_agent.think,
                min_support=int(config.skill.get("min_support", 4)),
                min_batches=int(config.skill.get("min_batches", 3)),
                min_mean_reward=float(
                    config.skill.get("min_mean_reward", config.reward_threshold)
                ),
                min_mature_insights=int(
                    config.skill.get("min_mature_insights", 5)
                ),
                trace_store=traces,
                serialized_operations=True,
                max_attempts=int(
                    config.skill.get("evolution_max_attempts", 3)
                ),
                max_skill_characters=int(
                    config.skill.get(
                        "max_skill_characters", DEFAULT_MAX_SKILL_CHARACTERS
                    )
                ),
                min_batches_between_evolutions=int(
                    config.skill.get("min_batches_between_evolutions", 0)
                ),
            ),
        )

    def process(batch_id: int, index: int, task: dict[str, Any]) -> None:
        agent = getattr(worker_local, "agent", None)
        if agent is None:
            agent = remember_agent(build_agent())
            worker_local.agent = agent
            if learner is not None and episode_summary_enabled:
                worker_local.episode_generator = EpisodeGenerator(
                    agent.think,
                    trace_store=traces,
                )
        _process_task(
            index=index,
            total=len(tasks),
            task=task,
            agent=agent,
            state=state,
            learner=learner,
            episode_generator=(
                getattr(worker_local, "episode_generator", None)
                if learner is not None and episode_summary_enabled
                else None
            ),
            episode_summary_enabled=episode_summary_enabled,
            batch_id=batch_id,
            run_dir=run_dir,
            service_urls=config.service_urls,
            feedback_timeout=config.feedback_timeout,
            feedback_max_attempts=config.feedback_max_attempts,
            feedback_retry_backoff=config.feedback_retry_backoff,
            feedback_error_mode=config.feedback_error_mode,
            feedback_mode=config.feedback_mode,
            frozen=config.freeze_state,
            first_trajectory_source=first_trajectory_source,
            defer_feedback=defer_feedback,
        )

    # Close every per-thread client even when one future fails; leaked clients otherwise
    # make retries look like service failures rather than a local lifecycle problem.
    completed = 0
    skipped = sum(
        state.is_committed(str(task["sample_id"])) for task in tasks
    )
    try:
        indexed_tasks = list(enumerate(tasks, 1))
        total_batches = (
            len(indexed_tasks) + config.batch_size - 1
        ) // config.batch_size
        for offset in range(0, len(indexed_tasks), config.batch_size):
            batch_id = offset // config.batch_size + 1
            batch = indexed_tasks[offset : offset + config.batch_size]
            pending = [
                (index, task)
                for index, task in batch
                if not state.is_committed(str(task["sample_id"]))
            ]
            batch_started = time.monotonic()
            worker_count = min(config.task_workers, len(pending)) if pending else 0
            print(
                f"\n[batch {batch_id}/{total_batches}] start "
                f"tasks={len(batch)} pending={len(pending)} workers={worker_count}",
                flush=True,
            )
            batch_completed = 0
            if pending:
                batch_completed = _run_task_batch(
                    pending,
                    workers=worker_count,
                    process=lambda index, task, current=batch_id: process(
                        current, index, task
                    ),
                )
                completed += batch_completed
            batch_summary = [
                f"tasks={batch_completed}/{len(pending)}",
                f"resumed={len(batch) - len(pending)}",
            ]
            if learner is not None:
                task_ids = [str(task["sample_id"]) for _, task in batch]
                update = learner.commit_batch(batch_id=batch_id, task_ids=task_ids)
                _write_json(
                    run_dir / "updates" / f"batch_{batch_id:06d}.json",
                    {
                        "batch_id": update.batch_id,
                        "duplicate": update.duplicate,
                        "episode_ids": update.episode_ids,
                        "accepted_operations": update.accepted_operations,
                        "rejected_operations": update.rejected_operations,
                        "touched_insights": update.touched_insights,
                        "skill_changes": update.skill_changes,
                    },
                )
                for task_id in task_ids:
                    if episodes.get(task_id) is not None:
                        state.mark_progress(task_id, "committed")
                checkpoint = save_training_checkpoint_if_due(
                    run_dir=run_dir,
                    state=state,
                    every_tasks=config.checkpoint_every_tasks,
                    batch_id=batch_id,
                )
                batch_rewards = [
                    float(reward)
                    for episode in episodes.for_batch(batch_id)
                    if isinstance(
                        reward := (episode.get("outcome") or {}).get("reward"),
                        (int, float),
                    )
                    and not isinstance(reward, bool)
                ]
                batch_summary.extend(
                    [
                        f"episodes={len(update.episode_ids)}",
                        "mean_reward="
                        + (
                            f"{sum(batch_rewards) / len(batch_rewards):.3f}"
                            if batch_rewards
                            else "n/a"
                        ),
                        f"insights=accepted:{len(update.accepted_operations)},"
                        f"rejected:{len(update.rejected_operations)},"
                        f"touched:{len(update.touched_insights)}",
                    ]
                )
                applied_skill_operations = sum(
                    len(change.get("changes", [change]))
                    for change in update.skill_changes
                )
                skill_cycle = state.read_record(
                    "skill_cycles", f"{state.round_id}:{batch_id}"
                )
                if skill_cycle is None:
                    batch_summary.append("skill_trigger=not_evaluated")
                else:
                    trigger_summary = str(skill_cycle.get("outcome") or "unknown")
                    if "pending_mature_count" in skill_cycle:
                        trigger_summary += (
                            ":insights="
                            f"{int(skill_cycle['pending_mature_count'])}/"
                            f"{int(skill_cycle['trigger_mature_insights'])}"
                        )
                    batch_summary.append(
                        f"skill_trigger={trigger_summary} "
                        f"changes={applied_skill_operations}"
                    )
                if checkpoint is not None:
                    batch_summary.append(f"checkpoint={checkpoint.name}")
            batch_summary.append(
                f"elapsed={time.monotonic() - batch_started:.1f}s"
            )
            print(
                f"[batch {batch_id}/{total_batches}] complete "
                + " | ".join(batch_summary),
                flush=True,
            )
    finally:
        for worker_agent in worker_agents:
            worker_agent._generator_session.close()
            worker_agent._mllm_http_client.close()

    # JSONL files are rebuilt from per-task records, making exports deterministic and
    # preventing duplicate lines when the stream is resumed.
    _export_jsonl(run_dir / "private_feedback", run_dir / "private_evaluation.jsonl")
    _export_jsonl(run_dir / "contexts", run_dir / "contexts.jsonl")
    print(
        f"{'Frozen evaluation' if config.freeze_state else 'Batched evolution'} complete: "
        f"committed={completed}, resumed/skipped={skipped}, "
        f"total={len(tasks)}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

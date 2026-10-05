"""Configuration resolution shared by evolution and frozen-evaluation runners."""

from __future__ import annotations

import hashlib
import json
import os
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable
from urllib.parse import urlsplit


_VARIANT_CAPABILITIES = {
    "evolved": (True, True),
    "full": (True, True),
    "baseline": (False, False),
    "empty": (False, False),
    "memory_only": (False, True),
    "skill_only": (True, False),
}


def _section(value: dict[str, Any], name: str) -> dict[str, Any]:
    section = value.get(name)
    if section is None:
        return {}
    if not isinstance(section, dict):
        raise ValueError(f"{name} config must be a JSON object")
    return dict(section)


def resolve_task_workers(
    *,
    freeze_state: bool,
    cli_workers: int | None,
    evaluation_config: dict[str, Any],
) -> int:
    """Resolve task concurrency; evolution writes occur only at the batch barrier."""
    configured = evaluation_config.get("workers")
    default = 4
    workers = int(
        cli_workers
        if cli_workers is not None
        else configured if configured is not None else default
    )
    if workers <= 0:
        raise ValueError("evaluation.workers/--workers must be positive")
    return workers


@dataclass(frozen=True)
class EvolutionConfig:
    """Validated, CLI-resolved view of one evolution configuration.

    Runners consume stable properties from this class instead of each reimplementing
    defaults, compatibility aliases, and validation in their orchestration code.
    """

    data: dict[str, Any]
    manifest_paths: tuple[Path, ...]
    run_id: str
    run_root: Path
    round_id: int
    freeze_state: bool
    skip_health_check: bool
    max_tasks: int | None
    shuffle_seed: int | None
    generator: dict[str, Any]
    mllm: dict[str, Any]
    agent: dict[str, Any]
    verification: dict[str, Any]
    feedback: dict[str, Any]
    preflight: dict[str, Any]
    evaluation: dict[str, Any]
    memory: dict[str, Any]
    skill: dict[str, Any]
    generator_url: str
    mllm_url: str
    generator_model: str
    service_urls: dict[str, Any]
    feedback_timeout: float
    feedback_max_attempts: int
    feedback_retry_backoff: float
    feedback_error_mode: str
    feedback_mode: str
    reward_threshold: float
    task_workers: int
    batch_size: int
    checkpoint_every_tasks: int
    enable_skills: bool
    enable_memory: bool

    @classmethod
    def load(
        cls,
        path: Path,
        *,
        manifests: list[Path] | None = None,
        run_id: str | None = None,
        gen_url: str | None = None,
        mllm_url: str | None = None,
        max_tasks: int | None = None,
        shuffle_seed: int | None = None,
        freeze_state: bool = False,
        workers: int | None = None,
        skip_health_check: bool = False,
        feedback_error_mode: str | None = None,
    ) -> "EvolutionConfig":
        value = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(value, dict):
            raise ValueError("evolution config must be a JSON object")
        return cls.from_mapping(
            value,
            manifests=manifests,
            run_id=run_id,
            gen_url=gen_url,
            mllm_url=mllm_url,
            max_tasks=max_tasks,
            shuffle_seed=shuffle_seed,
            freeze_state=freeze_state,
            workers=workers,
            skip_health_check=skip_health_check,
            feedback_error_mode=feedback_error_mode,
        )

    @classmethod
    def from_mapping(
        cls,
        source: dict[str, Any],
        *,
        manifests: list[Path] | None = None,
        run_id: str | None = None,
        gen_url: str | None = None,
        mllm_url: str | None = None,
        max_tasks: int | None = None,
        shuffle_seed: int | None = None,
        freeze_state: bool = False,
        workers: int | None = None,
        skip_health_check: bool = False,
        feedback_error_mode: str | None = None,
    ) -> "EvolutionConfig":
        """Normalize legacy aliases and overrides exactly once at the input boundary."""
        value = json.loads(json.dumps(source))
        if int(value.get("schema_version", 1)) != 1:
            raise ValueError("unsupported evolution config schema_version")

        configured_manifests = value.get("manifests")
        if configured_manifests is None and value.get("manifest") is not None:
            configured_manifests = [value["manifest"]]
        manifest_paths = tuple(
            manifests or [Path(item) for item in (configured_manifests or [])]
        )
        if not manifest_paths:
            raise ValueError("Configure manifests or pass --manifest")

        resolved_max_tasks = (
            max_tasks if max_tasks is not None else value.get("max_tasks")
        )
        if resolved_max_tasks is not None:
            resolved_max_tasks = int(resolved_max_tasks)
            if resolved_max_tasks < 0:
                raise ValueError("max_tasks must be non-negative or null")
        checkpoint_every_tasks = int(value.get("checkpoint_every_tasks", 80))
        if checkpoint_every_tasks <= 0:
            raise ValueError("checkpoint_every_tasks must be positive")
        resolved_shuffle_seed = (
            shuffle_seed if shuffle_seed is not None else value.get("shuffle_seed")
        )
        if resolved_shuffle_seed is not None:
            resolved_shuffle_seed = int(resolved_shuffle_seed)

        generator = _section(value, "generator")
        mllm = _section(value, "mllm")
        agent = _section(value, "agent")
        verification = _section(value, "verification")
        feedback = _section(value, "feedback")
        preflight = _section(value, "preflight")
        evaluation = _section(value, "evaluation")
        memory = _section(value, "memory")
        skill = _section(value, "skill")
        variant = str(evaluation.get("variant") or "").casefold()
        default_skills, default_memory = _VARIANT_CAPABILITIES.get(
            variant,
            (True, True),
        )
        enable_skills = evaluation.get("enable_skills", default_skills)
        enable_memory = evaluation.get("enable_memory", default_memory)
        if not isinstance(enable_skills, bool):
            raise ValueError("evaluation.enable_skills must be a boolean")
        if not isinstance(enable_memory, bool):
            raise ValueError("evaluation.enable_memory must be a boolean")
        first_trajectory_run = evaluation.get("first_trajectory_run")
        if first_trajectory_run is not None:
            if not isinstance(first_trajectory_run, str) or not first_trajectory_run.strip():
                raise ValueError(
                    "evaluation.first_trajectory_run must be a non-empty path string"
                )
            first_trajectory_run = str(
                Path(first_trajectory_run).expanduser().resolve()
            )
            if not (enable_skills and enable_memory):
                raise ValueError(
                    "evaluation.first_trajectory_run requires Skills and Memory enabled"
                )
        embedding = memory.get("embedding") or {}
        if not isinstance(embedding, dict):
            raise ValueError("memory.embedding config must be a JSON object")
        semantic_threshold = float(memory.get("semantic_threshold", 0.35))
        if not -1.0 <= semantic_threshold <= 1.0:
            raise ValueError("memory.semantic_threshold must be between -1 and 1")
        duplicate_threshold = float(memory.get("duplicate_threshold", 0.92))
        if not -1.0 <= duplicate_threshold <= 1.0:
            raise ValueError("memory.duplicate_threshold must be between -1 and 1")
        retrieval_strategy = str(
            memory.get("retrieval_strategy") or "embedding"
        ).strip().casefold()
        if retrieval_strategy not in {"embedding", "embedding_evidence"}:
            raise ValueError(
                "memory.retrieval_strategy must be 'embedding' or "
                "'embedding_evidence'"
            )
        episode_summary = memory.get("episode_summary", False)
        if not isinstance(episode_summary, bool):
            raise ValueError("memory.episode_summary must be a boolean")
        embedding_preload = embedding.get("preload", False)
        if not isinstance(embedding_preload, bool):
            raise ValueError("memory.embedding.preload must be a boolean")
        memory = {
            **memory,
            "top_k": int(memory.get("top_k", 4)),
            "consolidation_top_k": int(memory.get("consolidation_top_k", 4)),
            "semantic_threshold": semantic_threshold,
            "duplicate_threshold": duplicate_threshold,
            "retrieval_strategy": retrieval_strategy,
            "episode_summary": episode_summary,
            "embedding": {
                "provider": str(embedding.get("provider") or "transformers"),
                "model": str(
                    embedding.get("model") or "BAAI/bge-m3"
                ),
                "device": str(embedding.get("device") or "cpu"),
                "pooling": str(embedding.get("pooling") or "cls"),
                "batch_size": int(embedding.get("batch_size", 32)),
                "max_length": int(embedding.get("max_length", 512)),
                "query_prefix": str(embedding.get("query_prefix") or ""),
                "document_prefix": str(embedding.get("document_prefix") or ""),
                "trust_remote_code": bool(
                    embedding.get("trust_remote_code", False)
                ),
                "local_files_only": bool(
                    embedding.get("local_files_only", False)
                ),
                "preload": embedding_preload,
            },
        }
        if memory["top_k"] <= 0:
            raise ValueError("memory.top_k must be positive")
        if memory["consolidation_top_k"] <= 0:
            raise ValueError("memory.consolidation_top_k must be positive")

        resolved_gen_url = gen_url or str(
            generator.get("url") or value.get("gen_url") or ""
        )
        resolved_mllm_url = mllm_url or str(
            mllm.get("url") or value.get("mllm_url") or ""
        )
        if not resolved_gen_url or not resolved_mllm_url:
            raise ValueError("Configure gen_url and mllm_url or pass CLI overrides")

        generator_model = str(
            generator.get("model")
            or generator.get("version")
            or value.get("generator_version")
            or "unknown"
        )
        service_urls = dict(
            feedback.get("services") or value.get("feedback_services") or {}
        )
        feedback_timeout = float(
            feedback.get("timeout_seconds", value.get("feedback_timeout", 600.0))
        )
        feedback_max_attempts = int(feedback.get("max_attempts", 3))
        feedback_retry_backoff = float(feedback.get("retry_backoff_seconds", 2.0))
        resolved_error_mode = feedback_error_mode or str(
            feedback.get("on_error") or value.get("on_feedback_error") or "abort"
        )
        if resolved_error_mode not in {"abort", "skip_update"}:
            raise ValueError("feedback.on_error must be 'abort' or 'skip_update'")
        feedback_mode = str(feedback.get("evolution_input") or "both").casefold()
        if feedback_mode not in {"text", "reward", "both"}:
            raise ValueError(
                "feedback.evolution_input must be 'text', 'reward', or 'both'"
            )
        reward_threshold = float(feedback.get("reward_threshold", 0.5))
        if not 0.0 <= reward_threshold <= 1.0:
            raise ValueError("feedback.reward_threshold must be between 0 and 1")

        resolved_freeze_state = freeze_state or bool(value.get("freeze_state", False))
        if first_trajectory_run is not None and not resolved_freeze_state:
            raise ValueError(
                "evaluation.first_trajectory_run is only supported for frozen evaluation"
            )
        task_workers = resolve_task_workers(
            freeze_state=resolved_freeze_state,
            cli_workers=workers,
            evaluation_config=evaluation,
        )
        batch_size = int(evaluation.get("batch_size", 8))
        if batch_size <= 0:
            raise ValueError("evaluation.batch_size must be positive")
        resolved_skip_health_check = (
            skip_health_check
            or bool(value.get("skip_health_check", False))
            or not bool(preflight.get("enabled", True))
        )
        run_root = Path(value.get("run_root") or "runs").expanduser().resolve()

        # Store only the normalized form so every downstream consumer observes the
        # same values that are written to effective_config.json.
        resolved_run_id = run_id or str(value.get("run_id") or "evolution-run")
        normalized = {
            **value,
            "run_id": resolved_run_id,
            "run_root": str(run_root),
            "manifests": [str(path) for path in manifest_paths],
            "max_tasks": resolved_max_tasks,
            "checkpoint_every_tasks": checkpoint_every_tasks,
            "shuffle_seed": resolved_shuffle_seed,
            "round_id": int(value.get("round_id", 0)),
            "freeze_state": resolved_freeze_state,
            "skip_health_check": resolved_skip_health_check,
            "agent": agent,
            "verification": verification,
            "evaluation": {
                **evaluation,
                "workers": task_workers,
                "batch_size": batch_size,
                "enable_skills": enable_skills,
                "enable_memory": enable_memory,
                **(
                    {"first_trajectory_run": first_trajectory_run}
                    if first_trajectory_run is not None
                    else {}
                ),
            },
            "memory": memory,
            "skill": skill,
            "preflight": {
                **preflight,
                "enabled": not resolved_skip_health_check,
            },
            "generator": {
                **generator,
                "url": resolved_gen_url,
                "model": generator_model,
            },
            "mllm": {**mllm, "url": resolved_mllm_url},
            "feedback": {
                **feedback,
                "services": service_urls,
                "timeout_seconds": feedback_timeout,
                "max_attempts": feedback_max_attempts,
                "retry_backoff_seconds": feedback_retry_backoff,
                "on_error": resolved_error_mode,
                "evolution_input": feedback_mode,
                "reward_threshold": reward_threshold,
            },
        }
        normalized.pop("manifest", None)
        return cls(
            data=normalized,
            manifest_paths=manifest_paths,
            run_id=resolved_run_id,
            run_root=run_root,
            round_id=int(normalized["round_id"]),
            freeze_state=resolved_freeze_state,
            skip_health_check=resolved_skip_health_check,
            max_tasks=resolved_max_tasks,
            shuffle_seed=resolved_shuffle_seed,
            generator=normalized["generator"],
            mllm=normalized["mllm"],
            agent=agent,
            verification=verification,
            feedback=normalized["feedback"],
            preflight=normalized["preflight"],
            evaluation=normalized["evaluation"],
            memory=memory,
            skill=skill,
            generator_url=resolved_gen_url,
            mllm_url=resolved_mllm_url,
            generator_model=generator_model,
            service_urls=service_urls,
            feedback_timeout=feedback_timeout,
            feedback_max_attempts=feedback_max_attempts,
            feedback_retry_backoff=feedback_retry_backoff,
            feedback_error_mode=resolved_error_mode,
            feedback_mode=feedback_mode,
            reward_threshold=reward_threshold,
            task_workers=task_workers,
            batch_size=batch_size,
            checkpoint_every_tasks=checkpoint_every_tasks,
            enable_skills=enable_skills,
            enable_memory=enable_memory,
        )

    @property
    def run_dir(self) -> Path:
        return self.run_root / self.run_id

    @property
    def state_dir(self) -> Path:
        return self.run_dir / "evolution"

    def load_tasks(self) -> list[dict[str, Any]]:
        """Load, de-duplicate, order, and cap the configured task stream."""
        tasks: list[dict[str, Any]] = []
        seen: set[str] = set()
        for path in self.manifest_paths:
            with path.open("r", encoding="utf-8") as stream:
                for line_number, line in enumerate(stream, 1):
                    if not line.strip():
                        continue
                    value = json.loads(line)
                    if not isinstance(value, dict):
                        raise ValueError(f"Expected object at {path}:{line_number}")
                    task_id = value.get("sample_id")
                    benchmark = value.get("benchmark")
                    prompt = value.get("prompt")
                    if not all(
                        isinstance(item, str) and item
                        for item in (task_id, benchmark, prompt)
                    ):
                        raise ValueError(f"Invalid task record at {path}:{line_number}")
                    if task_id in seen:
                        raise ValueError(f"Duplicate sample_id in task stream: {task_id}")
                    seen.add(task_id)
                    tasks.append(value)
        if self.shuffle_seed is not None:
            random.Random(self.shuffle_seed).shuffle(tasks)
        return tasks[: self.max_tasks] if self.max_tasks is not None else tasks

    def validate_feedback_services(self, tasks: Iterable[dict[str, Any]]) -> list[str]:
        required = sorted({str(task["benchmark"]) for task in tasks})
        missing = [name for name in required if not self.service_urls.get(name)]
        if missing:
            raise ValueError(f"Missing feedback service URLs: {missing}")
        return required

    def to_dict(self) -> dict[str, Any]:
        return json.loads(json.dumps(self.data))

    def effective_dict(self) -> dict[str, Any]:
        value = self.to_dict()
        value["manifest_sha256"] = {
            str(path): hashlib.sha256(path.read_bytes()).hexdigest()
            for path in self.manifest_paths
        }
        return value

    def require_checkpoint(self) -> tuple[Path, Path]:
        metadata = self.state_dir / "metadata.json"
        if not metadata.is_file():
            raise FileNotFoundError(
                f"source evolution state is missing: {self.state_dir}"
            )
        return self.run_dir, self.state_dir

    def held_out_manifests(
        self,
        *,
        split: str,
        explicit: list[Path] | None,
    ) -> list[Path]:
        manifests = (
            [path.expanduser().resolve() for path in explicit]
            if explicit
            else [
                path.with_name(f"{split}.jsonl").expanduser().resolve()
                for path in self.manifest_paths
            ]
        )
        missing = [str(path) for path in manifests if not path.is_file()]
        if missing:
            raise FileNotFoundError(f"evaluation manifests are missing: {missing}")
        return manifests

    def frozen_variant(
        self,
        *,
        variant: str,
        run_root: Path,
        manifests: list[Path],
        max_tasks: int | None,
        workers: int,
        batch_size: int | None = None,
        first_trajectory_run: Path | None = None,
    ) -> "EvolutionConfig":
        """Derive an isolated, non-learning evaluation config from this checkpoint."""
        if variant not in {
            "evolved",
            "empty",
            "full",
            "baseline",
            "memory_only",
            "skill_only",
        }:
            raise ValueError(f"unsupported frozen evaluation variant: {variant}")
        value = self.to_dict()
        if variant in {
            "evolved",
            "full",
            "baseline",
            "memory_only",
            "skill_only",
        }:
            # State-backed variants live under separate variant roots, so they can
            # retain the checkpoint run_id recorded in metadata.json.
            value["run_id"] = self.run_id
        elif variant == "empty":
            value["run_id"] = f"{self.run_id}-empty"
        value["run_root"] = str(run_root.resolve())
        value["manifests"] = [str(path) for path in manifests]
        value["max_tasks"] = max_tasks
        value["freeze_state"] = True
        enable_skills, enable_memory = _VARIANT_CAPABILITIES[variant]
        evaluation = {
            **self.evaluation,
            "workers": workers,
            "batch_size": (
                int(batch_size) if batch_size is not None else self.batch_size
            ),
            "variant": variant,
            "enable_skills": enable_skills,
            "enable_memory": enable_memory,
        }
        evaluation.pop("first_trajectory_run", None)
        if first_trajectory_run is not None:
            if not (enable_skills and enable_memory):
                raise ValueError(
                    "first-trajectory replay requires a full/evolved variant"
                )
            evaluation["first_trajectory_run"] = str(
                first_trajectory_run.expanduser().resolve()
            )
        value["evaluation"] = evaluation
        value["feedback"] = {**self.feedback, "on_error": "abort"}
        if variant == "empty":
            value["skill"] = {**self.skill, "initialization": "empty"}
        if not enable_memory:
            # Memory-disabled variants never retrieve Insights. Keep the embedder
            # lazy while retaining the complete, unmodified checkpoint snapshot.
            value["memory"] = {
                **self.memory,
                "embedding": {
                    **dict(self.memory.get("embedding") or {}),
                    "preload": False,
                },
            }
        return self.from_mapping(value)

    def configure_no_proxy(self) -> None:
        """Keep configured private services from being accidentally sent to a proxy."""
        urls = [
            self.generator_url,
            *[str(value) for value in self.service_urls.values()],
        ]
        hosts = [urlsplit(url).hostname for url in urls if urlsplit(url).hostname]
        existing = os.environ.get("NO_PROXY") or os.environ.get("no_proxy") or ""
        values = list(
            dict.fromkeys([*hosts, "127.0.0.1", "localhost", *existing.split(",")])
        )
        no_proxy = ",".join(value for value in values if value)
        os.environ["NO_PROXY"] = no_proxy
        os.environ["no_proxy"] = no_proxy


__all__ = ["EvolutionConfig", "resolve_task_workers"]

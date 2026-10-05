#!/usr/bin/env python3
"""Snapshot a GEMS checkpoint and evaluate it concurrently without learning."""

from __future__ import annotations

import argparse
import json
import re
import shutil
import uuid
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from runners.config import EvolutionConfig
from runners.evolve_stream import main as run_stream


STATE_VARIANTS = frozenset(
    {"evolved", "full", "baseline", "memory_only", "skill_only"}
)
ALL_VARIANTS = ("baseline", "memory_only", "skill_only", "full")
ABLATION_IMPLEMENTATION = "runtime_capabilities_v1"


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/evolution.geneval.train.json"),
        help="Evolution config whose run_id and services should be evaluated.",
    )
    parser.add_argument(
        "--split",
        choices=["validation", "test"],
        default="validation",
        help="Replace each configured manifest filename with this split.",
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        action="append",
        help="Explicit held-out manifest; repeat for multiple benchmarks.",
    )
    parser.add_argument(
        "--variant",
        choices=[
            "evolved",
            "empty",
            "both",
            "full",
            "baseline",
            "memory_only",
            "skill_only",
            "all",
        ],
        default="evolved",
        help=(
            "Evaluate one checkpoint variant; 'both' preserves the legacy "
            "evolved/empty pair and 'all' runs baseline plus both ablations and full."
        ),
    )
    parser.add_argument(
        "--checkpoint",
        default="live",
        help=(
            "Evolution-state source: 'live' (current run state), 'latest' "
            "(highest saved milestone), a task count such as '160', a saved "
            "directory name such as 'tasks_000160', or an explicit checkpoint path."
        ),
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=4,
        help="Concurrent task workers (each task may also verify in parallel).",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        help=(
            "Tasks admitted to each frozen-evaluation batch. Defaults to the "
            "source config; set it at least as high as --workers to use all workers."
        ),
    )
    parser.add_argument(
        "--max-tasks",
        type=int,
        help="Optional smoke-test cap; omitted means the complete split.",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("runs/evaluations"),
    )
    parser.add_argument(
        "--name",
        help="Optional evaluation-suite directory name.",
    )
    parser.add_argument(
        "--preflight-only",
        action="store_true",
        help="Prepare the snapshot/config and check services without evaluating.",
    )
    parser.add_argument(
        "--first-trajectory-run",
        type=Path,
        help=(
            "Skill-only run directory whose saved Skill selection, initial prompt, "
            "first image, checks, and experience are replayed before Full branches "
            "into Insight-backed refinement. Valid with --variant full/evolved."
        ),
    )
    return parser.parse_args(argv)


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def _record_ablation_implementation(
    suite_dir: Path,
    variants: list[str],
) -> None:
    """Prevent old state-pruning results from mixing with runtime-switch results."""
    if not any(variant in ALL_VARIANTS for variant in variants):
        return
    path = suite_dir / "ablation_implementation.json"
    expected = {
        "schema_version": 1,
        "implementation": ABLATION_IMPLEMENTATION,
    }
    if path.is_file():
        current = _json_object(path, label="ablation implementation")
        if current != expected:
            raise RuntimeError(
                f"evaluation suite {suite_dir} uses a different ablation "
                "implementation; choose a new --name"
            )
        return

    legacy = suite_dir / "checkpoints.json"
    if legacy.is_file():
        records = _json_object(legacy, label="legacy ablation checkpoints")
        compatible = all(
            isinstance(records.get(variant), dict)
            and records[variant].get("removed_state") == []
            and isinstance(records[variant].get("runtime_capabilities"), dict)
            for variant in variants
            if variant in ALL_VARIANTS
        )
        if not compatible:
            raise RuntimeError(
                f"evaluation suite {suite_dir} contains results from the former "
                "state-pruning ablation; choose a new --name"
            )
    _write_json(path, expected)


def _safe_component(value: str) -> str:
    safe = re.sub(r"[^A-Za-z0-9._-]+", "-", value).strip("-.")
    if not safe:
        raise ValueError("evaluation name must contain a safe path character")
    return safe[:180]


def _checkpoint_size(state_dir: Path) -> int:
    episodes = state_dir / "episodes"
    return len(list(episodes.glob("*.json"))) if episodes.is_dir() else 0


def _json_object(path: Path, *, label: str) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(f"{label} is missing: {path}")
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as error:
        raise ValueError(f"{label} is invalid JSON: {path}") from error
    if not isinstance(value, dict):
        raise ValueError(f"{label} must be a JSON object: {path}")
    return value


def _checkpoint_directory(run_dir: Path, selector: str) -> Path:
    value = str(selector).strip()
    checkpoint_root = run_dir / "checkpoints"
    if value.casefold() == "latest":
        candidates: list[tuple[int, Path]] = []
        if checkpoint_root.is_dir():
            for path in checkpoint_root.glob("tasks_*"):
                match = re.fullmatch(r"tasks_(\d+)", path.name)
                if path.is_dir() and match and (path / "checkpoint.json").is_file():
                    candidates.append((int(match.group(1)), path))
        if not candidates:
            raise FileNotFoundError(
                f"no saved checkpoints are available under {checkpoint_root}"
            )
        return max(candidates, key=lambda item: item[0])[1].resolve()
    if value.isdigit():
        return (checkpoint_root / f"tasks_{int(value):06d}").resolve()
    if re.fullmatch(r"tasks_\d+", value):
        return (checkpoint_root / value).resolve()
    return Path(value).expanduser().resolve()


def _resolve_checkpoint(
    config: EvolutionConfig,
    selector: str,
) -> tuple[Path, Path, dict[str, Any]]:
    """Resolve and validate one live or versioned evolution-state source."""
    value = str(selector or "live").strip() or "live"
    run_dir = config.run_dir.resolve()
    if value.casefold() == "live":
        _, state_dir = config.require_checkpoint()
        state_dir = state_dir.resolve()
        state_metadata = _json_object(
            state_dir / "metadata.json",
            label="live evolution metadata",
        )
        state_run_id = str(state_metadata.get("run_id") or "")
        if state_run_id and state_run_id != config.run_id:
            raise ValueError(
                f"live evolution run_id {state_run_id!r} does not match "
                f"config run_id {config.run_id!r}"
            )
        return run_dir, state_dir, {
            "selector": value,
            "kind": "live",
            "run_id": config.run_id,
            "checkpoint_dir": None,
            "state_dir": str(state_dir),
            "milestone_tasks": None,
            "committed_tasks": None,
            "batch_id": None,
            "created_at": None,
        }

    checkpoint_dir = _checkpoint_directory(run_dir, value)
    checkpoint = _json_object(
        checkpoint_dir / "checkpoint.json",
        label="checkpoint metadata",
    )
    checkpoint_run_id = str(checkpoint.get("run_id") or "")
    if checkpoint_run_id != config.run_id:
        raise ValueError(
            f"checkpoint run_id {checkpoint_run_id!r} does not match "
            f"config run_id {config.run_id!r}"
        )
    relative_state = Path(str(checkpoint.get("state_dir") or "evolution"))
    if relative_state.is_absolute():
        raise ValueError("checkpoint state_dir must be relative to its checkpoint")
    state_dir = (checkpoint_dir / relative_state).resolve()
    if checkpoint_dir not in state_dir.parents:
        raise ValueError("checkpoint state_dir escapes its checkpoint directory")
    state_metadata = _json_object(
        state_dir / "metadata.json",
        label="checkpoint evolution metadata",
    )
    state_run_id = str(state_metadata.get("run_id") or "")
    if state_run_id and state_run_id != config.run_id:
        raise ValueError(
            f"checkpoint evolution run_id {state_run_id!r} does not match "
            f"config run_id {config.run_id!r}"
        )
    milestone = int(checkpoint.get("milestone_tasks") or 0)
    committed = int(checkpoint.get("committed_tasks") or 0)
    if milestone <= 0 or committed < milestone:
        raise ValueError(
            f"invalid checkpoint task counts in {checkpoint_dir / 'checkpoint.json'}"
        )
    return run_dir, state_dir, {
        "selector": value,
        "kind": "saved_checkpoint",
        "run_id": config.run_id,
        "checkpoint_dir": str(checkpoint_dir),
        "state_dir": str(state_dir),
        "milestone_tasks": milestone,
        "committed_tasks": committed,
        "batch_id": int(checkpoint.get("batch_id") or 0),
        "created_at": checkpoint.get("created_at"),
    }


def _record_checkpoint_selection(
    suite_dir: Path,
    selection: dict[str, Any],
) -> None:
    """Pin a resumable suite to one source state and reject selector drift."""
    path = suite_dir / "checkpoint_selection.json"
    if path.is_file():
        existing = _json_object(path, label="evaluation checkpoint selection")
        identity_fields = (
            "kind",
            "run_id",
            "checkpoint_dir",
            "state_dir",
            "milestone_tasks",
            "created_at",
            "episodes",
        )
        if any(existing.get(key) != selection.get(key) for key in identity_fields):
            raise RuntimeError(
                f"evaluation suite {suite_dir} is already pinned to a different "
                "checkpoint; choose a new --name"
            )
        return
    legacy = suite_dir / "checkpoints.json"
    if legacy.is_file():
        records = _json_object(legacy, label="legacy evaluation checkpoints")
        source_states = {
            str(item.get("source_state"))
            for item in records.values()
            if isinstance(item, dict) and item.get("source_state")
        }
        if source_states and source_states != {str(selection["state_dir"])}:
            raise RuntimeError(
                f"evaluation suite {suite_dir} already contains a snapshot from a "
                "different checkpoint; choose a new --name"
            )
    _write_json(path, selection)


def _backup_state(source: Path, destination: Path) -> bool:
    """Copy one immutable file-state snapshot; return False when resuming one."""
    if destination.exists():
        return False
    if not source.is_dir():
        raise FileNotFoundError(source)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(f".{destination.name}.{uuid.uuid4().hex}.tmp")
    shutil.copytree(source, temporary)
    temporary.replace(destination)
    return True


def _load_evaluations(run_dir: Path) -> list[dict[str, Any]]:
    path = run_dir / "private_evaluation.jsonl"
    if not path.is_file():
        raise FileNotFoundError(f"evaluation output is missing: {path}")
    text = path.read_text(encoding="utf-8")
    records = []
    decoder = json.JSONDecoder()
    offset = 0
    while offset < len(text):
        while offset < len(text) and text[offset].isspace():
            offset += 1
        if offset >= len(text):
            break
        value, offset = decoder.raw_decode(text, offset)
        if not isinstance(value, dict):
            raise ValueError(f"invalid evaluation record in {path}")
        records.append(value)
    return records


def _call_totals(run_dir: Path) -> tuple[int, int]:
    generator_calls = 0
    mllm_calls = 0
    for path in (run_dir / "results").glob("*.json"):
        value = json.loads(path.read_text(encoding="utf-8"))
        generator_calls += int(value.get("generator_calls") or 0)
        mllm_calls += int(value.get("mllm_calls") or 0)
    return generator_calls, mllm_calls


def _summary(run_dir: Path, *, variant: str) -> dict[str, Any]:
    records = _load_evaluations(run_dir)
    if not records:
        raise RuntimeError(f"evaluation produced no records: {run_dir}")
    by_tag: dict[str, list[float]] = defaultdict(list)
    scores: list[float] = []
    binary = True
    for value in records:
        evaluator = dict(value.get("evaluator_result") or {})
        correct = evaluator.get("correct")
        if isinstance(correct, bool):
            score = float(correct)
        else:
            reward = value.get("reward")
            if isinstance(reward, bool) or not isinstance(reward, (int, float)):
                raise RuntimeError(
                    "frozen summary requires evaluator_result.correct or numeric reward"
                )
            score = float(reward)
            binary = False
        scores.append(score)
        metadata = dict(value.get("metadata") or {})
        tag = str(evaluator.get("tag") or metadata.get("tag") or "all")
        by_tag[tag].append(score)
    generator_calls, mllm_calls = _call_totals(run_dir)
    result = {
        "variant": variant,
        "run_dir": str(run_dir.resolve()),
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "tasks": len(scores),
        "metric_name": "accuracy" if binary else "mean_reward",
        "score": sum(scores) / len(scores),
        "accuracy": sum(scores) / len(scores),
        "generator_calls": generator_calls,
        "mllm_calls": mllm_calls,
        "generator_calls_per_task": generator_calls / len(scores),
        "mllm_calls_per_task": mllm_calls / len(scores),
        "by_tag": {
            tag: {
                "tasks": len(values),
                "score": sum(values) / len(values),
                "accuracy": sum(values) / len(values),
            }
            for tag, values in sorted(by_tag.items())
        },
    }
    if binary:
        result["correct"] = int(sum(scores))
        for values in result["by_tag"].values():
            values["correct"] = int(round(values["score"] * values["tasks"]))
    else:
        result["mean_reward"] = result["score"]
        for values in result["by_tag"].values():
            values["mean_reward"] = values["score"]
    _write_json(run_dir / "evaluation_summary.json", result)
    return result


def _comparison(
    evolved_dir: Path,
    empty_dir: Path,
    evolved_summary: dict[str, Any],
    empty_summary: dict[str, Any],
) -> dict[str, Any]:
    def indexed(run_dir: Path) -> dict[str, dict[str, Any]]:
        return {
            str(value["sample_id"]): value
            for value in _load_evaluations(run_dir)
        }

    evolved = indexed(evolved_dir)
    empty = indexed(empty_dir)
    if set(evolved) != set(empty):
        raise RuntimeError("evolved and empty variants evaluated different sample sets")
    def score(record: dict[str, Any]) -> float:
        evaluator = dict(record.get("evaluator_result") or {})
        correct = evaluator.get("correct")
        if isinstance(correct, bool):
            return float(correct)
        return float(record["reward"])

    evolved_wins = 0
    empty_wins = 0
    ties = 0
    by_tag: dict[str, list[str]] = defaultdict(list)
    for sample_id in evolved:
        evolved_correct = score(evolved[sample_id])
        empty_correct = score(empty[sample_id])
        evaluator = dict(evolved[sample_id].get("evaluator_result") or {})
        metadata = dict(evolved[sample_id].get("metadata") or {})
        tag = str(evaluator.get("tag") or metadata.get("tag") or "all")
        by_tag[tag].append(sample_id)
        if evolved_correct > empty_correct:
            evolved_wins += 1
        elif empty_correct > evolved_correct:
            empty_wins += 1
        else:
            ties += 1
    def paired_metrics(sample_ids: list[str]) -> dict[str, Any]:
        evolved_scores = [
            score(evolved[sample_id])
            for sample_id in sample_ids
        ]
        empty_scores = [
            score(empty[sample_id])
            for sample_id in sample_ids
        ]
        return {
            "tasks": len(sample_ids),
            "evolved_accuracy": sum(evolved_scores) / len(sample_ids),
            "empty_accuracy": sum(empty_scores) / len(sample_ids),
            "accuracy_delta": (
                sum(evolved_scores) - sum(empty_scores)
            ) / len(sample_ids),
            "evolved_wins": sum(
                evolved_score > empty_score
                for evolved_score, empty_score in zip(evolved_scores, empty_scores)
            ),
            "empty_wins": sum(
                empty_score > evolved_score
                for evolved_score, empty_score in zip(evolved_scores, empty_scores)
            ),
        }

    return {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "tasks": len(evolved),
        "evolved_accuracy": evolved_summary["accuracy"],
        "empty_accuracy": empty_summary["accuracy"],
        "accuracy_delta": evolved_summary["accuracy"] - empty_summary["accuracy"],
        "evolved_wins": evolved_wins,
        "empty_wins": empty_wins,
        "ties": ties,
        "evolved_generator_calls_per_task": evolved_summary[
            "generator_calls_per_task"
        ],
        "empty_generator_calls_per_task": empty_summary[
            "generator_calls_per_task"
        ],
        "evolved_mllm_calls_per_task": evolved_summary["mllm_calls_per_task"],
        "empty_mllm_calls_per_task": empty_summary["mllm_calls_per_task"],
        "by_tag": {
            tag: paired_metrics(sample_ids)
            for tag, sample_ids in sorted(by_tag.items())
        },
    }


def _paired_comparison(
    *,
    left_variant: str,
    right_variant: str,
    left_dir: Path,
    right_dir: Path,
    left_summary: dict[str, Any],
    right_summary: dict[str, Any],
) -> dict[str, Any]:
    """Compare two variants sample by sample, preserving direction in the result."""

    def indexed(run_dir: Path) -> dict[str, dict[str, Any]]:
        return {
            str(value["sample_id"]): value
            for value in _load_evaluations(run_dir)
        }

    left = indexed(left_dir)
    right = indexed(right_dir)
    if set(left) != set(right):
        raise RuntimeError(
            f"{left_variant} and {right_variant} evaluated different sample sets"
        )

    def score(values: dict[str, dict[str, Any]], sample_id: str) -> float:
        record = values[sample_id]
        evaluator = dict(record.get("evaluator_result") or {})
        correct = evaluator.get("correct")
        if isinstance(correct, bool):
            return float(correct)
        return float(record["reward"])

    sample_ids = sorted(left)
    tags: dict[str, list[str]] = defaultdict(list)
    for sample_id in sample_ids:
        evaluator = dict(left[sample_id].get("evaluator_result") or {})
        metadata = dict(left[sample_id].get("metadata") or {})
        tag = str(evaluator.get("tag") or metadata.get("tag") or "all")
        tags[tag].append(sample_id)

    def metrics(selected: list[str]) -> dict[str, Any]:
        left_scores = [score(left, sample_id) for sample_id in selected]
        right_scores = [score(right, sample_id) for sample_id in selected]
        return {
            "tasks": len(selected),
            "left_accuracy": sum(left_scores) / len(selected),
            "right_accuracy": sum(right_scores) / len(selected),
            "accuracy_delta": (
                sum(left_scores) - sum(right_scores)
            ) / len(selected),
            "left_wins": sum(
                left_score > right_score
                for left_score, right_score in zip(left_scores, right_scores)
            ),
            "right_wins": sum(
                right_score > left_score
                for left_score, right_score in zip(left_scores, right_scores)
            ),
            "ties": sum(
                left_score == right_score
                for left_score, right_score in zip(left_scores, right_scores)
            ),
        }

    return {
        "left_variant": left_variant,
        "right_variant": right_variant,
        **metrics(sample_ids),
        "left_generator_calls_per_task": left_summary["generator_calls_per_task"],
        "right_generator_calls_per_task": right_summary["generator_calls_per_task"],
        "left_mllm_calls_per_task": left_summary["mllm_calls_per_task"],
        "right_mllm_calls_per_task": right_summary["mllm_calls_per_task"],
        "by_tag": {
            tag: metrics(selected)
            for tag, selected in sorted(tags.items())
        },
    }


def _ablation_comparison(
    prepared: dict[str, tuple[EvolutionConfig, Path, Path]],
    summaries: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    """Produce baseline and full-centered paired comparisons for four variants."""
    required = set(ALL_VARIANTS)
    if set(summaries) != required:
        raise ValueError("ablation comparison requires all four evaluation variants")
    pairs = (
        ("memory_only", "baseline"),
        ("skill_only", "baseline"),
        ("full", "baseline"),
        ("full", "memory_only"),
        ("full", "skill_only"),
    )
    return {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "tasks": summaries["baseline"]["tasks"],
        "variant_order": list(ALL_VARIANTS),
        "accuracies": {
            variant: summaries[variant]["accuracy"]
            for variant in ALL_VARIANTS
        },
        "pairs": {
            f"{left}_vs_{right}": _paired_comparison(
                left_variant=left,
                right_variant=right,
                left_dir=prepared[left][2],
                right_dir=prepared[right][2],
                left_summary=summaries[left],
                right_summary=summaries[right],
            )
            for left, right in pairs
        },
    }


def main(argv: list[str] | None = None) -> int:
    """Snapshot once, run isolated variants, then summarize paired outcomes."""
    args = parse_args(argv)
    if args.workers <= 0:
        raise ValueError("--workers must be positive")
    if args.batch_size is not None and args.batch_size <= 0:
        raise ValueError("--batch-size must be positive when provided")
    if args.max_tasks is not None and args.max_tasks <= 0:
        raise ValueError("--max-tasks must be positive when provided")
    if args.first_trajectory_run is not None and args.variant not in {
        "full",
        "evolved",
    }:
        raise ValueError(
            "--first-trajectory-run requires --variant full or evolved"
        )

    source_config = EvolutionConfig.load(args.config)
    source_run_dir, source_state, checkpoint_selection = _resolve_checkpoint(
        source_config,
        args.checkpoint,
    )
    checkpoint_size = _checkpoint_size(source_state)
    checkpoint_selection = {
        **checkpoint_selection,
        "episodes": checkpoint_size,
    }
    manifests = source_config.held_out_manifests(
        split=args.split,
        explicit=args.manifest,
    )
    suite_name = _safe_component(
        args.name or f"{source_config.run_id}-evo{checkpoint_size}-{args.split}"
    )
    suite_dir = args.output_root.expanduser().resolve() / suite_name
    suite_dir.mkdir(parents=True, exist_ok=True)
    _record_checkpoint_selection(suite_dir, checkpoint_selection)
    print(
        "[checkpoint] "
        f"selector={args.checkpoint} kind={checkpoint_selection['kind']} "
        f"episodes={checkpoint_size} state={source_state}",
        flush=True,
    )
    if args.variant == "both":
        variants = ["evolved", "empty"]
    elif args.variant == "all":
        variants = list(ALL_VARIANTS)
    else:
        variants = [args.variant]
    _record_ablation_implementation(suite_dir, variants)

    prepared: dict[str, tuple[EvolutionConfig, Path, Path]] = {}
    checkpoints: dict[str, dict[str, Any]] = {}

    # Materialize every derived config before evaluation starts so resumed suites use
    # the exact same checkpoint, manifests, and isolation boundaries.
    for variant in variants:
        variant_root = suite_dir / variant
        value = source_config.frozen_variant(
            variant=variant,
            run_root=variant_root,
            manifests=manifests,
            max_tasks=args.max_tasks,
            workers=args.workers,
            batch_size=args.batch_size,
            first_trajectory_run=(
                args.first_trajectory_run
                if variant in {"full", "evolved"}
                else None
            ),
        )
        config_path = suite_dir / f"{variant}.config.json"
        _write_json(config_path, value.to_dict())
        run_dir = value.run_dir
        prepared[variant] = (value, config_path, run_dir)

        if variant in STATE_VARIANTS:
            # Evaluate an immutable copy rather than the live training directory so a
            # resumed training process cannot change the checkpoint mid-comparison.
            destination = run_dir / "evolution"
            created = _backup_state(source_state, destination)
            snapshot_size = _checkpoint_size(destination)
            if snapshot_size != checkpoint_size:
                raise RuntimeError(
                    f"snapshot expected evo{checkpoint_size}, got evo{snapshot_size}"
                )
            checkpoints[variant] = {
                "source_run_id": source_config.run_id,
                "source_run_dir": str(source_run_dir),
                "source_state": str(source_state),
                "checkpoint_selector": args.checkpoint,
                "source_checkpoint": checkpoint_selection.get("checkpoint_dir"),
                "checkpoint_milestone_tasks": checkpoint_selection.get(
                    "milestone_tasks"
                ),
                "checkpoint_episodes": snapshot_size,
                "source_episodes_at_invocation": checkpoint_size,
                "snapshot_state": str(destination),
                "snapshot_created": created,
                "removed_state": [],
                "runtime_capabilities": {
                    "enable_skills": value.enable_skills,
                    "enable_memory": value.enable_memory,
                },
                "first_trajectory_run": value.evaluation.get(
                    "first_trajectory_run"
                ),
            }
        else:
            checkpoints[variant] = {
                "source_run_id": source_config.run_id,
                "source_run_dir": str(source_run_dir),
                "source_state": str(source_state),
                "checkpoint_selector": args.checkpoint,
                "source_checkpoint": checkpoint_selection.get("checkpoint_dir"),
                "checkpoint_milestone_tasks": checkpoint_selection.get(
                    "milestone_tasks"
                ),
                "checkpoint_episodes": checkpoint_size,
                "snapshot_state": None,
                "snapshot_created": False,
                "removed_state": [],
                "runtime_capabilities": {
                    "enable_skills": value.enable_skills,
                    "enable_memory": value.enable_memory,
                },
                "first_trajectory_run": value.evaluation.get(
                    "first_trajectory_run"
                ),
            }
    _write_json(suite_dir / "checkpoints.json", checkpoints)

    source_config.configure_no_proxy()
    if args.preflight_only:
        first = variants[0]
        _, config_path, _ = prepared[first]
        return run_stream(
            [
                "--config",
                str(config_path),
                "--workers",
                str(args.workers),
                "--preflight-only",
            ]
        )

    summaries: dict[str, dict[str, Any]] = {}

    # Variants run serially against the shared external services; task-level
    # concurrency remains inside run_stream and cannot mutate the frozen checkpoint.
    for variant in variants:
        variant_config, config_path, run_dir = prepared[variant]
        print(
            f"\n=== Running {variant} variant with {args.workers} task workers "
            f"and batch_size={variant_config.batch_size} ===",
            flush=True,
        )
        result = run_stream(
            ["--config", str(config_path), "--workers", str(args.workers)]
        )
        if result != 0:
            return result
        if variant_config.evaluation.get("defer_feedback", False):
            summaries[variant] = {
                "variant": variant,
                "tasks": len(list((run_dir / "results").glob("*.json"))),
                "status": "pending_official_offline_scoring",
            }
            print(f"[{variant}] generation complete; official scoring pending", flush=True)
            continue
        summaries[variant] = _summary(run_dir, variant=variant)
        print(
            f"[{variant}] accuracy={summaries[variant]['accuracy']:.4f} "
            f"tasks={summaries[variant]['tasks']}",
            flush=True,
        )

    _write_json(suite_dir / "summaries.json", summaries)
    if source_config.evaluation.get("defer_feedback", False):
        return 0
    if set(summaries) == {"evolved", "empty"}:
        # Pair by sample ID instead of comparing only aggregate accuracy; this exposes
        # whether evolution actually wins tasks or merely changes which tasks succeed.
        comparison = _comparison(
            prepared["evolved"][2],
            prepared["empty"][2],
            summaries["evolved"],
            summaries["empty"],
        )
        _write_json(suite_dir / "comparison.json", comparison)
        print(
            "[comparison] "
            f"delta={comparison['accuracy_delta']:+.4f} "
            f"wins={comparison['evolved_wins']} "
            f"losses={comparison['empty_wins']} ties={comparison['ties']}",
            flush=True,
        )
    elif set(summaries) == set(ALL_VARIANTS):
        comparison = _ablation_comparison(prepared, summaries)
        _write_json(suite_dir / "comparison.json", comparison)
        baseline = comparison["accuracies"]["baseline"]
        print("[ablation] paired deltas from baseline:", flush=True)
        for variant in ("memory_only", "skill_only", "full"):
            accuracy = comparison["accuracies"][variant]
            print(
                f"  {variant}: accuracy={accuracy:.4f} "
                f"delta={accuracy - baseline:+.4f}",
                flush=True,
            )
    print(f"Evaluation suite: {suite_dir}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

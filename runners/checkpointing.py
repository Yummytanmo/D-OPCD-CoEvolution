"""Versioned training checkpoints for file-backed evolution state."""

from __future__ import annotations

import shutil
import uuid
from pathlib import Path

from agent.evolution.storage import StateStore, utc_now


def _committed_tasks(state: StateStore) -> int:
    return sum(
        value.get("status") == "committed"
        for value in state.records("progress")
    )


def save_training_checkpoint_if_due(
    *,
    run_dir: Path,
    state: StateStore,
    every_tasks: int,
    batch_id: int,
) -> Path | None:
    """Atomically snapshot the latest complete batch at each task milestone.

    The checkpoint contains only the compact evolution state and effective config;
    generated images and other large task artifacts remain in the owning run.
    """
    interval = int(every_tasks)
    if interval <= 0:
        raise ValueError("checkpoint_every_tasks must be positive")

    committed_tasks = _committed_tasks(state)
    milestone_tasks = committed_tasks // interval * interval
    if milestone_tasks == 0:
        return None
    committed_batches = [
        int(value.get("batch_id", 0))
        for value in state.records("batch_commits")
    ]
    checkpoint_batch_id = max(committed_batches, default=int(batch_id))

    checkpoint_root = run_dir / "checkpoints"
    destination = checkpoint_root / f"tasks_{milestone_tasks:06d}"
    if destination.exists():
        return None

    checkpoint_root.mkdir(parents=True, exist_ok=True)
    temporary = checkpoint_root / f".{destination.name}.{uuid.uuid4().hex}.tmp"
    try:
        temporary.mkdir()
        shutil.copytree(state.path, temporary / "evolution")
        effective_config = run_dir / "effective_config.json"
        if effective_config.is_file():
            shutil.copy2(effective_config, temporary / "effective_config.json")
        StateStore.save_json(
            temporary / "checkpoint.json",
            {
                "schema_version": 1,
                "run_id": state.run_id,
                "round_id": state.round_id,
                "milestone_tasks": milestone_tasks,
                "committed_tasks": committed_tasks,
                "batch_id": checkpoint_batch_id,
                "state_dir": "evolution",
                "created_at": utc_now(),
            },
        )
        try:
            temporary.replace(destination)
        except FileExistsError:
            return None
    finally:
        if temporary.exists():
            shutil.rmtree(temporary)
    return destination


__all__ = ["save_training_checkpoint_if_due"]

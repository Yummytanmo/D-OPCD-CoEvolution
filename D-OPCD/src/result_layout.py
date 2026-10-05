"""Canonical filesystem layout for D-OPCD runtime results."""

from __future__ import annotations

from pathlib import Path
import os
import re
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]
RESULTS_ROOT = Path(os.environ.get("DOPCD_RESULTS_ROOT", str(PROJECT_ROOT / "results"))).expanduser().resolve()
_IDENTIFIER = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]*$")


def validate_identifier(value: str, label: str) -> str:
    """Return a safe path component or raise a descriptive error."""
    normalized = str(value).strip()
    if not _IDENTIFIER.fullmatch(normalized):
        raise ValueError(
            f"{label} must match {_IDENTIFIER.pattern!r}; got {value!r}"
        )
    return normalized


def task_run_id(task: dict[str, Any]) -> str:
    value = task.get("run_id") or task.get("task_name")
    if not value:
        raise ValueError("Task config must define run_id")
    return validate_identifier(str(value), "run_id")


def run_root(run_id: str) -> Path:
    return RESULTS_ROOT / "runs" / validate_identifier(run_id, "run_id")


def training_dir(run_id: str) -> Path:
    return run_root(run_id) / "training"


def evaluations_dir(run_id: str) -> Path:
    return run_root(run_id) / "evaluations"


def evaluation_dir(run_id: str, evaluation_id: str) -> Path:
    return evaluations_dir(run_id) / validate_identifier(
        evaluation_id, "evaluation_id"
    )


def reports_dir(run_id: str) -> Path:
    return run_root(run_id) / "reports"


def metadata_dir(run_id: str) -> Path:
    return run_root(run_id) / "metadata"


def data_preparation_dir(dataset_id: str) -> Path:
    return RESULTS_ROOT / "data-preparation" / validate_identifier(
        dataset_id, "dataset_id"
    )

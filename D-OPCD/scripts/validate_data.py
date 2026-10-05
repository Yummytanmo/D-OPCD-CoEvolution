#!/usr/bin/env python3
"""Validate the strict prompt-only D-OPCD trainer JSONL schema."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from common import (  # noqa: E402
    DEFAULT_CONTEXT_LABEL,
    TEACHER_CONTEXT_MODES,
    TRAIN_FIELDS,
    compose_teacher_context,
    normalize_text,
    read_jsonl,
    sha256_file,
    teacher_context_description,
    write_json,
)


FORBIDDEN_FIELDS = {
    "image",
    "image_path",
    "selected_image_path",
    "reward",
    "score",
    "label",
    "judge",
    "questions",
    "teacher_context",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--expected", type=int)
    parser.add_argument("--selection", choices=["changed-only", "all"], default="changed-only")
    parser.add_argument("--context-label", default=DEFAULT_CONTEXT_LABEL)
    parser.add_argument("--teacher-context-mode", choices=TEACHER_CONTEXT_MODES, default="p_only")
    parser.add_argument("--report", type=Path)
    return parser.parse_args()


def validate_row(
    row: dict[str, Any], index: int, selection: str, context_label: str,
    teacher_context_mode: str = "p_only",
) -> list[str]:
    errors: list[str] = []
    extra = set(row) - TRAIN_FIELDS
    missing = TRAIN_FIELDS - set(row)
    if extra:
        errors.append(f"row {index}: unexpected fields {sorted(extra)}")
    if missing:
        errors.append(f"row {index}: missing fields {sorted(missing)}")
        return errors
    if not isinstance(row["sample_id"], int) or row["sample_id"] < 0:
        errors.append(f"row {index}: sample_id must be a non-negative integer")
    for field in ("source_sample_id", "tag", "original_query", "privileged_prompt"):
        if not str(row[field]).strip():
            errors.append(f"row {index}: {field} is empty")
    forbidden = FORBIDDEN_FIELDS & set(row)
    if forbidden:
        errors.append(f"row {index}: forbidden fields {sorted(forbidden)}")
    unchanged = normalize_text(row["original_query"]) == normalize_text(row["privileged_prompt"])
    if selection == "changed-only" and unchanged:
        errors.append(f"row {index}: unchanged privileged context")
    try:
        teacher = (
            str(row["privileged_prompt"]).strip()
            if teacher_context_mode == "p_only"
            else compose_teacher_context(row["original_query"], row["privileged_prompt"], context_label)
        )
        if not teacher:
            errors.append(f"row {index}: teacher context is empty")
        if teacher_context_mode == "q_plus_p" and not teacher.startswith(str(row["original_query"]).strip() + "\n\n"):
            errors.append(f"row {index}: teacher context does not retain q")
    except ValueError as exc:
        errors.append(f"row {index}: {exc}")
    return errors


def main() -> int:
    args = parse_args()
    data_path = args.data.expanduser().resolve()
    if not data_path.is_file():
        raise FileNotFoundError(data_path)
    rows = read_jsonl(data_path)
    errors: list[str] = []
    seen_ids: set[int] = set()
    seen_sources: set[str] = set()
    for index, row in enumerate(rows, start=1):
        errors.extend(validate_row(row, index, args.selection, args.context_label,
                                   args.teacher_context_mode))
        sample_id = row.get("sample_id")
        source_id = str(row.get("source_sample_id") or "")
        if isinstance(sample_id, int):
            if sample_id in seen_ids:
                errors.append(f"row {index}: duplicate sample_id {sample_id}")
            seen_ids.add(sample_id)
        if source_id:
            if source_id in seen_sources:
                errors.append(f"row {index}: duplicate source_sample_id {source_id!r}")
            seen_sources.add(source_id)
    if args.expected is not None and len(rows) != args.expected:
        errors.append(f"count={len(rows)}, expected={args.expected}")
    report = {
        "schema_version": 1,
        "method": "D-OPCD",
        "data": {"path": str(data_path), "sha256": sha256_file(data_path)},
        "count": len(rows),
        "expected": args.expected,
        "selection": args.selection,
        "student_context": "q",
        "teacher_context": teacher_context_description(args.teacher_context_mode),
        "teacher_context_mode": args.teacher_context_mode,
        "valid": not errors,
        "errors": errors[:100],
        "resource_policy": {"cpu_only": True, "training_run": False},
    }
    report_path = args.report or data_path.with_suffix(".validation.json")
    write_json(report_path, report)
    print(report_path)
    if errors:
        raise SystemExit("D-OPCD data validation failed: " + "; ".join(errors[:5]))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python3
"""Convert arbitrary JSONL prompt pairs into the D-OPCD training schema."""

from __future__ import annotations

import argparse
from collections import Counter
from pathlib import Path
import re
import sys
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from common import (  # noqa: E402
    DEFAULT_CONTEXT_LABEL,
    compose_teacher_context,
    normalize_text,
    read_jsonl,
    sha256_file,
    stable_integer_id,
    write_json,
    write_jsonl,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--source-manifest", type=Path)
    parser.add_argument("--manifest-id-field", default="source_sample_id")
    parser.add_argument("--manifest-query-field", default="prompt")
    parser.add_argument("--agent-run-id")
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--split", choices=["train", "validation"], default="train")
    parser.add_argument("--output-root", type=Path, default=PROJECT_ROOT / "data")
    parser.add_argument("--data-id", required=True)
    parser.add_argument("--id-field", default="sample_id")
    parser.add_argument("--tag-field", default="tag")
    parser.add_argument("--query-field", default="original_query")
    parser.add_argument("--privileged-prompt-field", default="privileged_prompt")
    parser.add_argument("--status-field")
    parser.add_argument("--required-status")
    parser.add_argument("--default-tag")
    parser.add_argument("--selection", choices=["changed-only", "all"], default="changed-only")
    parser.add_argument("--context-label", default=DEFAULT_CONTEXT_LABEL)
    parser.add_argument("--expected-input", type=int)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def required_text(row: dict[str, Any], field: str, line_number: int) -> str:
    raw = row.get(field)
    value = "" if raw is None else str(raw).strip()
    if not value:
        raise ValueError(f"row {line_number}: field {field!r} is empty")
    return value


def main() -> int:
    args = parse_args()
    if not re.fullmatch(r"[A-Za-z0-9_.-]+", args.dataset):
        raise ValueError("dataset may contain only letters, numbers, '.', '_' and '-'")
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]*", args.data_id):
        raise ValueError("data-id must be a safe non-empty path component")
    source = args.input.expanduser().resolve()
    if not source.is_file():
        raise FileNotFoundError(source)
    rows = read_jsonl(source)
    if args.expected_input is not None and len(rows) != args.expected_input:
        raise ValueError(f"Input has {len(rows)} rows, expected {args.expected_input}")
    if bool(args.status_field) != bool(args.required_status):
        raise ValueError("--status-field and --required-status must be provided together")

    manifest_path: Path | None = None
    manifest_rows: list[dict[str, Any]] = []
    manifest_by_id: dict[str, dict[str, Any]] = {}
    if args.source_manifest is not None:
        manifest_path = args.source_manifest.expanduser().resolve()
        if not manifest_path.is_file():
            raise FileNotFoundError(manifest_path)
        manifest_rows = read_jsonl(manifest_path)
        for line_number, row in enumerate(manifest_rows, start=1):
            source_id = required_text(row, args.manifest_id_field, line_number)
            if source_id in manifest_by_id:
                raise ValueError(
                    f"source manifest row {line_number}: duplicate sample ID {source_id!r}"
                )
            manifest_by_id[source_id] = row

    training_rows: list[dict[str, Any]] = []
    audit_rows: list[dict[str, Any]] = []
    seen_source_ids: set[str] = set()
    seen_ids: set[int] = set()
    dropped_unchanged = 0
    for line_number, row in enumerate(rows, start=1):
        if args.status_field and str(row.get(args.status_field) or "").strip() != args.required_status:
            raise ValueError(
                f"row {line_number}: field {args.status_field!r} must equal "
                f"{args.required_status!r}"
            )
        source_id = required_text(row, args.id_field, line_number)
        if source_id in seen_source_ids:
            raise ValueError(f"row {line_number}: duplicate source ID {source_id!r}")
        seen_source_ids.add(source_id)
        query = required_text(row, args.query_field, line_number)
        if manifest_by_id:
            manifest_row = manifest_by_id.get(source_id)
            if manifest_row is None:
                raise ValueError(
                    f"row {line_number}: source ID {source_id!r} is absent from source manifest"
                )
            manifest_query = required_text(
                manifest_row, args.manifest_query_field, line_number
            )
            if normalize_text(query) != normalize_text(manifest_query):
                raise ValueError(
                    f"row {line_number}: query does not match source manifest for {source_id!r}"
                )
        privileged = required_text(row, args.privileged_prompt_field, line_number)
        unchanged = normalize_text(query) == normalize_text(privileged)
        if args.selection == "changed-only" and unchanged:
            dropped_unchanged += 1
            continue
        tag = str(row.get(args.tag_field) or args.default_tag or args.dataset).strip()
        sample_id = stable_integer_id(args.dataset, args.split, source_id)
        if sample_id in seen_ids:
            raise RuntimeError(f"Stable integer ID collision for {source_id!r}")
        seen_ids.add(sample_id)
        trainer_row = {
            "sample_id": sample_id,
            "source_sample_id": source_id,
            "tag": tag,
            "original_query": query,
            "privileged_prompt": privileged,
        }
        audit_row = {
            **trainer_row,
            "dataset": args.dataset,
            "split": args.split,
            "source_jsonl": str(source),
            "source_manifest": str(manifest_path) if manifest_path else None,
            "agent_run_id": args.agent_run_id,
            "source_row_number": line_number,
            "teacher_context": compose_teacher_context(query, privileged, args.context_label),
            "student_context_role": "q",
            "teacher_context_role": "[q;p]",
            "no_context_advantage": unchanged,
            "privileged_information": "prompt_only",
        }
        training_rows.append(trainer_row)
        audit_rows.append(audit_row)

    if manifest_by_id and seen_source_ids != set(manifest_by_id):
        missing = sorted(set(manifest_by_id) - seen_source_ids)
        extra = sorted(seen_source_ids - set(manifest_by_id))
        raise ValueError(
            f"input/source manifest ID mismatch; missing={missing[:5]}, extra={extra[:5]}"
        )

    if not training_rows:
        raise ValueError("Selection produced an empty D-OPCD dataset")
    output_dir = args.output_root.expanduser().resolve() / args.dataset / args.data_id
    train_path = output_dir / f"{args.split}.jsonl"
    audit_path = output_dir / f"{args.split}.audit.jsonl"
    report_path = output_dir / f"{args.split}.report.json"
    existing = [path for path in (train_path, audit_path, report_path) if path.exists()]
    if existing and not args.overwrite:
        raise FileExistsError(
            "Refusing to overwrite: " + ", ".join(str(path) for path in existing)
        )
    write_jsonl(train_path, training_rows)
    write_jsonl(audit_path, audit_rows)
    report = {
        "schema_version": 1,
        "method": "D-OPCD",
        "data_class": "agent-derived-training",
        "data_id": args.data_id,
        "source_data_role": "original_query_and_metadata",
        "agent_output_role": "privileged_prompt_and_provenance",
        "dataset": args.dataset,
        "split": args.split,
        "source": {"path": str(source), "sha256": sha256_file(source)},
        "source_count": len(rows),
        "count": len(training_rows),
        "dropped_unchanged": dropped_unchanged,
        "filtering": {
            "input": len(rows),
            "output": len(training_rows),
            "dropped_unchanged": dropped_unchanged,
        },
        "selection": args.selection,
        "student_context": "q",
        "teacher_context": "[q;p]",
        "context_label": args.context_label,
        "by_tag": dict(sorted(Counter(row["tag"] for row in training_rows).items())),
        "training_data": {"path": str(train_path), "sha256": sha256_file(train_path)},
        "audit_data": {"path": str(audit_path), "sha256": sha256_file(audit_path)},
        "privileged_information": "prompt_only",
        "resource_policy": {"cpu_only": True, "training_run": False},
    }
    if manifest_path is not None:
        report["source_manifest"] = {
            "path": str(manifest_path),
            "sha256": sha256_file(manifest_path),
            "count": len(manifest_rows),
        }
    if args.agent_run_id:
        report["agent_run"] = {
            "run_id": args.agent_run_id,
            "index": {"path": str(source), "sha256": sha256_file(source)},
        }
    write_json(
        report_path,
        report,
    )
    print(report_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python3
"""Resolve and optionally launch an independent D-OPCD training task."""

from __future__ import annotations

import argparse
import ast
import json
import math
import os
from pathlib import Path
import shlex
import subprocess
import sys
from datetime import datetime, timezone
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]
TRAINING_DATA_ROOT = PROJECT_ROOT / "data"
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from common import (  # noqa: E402
    DEFAULT_CONTEXT_LABEL,
    TRAIN_FIELDS,
    normalize_text,
    read_json,
    read_jsonl,
    resolve_teacher_context_mode,
    resolve_project_path,
    sha256_file,
    teacher_context_description,
    write_json,
)
from result_layout import (  # noqa: E402
    RESULTS_ROOT,
    metadata_dir,
    task_run_id,
    training_dir,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--runtime-config", type=Path, default=PROJECT_ROOT / "configs" / "runtime.json")
    parser.add_argument("--report", type=Path)
    parser.add_argument("--execute", action="store_true")
    return parser.parse_args()


def append_arg(command: list[str], flag: str, value: Any) -> None:
    if isinstance(value, list):
        value = ",".join(str(item) for item in value)
    command.extend([flag, str(value)])


def resolve_project_executable(path: str | Path) -> Path:
    """Make project-relative executables absolute without dereferencing venv symlinks."""
    value = Path(path).expanduser()
    if not value.is_absolute():
        value = PROJECT_ROOT / value
    return value.absolute()


def trainer_cli_contract(command: list[str], trainer_path: Path) -> dict[str, Any]:
    """Compare generated trainer flags with statically declared argparse options."""
    errors: list[str] = []
    declared_options: set[str] = set()
    try:
        tree = ast.parse(trainer_path.read_text(encoding="utf-8"), filename=str(trainer_path))
    except (OSError, SyntaxError, UnicodeError) as exc:
        errors.append(f"cannot inspect trainer argparse declarations: {exc}")
    else:
        for node in ast.walk(tree):
            if not isinstance(node, ast.Call):
                continue
            function = node.func
            if not isinstance(function, ast.Attribute) or function.attr != "add_argument":
                continue
            for argument in node.args:
                if (
                    isinstance(argument, ast.Constant)
                    and isinstance(argument.value, str)
                    and argument.value.startswith("--")
                ):
                    declared_options.add(argument.value)

    try:
        trainer_index = command.index(str(trainer_path))
    except ValueError:
        errors.append(f"trainer path is absent from generated command: {trainer_path}")
        requested_options: set[str] = set()
    else:
        requested_options = {
            value.split("=", 1)[0]
            for value in command[trainer_index + 1 :]
            if value.startswith("--")
        }
    unsupported_options = sorted(requested_options - declared_options)
    return {
        "name": "trainer_cli_contract",
        "trainer": str(trainer_path),
        "requested_options": sorted(requested_options),
        "unsupported_options": unsupported_options,
        "errors": errors,
        "ok": not errors and not unsupported_options,
    }


def run_with_persistent_log(
    command: list[str], *, cwd: Path, environment: dict[str, str], output_dir: Path
) -> None:
    """Mirror trainer stdout/stderr to the console and a durable launch log."""
    output_dir.mkdir(parents=True, exist_ok=True)
    log_path = output_dir / "launch.log"
    with log_path.open("a", encoding="utf-8") as handle:
        handle.write(
            f"\n=== launch {datetime.now(timezone.utc).isoformat()} ===\n"
            f"command={shlex.join(command)}\n"
        )
        handle.flush()
        process = subprocess.Popen(
            command,
            cwd=cwd,
            env=environment,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            encoding="utf-8",
            errors="replace",
            bufsize=1,
        )
        assert process.stdout is not None
        try:
            for line in process.stdout:
                sys.stdout.write(line)
                sys.stdout.flush()
                handle.write(line)
                handle.flush()
        finally:
            process.stdout.close()
        return_code = process.wait()
        handle.write(f"=== exit {return_code} ===\n")
        handle.flush()
    if return_code != 0:
        raise subprocess.CalledProcessError(return_code, command)


def build_command(
    runtime: dict[str, Any], task: dict[str, Any], output_dir: Path | None = None
) -> list[str]:
    trainer_args = dict(runtime["trainer_args"])
    trainer_args.update(task.get("trainer_args") or {})
    model_path = resolve_project_path(task.get("model_path") or runtime["model_path"])
    data_path = resolve_project_path(task["data_path"])
    sample_weights_path = task.get("sample_weights_path")
    output_dir = output_dir or training_dir(task_run_id(task))
    trainer_path = resolve_project_path(runtime["trainer"])
    accelerate_config = resolve_project_path(runtime["accelerate_config"])
    command = [
        str(resolve_project_executable(runtime["accelerate"])),
        "launch",
        "--config_file",
        str(accelerate_config),
        "--main_process_port",
        str(runtime.get("main_process_port", 29500)),
        str(trainer_path),
        "--project_root",
        str(PROJECT_ROOT),
        "--pretrained_model_name_or_path",
        str(model_path),
        "--data_jsonl",
        str(data_path),
        "--output_dir",
        str(output_dir),
    ]
    if sample_weights_path:
        command.extend(
            ["--sample_weights_jsonl", str(resolve_project_path(sample_weights_path))]
        )
    flags = {
        "logging_dir": "--logging_dir",
        "context_label": "--context_label",
        "teacher_context_mode": "--teacher_context_mode",
        "image_manifest_jsonl": "--image_manifest_jsonl",
        "vlm_model_path": "--vlm_model_path",
        "vlm_min_pixels": "--vlm_min_pixels",
        "vlm_max_pixels": "--vlm_max_pixels",
        "ema_decay": "--ema_decay",
        "loss_type": "--loss_type",
        "timesteps": "--timesteps",
        "resolution": "--resolution",
        "vae_scale_factor": "--vae_scale_factor",
        "student_max_sequence_length": "--student_max_sequence_length",
        "teacher_max_sequence_length": "--teacher_max_sequence_length",
        "teacher_rope_axis_length": "--teacher_rope_axis_length",
        "rank": "--rank",
        "lora_alpha": "--lora_alpha",
        "lora_dropout": "--lora_dropout",
        "lora_layers": "--lora_layers",
        "train_batch_size": "--train_batch_size",
        "gradient_accumulation_steps": "--gradient_accumulation_steps",
        "dataloader_num_workers": "--dataloader_num_workers",
        "max_train_steps": "--max_train_steps",
        "checkpointing_steps": "--checkpointing_steps",
        "resume_from_checkpoint": "--resume_from_checkpoint",
        "learning_rate": "--learning_rate",
        "lr_scheduler": "--lr_scheduler",
        "lr_warmup_steps": "--lr_warmup_steps",
        "adam_beta1": "--adam_beta1",
        "adam_beta2": "--adam_beta2",
        "adam_weight_decay": "--adam_weight_decay",
        "adam_epsilon": "--adam_epsilon",
        "max_grad_norm": "--max_grad_norm",
        "mixed_precision": "--mixed_precision",
        "report_to": "--report_to",
        "tracker_project_name": "--tracker_project_name",
        "tracker_run_name": "--tracker_run_name",
        "tracker_run_id": "--tracker_run_id",
        "wandb_init_timeout": "--wandb_init_timeout",
        "parameter_logging_steps": "--parameter_logging_steps",
        "seed": "--seed",
    }
    for key, flag in flags.items():
        if trainer_args.get(key) is not None:
            append_arg(command, flag, trainer_args[key])
    for key, flag in (
        ("gradient_checkpointing", "--gradient_checkpointing"),
        ("allow_tf32", "--allow_tf32"),
    ):
        if trainer_args.get(key) is True:
            command.append(flag)
    return command


def main() -> int:
    args = parse_args()
    runtime_path = args.runtime_config.expanduser().resolve()
    task_path = args.config.expanduser().resolve()
    runtime = read_json(runtime_path)
    task = read_json(task_path)
    if runtime.get("schema_version") != 1 or task.get("schema_version") not in {1, 2}:
        raise ValueError("Unsupported config schema")
    task_name = str(task.get("task_name") or "task")
    if task.get("schema_version") == 2 and not task.get("run_id"):
        raise ValueError("Task config schema 2 requires run_id")
    run_id = task_run_id(task)
    output_dir = training_dir(run_id)
    configured_output = task.get("output_dir")
    if configured_output is not None:
        if task.get("schema_version") == 2:
            raise ValueError(
                "Task config schema 2 derives output from run_id; remove output_dir"
            )
        legacy_output = resolve_project_path(configured_output)
        if legacy_output != output_dir:
            raise ValueError(
                "output_dir no longer selects an arbitrary destination; remove it and "
                f"use run_id={run_id!r}. Canonical training path: {output_dir}"
            )
    checks: list[dict[str, Any]] = []
    blockers: list[str] = []

    for name, value, access_mode in (
        ("python", resolve_project_executable(runtime["python"]), os.X_OK),
        (
            "accelerate",
            resolve_project_executable(runtime["accelerate"]),
            os.R_OK | os.X_OK,
        ),
        ("trainer", resolve_project_path(runtime["trainer"]), os.R_OK),
        (
            "accelerate_config",
            resolve_project_path(runtime["accelerate_config"]),
            os.R_OK,
        ),
    ):
        path = Path(value)
        ok = path.is_file() and os.access(path, access_mode)
        checks.append(
            {
                "name": name,
                "path": str(path),
                "required_access": {
                    "readable": bool(access_mode & os.R_OK),
                    "executable": bool(access_mode & os.X_OK),
                },
                "ok": ok,
            }
        )
        if not ok:
            requirements = []
            if access_mode & os.R_OK:
                requirements.append("readable")
            if access_mode & os.X_OK:
                requirements.append("executable")
            blockers.append(
                f"{name} is unavailable or not {'/'.join(requirements)}: {path}"
            )

    model_path = resolve_project_path(task.get("model_path") or runtime["model_path"])
    backend = str(runtime.get("backend") or "zimage")
    if backend not in {"zimage", "qwen_image"}:
        raise ValueError(f"Unsupported training backend: {backend!r}")
    required_model_files = [
        model_path / "transformer" / "config.json",
        model_path / "text_encoder" / "config.json",
        model_path / "tokenizer" / "tokenizer_config.json",
    ]
    if backend == "qwen_image":
        required_model_files.extend(
            [model_path / "model_index.json", model_path / "vae" / "config.json"]
        )
    model_ok = model_path.is_dir() and all(path.is_file() for path in required_model_files)
    model_config_errors: list[str] = []
    if model_ok:
        transformer_config = read_json(model_path / "transformer" / "config.json")
        expected_class = {
            "zimage": "ZImageTransformer2DModel",
            "qwen_image": "QwenImageTransformer2DModel",
        }[backend]
        if transformer_config.get("_class_name") != expected_class:
            model_config_errors.append(
                f"transformer class={transformer_config.get('_class_name')!r}, "
                f"expected={expected_class!r}"
            )
        if backend == "qwen_image":
            if bool(transformer_config.get("guidance_embeds")):
                model_config_errors.append("Qwen-Image guidance_embeds must be false")
            vae_config = read_json(model_path / "vae" / "config.json")
            configured_scale = int(
                (task.get("trainer_args") or {}).get(
                    "vae_scale_factor",
                    runtime.get("trainer_args", {}).get("vae_scale_factor", 8),
                )
            )
            expected_scale = 2 ** len(vae_config.get("temperal_downsample") or [])
            if configured_scale != expected_scale:
                model_config_errors.append(
                    f"vae_scale_factor={configured_scale}, expected={expected_scale}"
                )
        model_ok = not model_config_errors
    checks.append(
        {
            "name": f"{backend}_model",
            "path": str(model_path),
            "ok": model_ok,
            "required_files": [str(path) for path in required_model_files],
            "config_errors": model_config_errors,
        }
    )
    if not model_ok:
        blockers.append(f"{backend} model is incomplete: {model_path}")

    data_path = resolve_project_path(task["data_path"])
    data_errors: list[str] = []
    rows: list[dict[str, Any]] = []
    data_layout_ok = TRAINING_DATA_ROOT in data_path.parents
    if not data_layout_ok:
        data_errors.append(
            f"path must be under {TRAINING_DATA_ROOT}; source, agent-output, and archive "
            "paths are not trainer inputs"
        )
    if not data_path.is_file():
        data_errors.append("file is missing")
    else:
        rows = read_jsonl(data_path)
        seen_ids: set[int] = set()
        for index, row in enumerate(rows, start=1):
            if set(row) != TRAIN_FIELDS:
                data_errors.append(
                    f"row {index}: fields={sorted(row)}, expected={sorted(TRAIN_FIELDS)}"
                )
            sample_id = row.get("sample_id")
            if not isinstance(sample_id, int) or sample_id < 0 or sample_id in seen_ids:
                data_errors.append(f"row {index}: invalid/duplicate sample_id")
            elif isinstance(sample_id, int):
                seen_ids.add(sample_id)
            for field in ("source_sample_id", "tag", "original_query", "privileged_prompt"):
                if not str(row.get(field) or "").strip():
                    data_errors.append(f"row {index}: {field} is empty")
            if task.get("selection", "changed-only") == "changed-only" and (
                normalize_text(row.get("original_query") or "")
                == normalize_text(row.get("privileged_prompt") or "")
            ):
                data_errors.append(f"row {index}: unchanged privileged context")
    expected = task.get("expected_count")
    if expected is not None and len(rows) != int(expected):
        data_errors.append(f"count={len(rows)}, expected={expected}")
    checks.append(
        {
            "name": "prompt_context_data",
            "path": str(data_path),
            "count": len(rows),
            "expected": expected,
            "data_class": "agent-derived-training",
            "data_id": data_path.parent.name,
            "layout_ok": data_layout_ok,
            "ok": not data_errors,
            "errors": data_errors[:10],
        }
    )
    if data_errors:
        blockers.append("Training data is invalid: " + "; ".join(data_errors[:3]))

    sample_weights_path = task.get("sample_weights_path")
    if sample_weights_path:
        weights_path = resolve_project_path(sample_weights_path)
        weight_errors: list[str] = []
        weight_rows: list[dict[str, Any]] = []
        if not weights_path.is_file():
            weight_errors.append("file is missing")
        else:
            weight_rows = read_jsonl(weights_path)
            seen_weight_ids: set[int] = set()
            for index, row in enumerate(weight_rows, start=1):
                sample_id = row.get("sample_id")
                sample_weight = row.get("sample_weight")
                if (
                    not isinstance(sample_id, int)
                    or sample_id < 0
                    or sample_id in seen_weight_ids
                ):
                    weight_errors.append(f"row {index}: invalid/duplicate sample_id")
                else:
                    seen_weight_ids.add(sample_id)
                if (
                    not isinstance(sample_weight, (int, float))
                    or not math.isfinite(float(sample_weight))
                    or sample_weight <= 0
                ):
                    weight_errors.append(f"row {index}: sample_weight must be finite and > 0")
            data_ids = {int(row["sample_id"]) for row in rows if isinstance(row.get("sample_id"), int)}
            if seen_weight_ids != data_ids:
                weight_errors.append(
                    "sample IDs do not exactly match training data: "
                    f"missing={sorted(data_ids - seen_weight_ids)[:5]}, "
                    f"extra={sorted(seen_weight_ids - data_ids)[:5]}"
                )
        checks.append(
            {
                "name": "sample_weights",
                "path": str(weights_path),
                "count": len(weight_rows),
                "ok": not weight_errors,
                "errors": weight_errors[:10],
            }
        )
        if weight_errors:
            blockers.append("Sample weights are invalid: " + "; ".join(weight_errors[:3]))
    else:
        checks.append({"name": "sample_weights", "mode": "uniform", "ok": True})

    output_ok = RESULTS_ROOT in output_dir.parents
    checks.append(
        {
            "name": "canonical_result_layout",
            "run_id": run_id,
            "path": str(output_dir),
            "ok": output_ok,
        }
    )
    if not output_ok:
        blockers.append(f"Training output is outside results/: {output_dir}")

    trainer_args = dict(runtime["trainer_args"])
    trainer_args.update(task.get("trainer_args") or {})
    routing_ok = bool(str(trainer_args.get("context_label", DEFAULT_CONTEXT_LABEL) or "").strip())
    routing_error = None
    try:
        teacher_mode = resolve_teacher_context_mode(
            trainer_args.get("teacher_context_mode"), output_dir / "args.json",
            bool(trainer_args.get("resume_from_checkpoint")),
        )
    except ValueError as exc:
        teacher_mode = None
        routing_ok = False
        routing_error = str(exc)
    checks.append(
        {
            "name": "context_routing",
            "ok": routing_ok,
            "student": "q",
            "teacher": teacher_context_description(teacher_mode) if teacher_mode else None,
            "teacher_context_mode": teacher_mode,
            "error": routing_error,
            "implemented_in": str(resolve_project_path(runtime["trainer"])),
        }
    )
    if not routing_ok:
        blockers.append(routing_error or "context_label is empty")

    if teacher_mode in {"vlm_q_image", "vlm_q_p_image"}:
        image_errors: list[str] = []
        image_path_raw = trainer_args.get("image_manifest_jsonl")
        image_path = resolve_project_path(image_path_raw) if image_path_raw else None
        image_rows: list[dict[str, Any]] = []
        if image_path is None or not image_path.is_file():
            image_errors.append("image_manifest_jsonl is missing")
        elif TRAINING_DATA_ROOT not in image_path.parents:
            image_errors.append("image_manifest_jsonl must be under data/")
        else:
            image_rows = read_jsonl(image_path)
            data_ids = {row.get("sample_id") for row in rows}
            image_ids = {row.get("sample_id") for row in image_rows}
            if len(image_ids) != len(image_rows) or data_ids != image_ids:
                image_errors.append("image manifest sample IDs do not exactly match training data")
            for index, image_row in enumerate(image_rows, start=1):
                if set(image_row) != {"sample_id", "target_image", "target_prompt_role", "target_sha256"}:
                    image_errors.append(f"image row {index}: unexpected fields")
                    continue
                if image_row["target_prompt_role"] != "privileged_prompt":
                    image_errors.append(f"image row {index}: wrong target_prompt_role")
                relative = Path(image_row["target_image"])
                target = (image_path.parent / relative).resolve() if not relative.is_absolute() else relative.resolve()
                if not target.is_file() or sha256_file(target) != image_row["target_sha256"]:
                    image_errors.append(f"image row {index}: missing image or SHA256 mismatch")
        vlm_path = resolve_project_path(trainer_args.get("vlm_model_path") or
                                        os.environ.get("DOPCD_VLM_MODEL_PATH", "models/Qwen3-VL-4B-Instruct"))
        if not (vlm_path / "config.json").is_file() or not (vlm_path / "preprocessor_config.json").is_file():
            image_errors.append(f"Qwen3-VL model/processor unavailable: {vlm_path}")
        checks.append({
            "name": "vlm_image_context", "ok": not image_errors,
            "mode": teacher_mode, "image_manifest": str(image_path) if image_path else None,
            "image_count": len(image_rows), "vlm_model_path": str(vlm_path),
            "errors": image_errors[:10],
        })
        if image_errors:
            blockers.append("VLM image context is invalid: " + "; ".join(image_errors[:3]))
    elif trainer_args.get("image_manifest_jsonl"):
        blockers.append("image_manifest_jsonl requires a VLM image teacher mode")

    command = build_command(runtime, task, output_dir)
    cli_contract = trainer_cli_contract(
        command, resolve_project_path(runtime["trainer"])
    )
    checks.append(cli_contract)
    if not cli_contract["ok"]:
        details = list(cli_contract["errors"])
        if cli_contract["unsupported_options"]:
            details.append(
                "unsupported options: "
                + ", ".join(cli_contract["unsupported_options"])
            )
        blockers.append("Trainer CLI contract is invalid: " + "; ".join(details))
    run_metadata_dir = metadata_dir(run_id)
    report_path = args.report or run_metadata_dir / "preflight.json"
    report_path = report_path.expanduser()
    if not report_path.is_absolute():
        report_path = PROJECT_ROOT / report_path
    report_path = report_path.resolve()
    if output_dir.parent not in report_path.parents:
        raise ValueError(
            f"Preflight report must stay inside run root {output_dir.parent}: {report_path}"
        )
    resolved_path = run_metadata_dir / "resolved-config.json"
    if args.execute:
        write_json(
            resolved_path,
            {
                "schema_version": 2,
                "run_id": run_id,
                "runtime_config": str(runtime_path),
                "task_config": str(task_path),
                "runtime": runtime,
                "task": task,
                "result_layout": {
                    "run_root": str(output_dir.parent),
                    "training": str(output_dir),
                    "evaluations": str(output_dir.parent / "evaluations"),
                    "reports": str(output_dir.parent / "reports"),
                    "metadata": str(run_metadata_dir),
                },
                "command": command,
            },
        )
    report = {
        "schema_version": 2,
        "method": "D-OPSD" if teacher_mode == "vlm_q_image" else "D-OPCD",
        "backend": backend,
        "task_name": task_name,
        "run_id": run_id,
        "training_directory": str(output_dir),
        "mode": "execute" if args.execute else "dry-run",
        "ready": not blockers,
        "checks": checks,
        "blockers": blockers,
        "student_context": "q",
        "teacher_context": teacher_context_description(teacher_mode) if teacher_mode else None,
        "teacher_context_mode": teacher_mode,
        "command": command,
        "shell_command": shlex.join(command),
        "resolved_config": str(resolved_path) if args.execute else None,
        "training_executed": False,
        "resource_policy": {
            "default_dry_run": True,
            "model_loaded_during_preflight": False,
            "execute_flag_required": True,
        },
    }
    if args.execute or args.report:
        write_json(report_path, report)
    print(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True))
    if not args.execute:
        return 0
    if blockers:
        raise RuntimeError("Preflight failed: " + "; ".join(blockers))
    environment = os.environ.copy()
    environment.update({str(key): str(value) for key, value in runtime.get("environment", {}).items()})
    source_root = str(PROJECT_ROOT / "src")
    previous_pythonpath = environment.get("PYTHONPATH")
    environment["PYTHONPATH"] = (
        source_root if not previous_pythonpath else f"{source_root}{os.pathsep}{previous_pythonpath}"
    )
    run_with_persistent_log(
        command,
        cwd=PROJECT_ROOT,
        environment=environment,
        output_dir=output_dir,
    )
    report["training_executed"] = True
    write_json(report_path, report)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

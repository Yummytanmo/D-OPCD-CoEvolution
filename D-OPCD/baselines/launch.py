#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import os
import re
import shlex
import subprocess
import sys
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from common import ensure_within, read_json, read_jsonl, resolve_project_path  # noqa: E402

try:
    from .data import load_joined_rows
except ImportError:
    from data import load_joined_rows


FAIR_ARG_MAP = {
    "resolution": "resolution",
    "student_max_sequence_length": "max_sequence_length",
    "rank": "rank",
    "lora_alpha": "lora_alpha",
    "lora_dropout": "lora_dropout",
    "lora_layers": "lora_layers",
    "train_batch_size": "train_batch_size",
    "gradient_accumulation_steps": "gradient_accumulation_steps",
    "dataloader_num_workers": "dataloader_num_workers",
    "max_train_steps": "max_train_steps",
    "checkpointing_steps": "checkpointing_steps",
    "resume_from_checkpoint": "resume_from_checkpoint",
    "learning_rate": "learning_rate",
    "lr_scheduler": "lr_scheduler",
    "lr_warmup_steps": "lr_warmup_steps",
    "adam_beta1": "adam_beta1",
    "adam_beta2": "adam_beta2",
    "adam_weight_decay": "adam_weight_decay",
    "adam_epsilon": "adam_epsilon",
    "max_grad_norm": "max_grad_norm",
    "mixed_precision": "mixed_precision",
    "report_to": "report_to",
    "tracker_project_name": "tracker_project_name",
    "parameter_logging_steps": "parameter_logging_steps",
    "seed": "seed",
    "gradient_checkpointing": "gradient_checkpointing",
    "allow_tf32": "allow_tf32",
}
BOOLEAN_ARGS = {"gradient_checkpointing", "allow_tf32", "random_horizontal_flip"}
SAFE_RUN_ID = re.compile(r"^[A-Za-z0-9._-]+$")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Launch an SFT or flow-DPO baseline with the shared D-OPCD runtime.")
    parser.add_argument("--config", type=Path, required=True, help="Method task JSON.")
    parser.add_argument(
        "--protocol",
        type=Path,
        default=PROJECT_ROOT / "baselines" / "configs" / "zimage_fair.json",
    )
    parser.add_argument("--dry-run", action="store_true", help="Validate and print without launching.")
    return parser.parse_args()


def _project_path(value: str | Path) -> Path:
    return resolve_project_path(value, PROJECT_ROOT)


def load_configs(protocol_path: Path, task_path: Path) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    protocol = read_json(protocol_path.expanduser().resolve())
    task = read_json(task_path.expanduser().resolve())
    if protocol.get("schema_version") != 1 or task.get("schema_version") != 1:
        raise ValueError("Unsupported baseline config schema")
    reference_path = _project_path(protocol["reference_runtime_config"])
    runtime = read_json(reference_path)
    if runtime.get("schema_version") != 1 or runtime.get("backend") != "zimage":
        raise ValueError("Fair baseline reference must be a schema-1 Z-Image runtime")
    return protocol, task, runtime


def build_effective_args(
    protocol: dict[str, Any], task: dict[str, Any], runtime: dict[str, Any]
) -> dict[str, Any]:
    method = task.get("method")
    methods = protocol.get("methods") or {}
    if method not in methods:
        raise ValueError(f"Unknown baseline method {method!r}; expected one of {sorted(methods)}")
    reference_args = dict(protocol.get("reference_trainer_defaults") or {})
    reference_args.update(runtime.get("trainer_args") or {})
    missing = sorted(key for key in FAIR_ARG_MAP if key not in reference_args)
    if missing:
        raise ValueError(f"Reference D-OPCD runtime is missing fairness keys: {missing}")
    effective = {
        destination: reference_args[source]
        for source, destination in FAIR_ARG_MAP.items()
    }
    effective.update(methods[method].get("trainer_args") or {})
    effective["tracker_run_name"] = task["run_id"]
    effective["tracker_run_id"] = task["run_id"]
    return effective


def _append_cli(command: list[str], key: str, value: Any) -> None:
    if value is None:
        return
    flag = f"--{key}"
    if key in BOOLEAN_ARGS:
        if value:
            command.append(flag)
        return
    if isinstance(value, list):
        value = ",".join(str(item) for item in value)
    command.extend([flag, str(value)])


def validate_task(task: dict[str, Any]) -> dict[str, Any]:
    run_id = task.get("run_id")
    if not isinstance(run_id, str) or not SAFE_RUN_ID.fullmatch(run_id):
        raise ValueError("run_id must contain only letters, digits, '.', '_' or '-'")
    method = task.get("method")
    if method not in {"sft", "flow_dpo"}:
        raise ValueError("method must be 'sft' or 'flow_dpo'")
    training_root = (PROJECT_ROOT / "data").resolve()
    context_path = ensure_within(_project_path(task["context_jsonl"]), training_root)
    manifest_path = ensure_within(_project_path(task["image_manifest_jsonl"]), training_root)
    if context_path.parent != manifest_path.parent:
        raise ValueError("Context and image manifest must be in the same immutable data ID")
    weights_path = None
    if task.get("sample_weights_jsonl"):
        weights_path = ensure_within(_project_path(task["sample_weights_jsonl"]), training_root)
        if weights_path.parent != context_path.parent:
            raise ValueError("Sample weights must be in the same immutable data ID")
    joined = load_joined_rows(
        context_path,
        manifest_path,
        method,
        sample_weights_jsonl=weights_path,
        image_root=context_path.parent,
    )
    expected_count = task.get("expected_count")
    if expected_count is not None and expected_count != len(joined):
        raise ValueError(f"expected_count={expected_count}, but the exact joined count is {len(joined)}")
    return {
        "run_id": run_id,
        "method": method,
        "context_path": context_path,
        "manifest_path": manifest_path,
        "weights_path": weights_path,
        "count": len(joined),
    }


def build_command(
    protocol: dict[str, Any],
    task: dict[str, Any],
    runtime: dict[str, Any],
    validated: dict[str, Any],
) -> tuple[list[str], Path, dict[str, Any]]:
    effective_args = build_effective_args(protocol, task, runtime)
    method_config = protocol["methods"][validated["method"]]
    trainer = _project_path(method_config["trainer"])
    accelerate = _project_path(runtime["accelerate"])
    accelerate_config = _project_path(runtime["accelerate_config"])
    model_path = _project_path(task.get("model_path") or runtime["model_path"])
    results_root = Path(os.environ.get("DOPCD_RESULTS_ROOT", PROJECT_ROOT / "results")).expanduser().resolve()
    output_dir = ensure_within(results_root / "runs" / validated["run_id"] / "training", results_root)
    command = [
        str(accelerate),
        "launch",
        "--config_file",
        str(accelerate_config),
        "--main_process_port",
        str(runtime.get("main_process_port", 29500)),
        str(trainer),
        "--project_root",
        str(PROJECT_ROOT),
        "--pretrained_model_name_or_path",
        str(model_path),
        "--context_jsonl",
        str(validated["context_path"]),
        "--image_manifest_jsonl",
        str(validated["manifest_path"]),
        "--output_dir",
        str(output_dir),
    ]
    if validated["weights_path"] is not None:
        command.extend(["--sample_weights_jsonl", str(validated["weights_path"])])
    for key, value in effective_args.items():
        _append_cli(command, key, value)
    return command, output_dir, effective_args


def main() -> None:
    args = parse_args()
    protocol_path = args.protocol if args.protocol.is_absolute() else PROJECT_ROOT / args.protocol
    task_path = args.config if args.config.is_absolute() else PROJECT_ROOT / args.config
    protocol, task, runtime = load_configs(protocol_path, task_path)
    validated = validate_task(task)
    command, output_dir, effective_args = build_command(protocol, task, runtime, validated)
    summary = {
        "protocol": protocol["protocol_name"],
        "reference_runtime_config": str(_project_path(protocol["reference_runtime_config"])),
        "method": validated["method"],
        "run_id": validated["run_id"],
        "joined_count": validated["count"],
        "output_dir": str(output_dir),
        "effective_trainer_args": effective_args,
        "command": command,
        "shell_command": shlex.join(command),
    }
    print(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True))
    if args.dry_run:
        return
    environment = os.environ.copy()
    environment.update({str(key): str(value) for key, value in runtime.get("environment", {}).items()})
    environment.update({str(key): str(value) for key, value in protocol.get("environment", {}).items()})
    subprocess.run(command, cwd=PROJECT_ROOT, env=environment, check=True)


if __name__ == "__main__":
    main()

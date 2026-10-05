"""D-OPCD training entry: reusable from runners or runnable as a Python file.

Numerical training remains in the Z-Image/Qwen trainer. This entry resolves
reusable task templates before calling the shared launcher.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

if not __package__:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from train.common import ROOT, run_python
from runners.common import SAFE_ID, path, read_json, read_jsonl, reserve_run, write_json


def run(config: Path, runtime_config: Path, *, execute: bool = False,
        method: str = "dopcd") -> str | None:
    config = path(config)
    runtime_config = path(runtime_config)
    task = read_json(config)
    run_id = task.get("run_id")
    if not run_id:
        identifier = task.get("experiment_id")
        if not isinstance(identifier, str) or not SAFE_ID.fullmatch(identifier):
            raise ValueError("Reusable training template needs experiment_id")
        data = path(task["data_path"])
        if task.get("expected_count") != len(read_jsonl(data)):
            raise ValueError("Training data count mismatch")
        if not runtime_config.is_file():
            raise FileNotFoundError(runtime_config)
        if not execute:
            print(json.dumps({"status": "configured", "experiment_id": identifier,
                              "method": method, "training_count": task["expected_count"],
                              "runtime_config": str(runtime_config)}, indent=2))
            return None
        run_id, run_root = reserve_run(identifier)
        task.update(run_id=run_id, task_name=run_id)
        task.setdefault("trainer_args", {}).update(tracker_run_name=run_id,
                                                    tracker_run_id=run_id)
        config = run_root / "metadata/training-task.json"
        write_json(config, task)
    args = ["--config", str(config), "--runtime-config", str(runtime_config)]
    if execute:
        args.append("--execute")
    run_python(ROOT / "scripts" / "launch_training.py", *args)
    return run_id


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--runtime-config", type=Path, required=True)
    parser.add_argument("--execute", action="store_true", help="Dry-run by default")
    args = parser.parse_args()
    run(args.config, args.runtime_config, execute=args.execute)


if __name__ == "__main__":
    main()

"""Small process boundary around the existing, validated training launchers."""

from __future__ import annotations

import json
import os
from pathlib import Path
import subprocess
import sys


ROOT = Path(__file__).resolve().parents[1]


def run_python(script: Path, *args: str, env: dict[str, str] | None = None) -> None:
    subprocess.run(
        [sys.executable, str(script), *map(str, args)],
        cwd=ROOT,
        env=env or os.environ.copy(),
        check=True,
    )


def run_baseline(method: str, config: Path, *, protocol: Path | None = None,
                 dry_run: bool = False) -> None:
    task = json.loads(config.read_text(encoding="utf-8"))
    if task.get("method") != method:
        raise ValueError(f"Expected {method!r} task, got {task.get('method')!r}")
    args = ["--config", str(config)]
    if protocol is not None:
        args += ["--protocol", str(protocol)]
    if dry_run:
        args.append("--dry-run")
    run_python(ROOT / "baselines" / "launch.py", *args)

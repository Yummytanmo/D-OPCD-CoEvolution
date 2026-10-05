"""D-OPSD training with a query-and-image-conditioned EMA teacher."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

if not __package__:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from train.common import ROOT
from train.dopcd import run as run_training


def run(config: Path, runtime_config: Path, *, execute: bool = False) -> str | None:
    return run_training(config, runtime_config, execute=execute, method="dopsd")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--runtime-config", type=Path, default=ROOT / "configs/runtime.json")
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--execute", action="store_true", help="Start training; dry-run by default")
    mode.add_argument("--dry-run", action="store_true", help="Check the task definition without training")
    args = parser.parse_args()
    run(args.config, args.runtime_config, execute=args.execute)


if __name__ == "__main__":
    main()

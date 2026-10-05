"""Standalone flow-DPO baseline training entry."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

if not __package__:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from train.common import run_baseline


def run(config: Path, *, protocol: Path | None = None, dry_run: bool = False) -> None:
    run_baseline("flow_dpo", config, protocol=protocol, dry_run=dry_run)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--protocol", type=Path)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    run(args.config, protocol=args.protocol, dry_run=args.dry_run)


if __name__ == "__main__":
    main()

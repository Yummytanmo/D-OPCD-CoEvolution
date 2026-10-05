#!/usr/bin/env python3
"""Deploy GenEval assets, construct its feedback agent, then serve it."""

from __future__ import annotations

import argparse
from pathlib import Path

from evaluation.assets import configure_cache_environment, default_model_root
from evaluation.evaluators.geneval_runtime import GenEvalRuntime
from evaluation.feedback_agents.geneval import GenEvalFeedbackAgent
from evaluation.model_config import config_section
from evaluation.services.bootstrap import (
    add_server_arguments,
    configure_service_file_logging,
    evaluator_concurrency,
    optional_verbalizer,
)
from evaluation.services.runtime import serve_feedback_agent


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    add_server_arguments(parser, benchmark="geneval", default_port=8101)
    root = default_model_root()
    geneval_config = config_section("geneval")
    parser.add_argument(
        "--model-path",
        type=Path,
        default=Path(str(geneval_config.get("model_path", root / "geneval"))),
    )
    parser.add_argument(
        "--open-clip-cache-dir",
        type=Path,
        default=Path(str(geneval_config.get("open_clip_cache_dir", root / "open_clip"))),
    )
    parser.add_argument(
        "--model-config",
        type=Path,
        default=Path(str(geneval_config["model_config"]))
        if geneval_config.get("model_config")
        else None,
    )
    parser.add_argument(
        "--reward-only",
        action="store_true",
        help="Disable the feedback verbalizer and accept only reward requests.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    configure_service_file_logging(
        "geneval",
        log_file=args.service_log_file,
        record_file=args.feedback_record_file,
    )
    configure_cache_environment(args.model_path.parent)
    concurrency = evaluator_concurrency()
    evaluator_runtimes = []
    for _ in range(concurrency):
        runtime = GenEvalRuntime(
            model_path=args.model_path,
            open_clip_cache_dir=args.open_clip_cache_dir,
            model_config=args.model_config,
            device="cuda:0",
        )
        runtime.load()
        evaluator_runtimes.append(runtime)
    agent = GenEvalFeedbackAgent(
        model_path=args.model_path,
        open_clip_cache_dir=args.open_clip_cache_dir,
        model_config=args.model_config,
        evaluator_runtimes=evaluator_runtimes,
        verbalizer=None if args.reward_only else optional_verbalizer(),
    )
    serve_feedback_agent(
        agent,
        host=args.host,
        port=args.port,
        warmup=False,
        feedback_modes={"reward"} if args.reward_only else None,
    )


if __name__ == "__main__":
    main()

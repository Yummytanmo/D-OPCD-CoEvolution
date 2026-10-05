#!/usr/bin/env python3
"""Load the GenEval2 Qwen3-VL evaluator and serve Soft-TIFA rewards."""

from __future__ import annotations

import argparse
import os

from evaluation.evaluators.geneval2_runtime import GenEval2Runtime
from evaluation.feedback_agents import GenEval2FeedbackAgent
from evaluation.model_config import config_section
from evaluation.services.bootstrap import (
    add_server_arguments,
    configure_service_file_logging,
    optional_verbalizer,
)
from evaluation.services.runtime import serve_feedback_agent


def default_model_path() -> str:
    configured = config_section("geneval2").get("model_path")
    if os.environ.get("GENEVAL2_MODEL_PATH"):
        return str(os.environ["GENEVAL2_MODEL_PATH"])
    if configured:
        return str(configured)
    return "Qwen/Qwen3-VL-8B-Instruct"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    add_server_arguments(parser, benchmark="geneval2", default_port=8104)
    geneval2_config = config_section("geneval2")
    parser.add_argument("--model-path", default=default_model_path())
    parser.add_argument(
        "--evaluator-concurrency",
        type=int,
        default=int(geneval2_config.get("evaluator_concurrency", 1)),
    )
    parser.add_argument("--dtype", default=str(geneval2_config.get("dtype", "bfloat16")))
    parser.add_argument(
        "--allow-download",
        action="store_true",
        help="Allow Hugging Face downloads instead of requiring local model files.",
    )
    parser.add_argument(
        "--reward-only",
        action="store_true",
        help="Disable the feedback verbalizer and accept only reward requests.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.evaluator_concurrency <= 0:
        raise ValueError("--evaluator-concurrency must be positive")
    configure_service_file_logging(
        "geneval2",
        log_file=args.service_log_file,
        record_file=args.feedback_record_file,
    )
    runtimes = [
        GenEval2Runtime(
            model_path=args.model_path,
            dtype=args.dtype,
            local_files_only=not args.allow_download,
        )
        for _ in range(args.evaluator_concurrency)
    ]
    for runtime in runtimes:
        runtime.load()
    agent = GenEval2FeedbackAgent(
        evaluator_runtimes=runtimes,
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

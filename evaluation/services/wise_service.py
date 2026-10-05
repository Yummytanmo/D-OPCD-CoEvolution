#!/usr/bin/env python3
"""Serve the official WISE_Verified judge as reward-only feedback."""

from __future__ import annotations

import argparse
import os

from evaluation.evaluators.wise_verified import WiseVerifiedRuntime
from evaluation.feedback_agents.wise import WiseFeedbackAgent
from evaluation.services.bootstrap import (
    add_server_arguments,
    configure_service_file_logging,
)
from evaluation.services.runtime import serve_feedback_agent


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    add_server_arguments(parser, benchmark="wise", default_port=8105)
    parser.add_argument(
        "--judge-api-base",
        default=os.getenv("WISE_JUDGE_API_BASE", "http://127.0.0.1:8205/v1"),
    )
    parser.add_argument(
        "--judge-model",
        default=os.getenv("WISE_JUDGE_MODEL", "Qwen3.5-35B-A3B"),
    )
    parser.add_argument(
        "--judge-api-key",
        default=os.getenv("WISE_JUDGE_API_KEY", "EMPTY"),
    )
    parser.add_argument("--judge-timeout-seconds", type=float, default=300.0)
    parser.add_argument("--max-extract-attempts", type=int, default=3)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    configure_service_file_logging(
        "wise",
        log_file=args.service_log_file,
        record_file=args.feedback_record_file,
    )
    runtime = WiseVerifiedRuntime(
        api_base=args.judge_api_base,
        model=args.judge_model,
        api_key=args.judge_api_key,
        timeout_seconds=args.judge_timeout_seconds,
        max_extract_attempts=args.max_extract_attempts,
    )
    runtime.check_ready(timeout_seconds=30.0)
    serve_feedback_agent(
        WiseFeedbackAgent(evaluator_runtime=runtime),
        host=args.host,
        port=args.port,
        warmup=False,
        feedback_modes={"reward"},
    )


if __name__ == "__main__":
    main()

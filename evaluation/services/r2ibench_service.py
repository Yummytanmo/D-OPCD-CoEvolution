"""Local R2I-Bench feedback and reward service."""

from __future__ import annotations

import argparse
import os

from evaluation.evaluators.r2ibench import R2IBenchRuntime
from evaluation.feedback_agents.r2ibench import R2IBenchFeedbackAgent
from evaluation.model_config import config_section
from evaluation.services.bootstrap import (
    add_server_arguments,
    configure_service_file_logging,
)
from evaluation.services.runtime import serve_feedback_agent


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    add_server_arguments(parser, benchmark="r2ibench", default_port=8108)
    settings = config_section("r2ibench")
    parser.add_argument(
        "--judge-api-base",
        default=os.getenv(
            "R2I_JUDGE_API_BASE", settings.get("api_base", "http://127.0.0.1:8208/v1")
        ),
    )
    parser.add_argument(
        "--judge-model",
        default=os.getenv(
            "R2I_JUDGE_MODEL", settings.get("model", "Qwen3-VL-32B-Instruct")
        ),
    )
    parser.add_argument("--judge-api-key", default=os.getenv("R2I_JUDGE_API_KEY", "EMPTY"))
    parser.add_argument(
        "--judge-timeout-seconds", type=float,
        default=float(settings.get("timeout_seconds", 900)),
    )
    parser.add_argument(
        "--judge-concurrency", type=int, default=int(settings.get("concurrency", 1)),
    )
    parser.add_argument(
        "--judge-cache-dir", default=settings.get("cache_dir", "runs/cache/r2ibench-judge"),
    )
    args = parser.parse_args()
    configure_service_file_logging(
        "r2ibench",
        log_file=args.service_log_file,
        record_file=args.feedback_record_file,
    )
    runtime = R2IBenchRuntime(
        api_base=args.judge_api_base,
        model=args.judge_model,
        api_key=args.judge_api_key,
        timeout_seconds=args.judge_timeout_seconds,
        max_concurrent_requests=args.judge_concurrency,
        cache_dir=args.judge_cache_dir,
    )
    runtime.check_ready(timeout_seconds=30)
    serve_feedback_agent(
        R2IBenchFeedbackAgent(evaluator_runtime=runtime),
        host=args.host,
        port=args.port,
        warmup=False,
        feedback_modes={"both", "reward"},
    )


if __name__ == "__main__":
    main()

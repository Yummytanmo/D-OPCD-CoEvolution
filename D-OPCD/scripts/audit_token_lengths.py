#!/usr/bin/env python3
"""Audit D-OPCD student q and teacher p (or legacy [q;p]) tokenizer lengths."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

from transformers import Qwen2Tokenizer


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from common import (  # noqa: E402
    DEFAULT_CONTEXT_LABEL, TEACHER_CONTEXT_MODES, compose_teacher_context,
    read_jsonl, teacher_context_description, write_json,
)
from prompt_encoder import render_chat_prompts  # noqa: E402
from qwen_image_prompt_encoder import (  # noqa: E402
    load_qwen_image_tokenizer,
    qwen_image_prompt_token_lengths,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--model-path", type=Path, required=True)
    parser.add_argument("--student-max", type=int, default=512)
    parser.add_argument("--teacher-max", type=int, default=1024)
    parser.add_argument("--context-label", default=DEFAULT_CONTEXT_LABEL)
    parser.add_argument("--teacher-context-mode", choices=TEACHER_CONTEXT_MODES, default="p_only")
    parser.add_argument(
        "--prompt-format",
        choices=["zimage", "qwen-image"],
        default="zimage",
        help="Apply the exact prompt wrapper used by the selected model backend.",
    )
    parser.add_argument("--report", type=Path, required=True)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    rows = read_jsonl(args.data.expanduser().resolve())
    if args.prompt_format == "qwen-image":
        tokenizer = load_qwen_image_tokenizer(args.model_path)
    else:
        tokenizer = Qwen2Tokenizer.from_pretrained(
            args.model_path.expanduser().resolve(),
            subfolder="tokenizer",
            local_files_only=True,
        )
    student_prompts = [str(row["original_query"]) for row in rows]
    teacher_prompts = [
        str(row["privileged_prompt"]).strip()
        if args.teacher_context_mode == "p_only"
        else compose_teacher_context(
            row["original_query"], row["privileged_prompt"], args.context_label
        )
        for row in rows
    ]
    if args.prompt_format == "qwen-image":
        student_lengths = qwen_image_prompt_token_lengths(tokenizer, student_prompts)
        teacher_lengths = qwen_image_prompt_token_lengths(tokenizer, teacher_prompts)
    else:
        student_tokens = tokenizer(
            render_chat_prompts(tokenizer, student_prompts),
            padding=False,
            truncation=False,
            return_attention_mask=False,
        )["input_ids"]
        teacher_tokens = tokenizer(
            render_chat_prompts(tokenizer, teacher_prompts),
            padding=False,
            truncation=False,
            return_attention_mask=False,
        )["input_ids"]
        student_lengths = [len(value) for value in student_tokens]
        teacher_lengths = [len(value) for value in teacher_tokens]
    student_over = [index for index, length in enumerate(student_lengths) if length > args.student_max]
    teacher_over = [index for index, length in enumerate(teacher_lengths) if length > args.teacher_max]
    report = {
        "schema_version": 1,
        "method": "D-OPCD",
        "count": len(rows),
        "prompt_format": args.prompt_format,
        "teacher_context": teacher_context_description(args.teacher_context_mode),
        "teacher_context_mode": args.teacher_context_mode,
        "student": {
            "max_allowed": args.student_max,
            "min": min(student_lengths),
            "max": max(student_lengths),
            "mean": sum(student_lengths) / len(student_lengths),
            "over_limit": len(student_over),
            "first_over_limit_rows": [index + 1 for index in student_over[:20]],
        },
        "teacher": {
            "max_allowed": args.teacher_max,
            "min": min(teacher_lengths),
            "max": max(teacher_lengths),
            "mean": sum(teacher_lengths) / len(teacher_lengths),
            "over_limit": len(teacher_over),
            "first_over_limit_rows": [index + 1 for index in teacher_over[:20]],
        },
        "valid": not student_over and not teacher_over,
        "resource_policy": {
            "tokenizer_only": True,
            "text_encoder_loaded": False,
            "diffusion_model_loaded": False,
            "training_run": False,
        },
    }
    write_json(args.report, report)
    print(args.report)
    if student_over or teacher_over:
        raise SystemExit(
            f"Token length audit failed: student_over={len(student_over)}, "
            f"teacher_over={len(teacher_over)}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python3
"""Generate benchmark split images with Z-Image-Turbo and an optional Student LoRA.

The script is dry-run by default. ``--execute`` loads Z-Image-Turbo and the
Student LoRA. Under Accelerate, every rank owns a deterministic manifest shard,
writes a resumable records file, and rank 0 merges the shards into the
``records.jsonl`` schema consumed by the benchmark staging scripts in
``evaluation/benchmarks/``.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
from pathlib import Path
import sys
import time
from typing import Any


SCRIPT_PATH = Path(__file__).resolve()
PROJECT_ROOT = SCRIPT_PATH.parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from common import read_jsonl, write_json, write_jsonl  # noqa: E402


LOGGER = logging.getLogger("dopcd.benchmark.inference")
BENCHMARKS = (
    "geneval", "geneval2", "wise", "r2ibench",
    "t2i-compbench-plusplus",
)


def positive_int(value: str) -> int:
    parsed = int(value)
    if parsed < 1:
        raise argparse.ArgumentTypeError("value must be >= 1")
    return parsed


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-path", type=Path, required=True)
    parser.add_argument("--lora-path", type=Path)
    parser.add_argument(
        "--no-lora",
        action="store_true",
        help="Run the unmodified Z-Image-Turbo baseline without loading an adapter.",
    )
    parser.add_argument("--data-path", type=Path, required=True)
    parser.add_argument("--run-directory", type=Path, required=True)
    parser.add_argument("--benchmark", choices=BENCHMARKS, default="geneval")
    parser.add_argument("--split", choices=("train", "validation", "test"), default="test")
    parser.add_argument("--method", default="dopcd")
    parser.add_argument(
        "--prompt-source",
        choices=(
            "original", "q", "skill_p", "scaffold_p", "skill_qp", "scaffold_qp",
        ),
        default="original",
        help="Provenance label for the manifest's generation prompt.",
    )
    parser.add_argument("--expected-count", type=positive_int, required=True)
    parser.add_argument("--width", type=positive_int, default=1024)
    parser.add_argument("--height", type=positive_int, default=1024)
    parser.add_argument("--num-inference-steps", type=positive_int, default=8)
    parser.add_argument("--guidance-scale", type=float, default=0.0)
    parser.add_argument("--max-sequence-length", type=positive_int, default=512)
    parser.add_argument("--lora-scale", type=float, default=1.0)
    parser.add_argument(
        "--adapter-role", choices=("student", "ema_teacher"), default="student",
        help="Provenance role of the loaded LoRA; default preserves existing student evaluations.",
    )
    parser.add_argument(
        "--seed-override",
        type=int,
        help=(
            "Use one registered evaluation seed for every manifest row. "
            "The default preserves each row's inference_seed."
        ),
    )
    parser.add_argument("--log-every", type=positive_int, default=10)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--execute", action="store_true")
    return parser.parse_args()


def setup_logging() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        force=True,
    )


def generation_config(args: argparse.Namespace, seed: int) -> dict[str, Any]:
    return {
        "guidance_scale": args.guidance_scale,
        "height": args.height,
        "lora_scale": None if args.no_lora else args.lora_scale,
        "max_sequence_length": args.max_sequence_length,
        "num_inference_steps": args.num_inference_steps,
        "scheduler": "z-image-turbo-default",
        "seed": seed,
        "width": args.width,
    }


def record_seed(source: dict[str, Any], args: argparse.Namespace) -> int:
    override = getattr(args, "seed_override", None)
    seed = (
        int(override)
        if override is not None
        else int(source.get("inference_seed", 42))
    )
    if seed < 0:
        raise ValueError("generation seed must be nonnegative")
    return seed


def image_name(sample_id: str) -> str:
    return sample_id.replace(":", "_").replace("/", "_") + ".png"


def manifest_sample_id(source: dict[str, Any]) -> str:
    """Accept both the current source schema and the legacy eval schema."""

    return str(source.get("source_sample_id") or source.get("sample_id") or "")


def record_is_resumable(
    previous: dict[str, Any] | None,
    source: dict[str, Any],
    args: argparse.Namespace,
    lora_path: Path | None,
) -> bool:
    if not previous or previous.get("status") != "complete":
        return False
    image_path = Path(str(previous.get("image_path") or ""))
    if not image_path.is_file() or image_path.stat().st_size == 0:
        return False
    seed = record_seed(source, args)
    return (
        previous.get("sample_id") == manifest_sample_id(source)
        and previous.get("method") == args.method
        and previous.get("split") == args.split
        and previous.get("original_prompt") == source.get("original_query", source["prompt"])
        and previous.get("generation_prompt") == source["prompt"]
        and previous.get("prompt_source") == args.prompt_source
        and previous.get("generation") == generation_config(args, seed)
        and previous.get("student_lora_path")
        == (None if lora_path is None else str(lora_path))
        and (previous.get("adapter_role") or "student") == args.adapter_role
    )


def load_previous(paths: list[Path]) -> dict[str, dict[str, Any]]:
    output: dict[str, dict[str, Any]] = {}
    for path in paths:
        if not path.is_file():
            continue
        for row in read_jsonl(path):
            sample_id = str(row.get("sample_id") or "")
            if sample_id:
                output[sample_id] = row
    return output


def validate_manifest(
    path: Path,
    expected_count: int,
    expected_split: str,
    expected_benchmark: str = "geneval",
) -> list[dict[str, Any]]:
    records = read_jsonl(path)
    if len(records) != expected_count:
        raise ValueError(
            f"{expected_benchmark} {expected_split} count={len(records)}, "
            f"expected={expected_count}"
        )
    sample_ids: set[str] = set()
    for index, record in enumerate(records, start=1):
        benchmark = record.get("benchmark") or record.get("dataset_id")
        if benchmark != expected_benchmark or record.get("split") != expected_split:
            raise ValueError(
                f"{path}:{index}: expected benchmark={expected_benchmark}, "
                f"split={expected_split}"
            )
        sample_id = manifest_sample_id(record)
        prompt = str(record.get("prompt") or "").strip()
        if not sample_id or not prompt:
            raise ValueError(f"{path}:{index}: missing sample_id/prompt")
        if sample_id in sample_ids:
            raise ValueError(f"{path}:{index}: duplicate sample_id={sample_id}")
        sample_ids.add(sample_id)
    return records


def write_image_atomic(image: Any, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        image.save(temporary, format="PNG")
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def main() -> int:
    args = parse_args()
    setup_logging()
    if not args.no_lora and args.lora_path is None:
        raise ValueError("--lora-path is required unless --no-lora is set")
    model_path = args.model_path.expanduser().resolve()
    lora_path = None if args.no_lora else args.lora_path.expanduser().resolve()
    data_path = args.data_path.expanduser().resolve()
    run_directory = args.run_directory.expanduser().resolve()
    manifest = validate_manifest(
        data_path, args.expected_count, args.split, args.benchmark
    )
    plan = {
        "schema_version": 1,
        "method": args.method,
        "benchmark": args.benchmark,
        "split": args.split,
        "model_path": str(model_path),
        "inference_variant": (
            "zimage-turbo-baseline" if args.no_lora
            else "ema-teacher-lora" if args.adapter_role == "ema_teacher"
            else "student-lora"
        ),
        "adapter_role": args.adapter_role,
        "student_lora_path": None if lora_path is None else str(lora_path),
        "student_lora_exists": None if lora_path is None else lora_path.is_file(),
        "student_lora_readable_in_current_process": (
            None if lora_path is None else os.access(lora_path, os.R_OK)
        ),
        "data_path": str(data_path),
        "sample_count": len(manifest),
        "run_directory": str(run_directory),
        "generation": generation_config(args, 42)
        | {
            "seed": (
                args.seed_override
                if args.seed_override is not None
                else "manifest:inference_seed"
            )
        },
        "prompt_source": args.prompt_source,
        "execute_requested": args.execute,
        "inference_executed": False,
    }
    if not args.execute:
        print(json.dumps(plan, ensure_ascii=False, indent=2, sort_keys=True))
        return 0
    if not model_path.is_dir():
        raise FileNotFoundError(f"Base model is missing: {model_path}")
    if lora_path is not None:
        if not lora_path.is_file():
            raise FileNotFoundError(f"LoRA is missing: {lora_path}")
        if not os.access(lora_path, os.R_OK):
            raise PermissionError(
                f"LoRA is not readable by uid={os.geteuid()}: {lora_path}. "
                "Make the checkpoint readable before running inference."
            )

    import torch
    from accelerate import Accelerator
    from diffusers import ZImagePipeline
    from tqdm.auto import tqdm

    accelerator = Accelerator(mixed_precision="bf16")
    rank = accelerator.process_index
    world_size = accelerator.num_processes
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for D-OPCD benchmark inference")
    run_directory.mkdir(parents=True, exist_ok=True)
    shard_path = run_directory / f"records.rank-{rank:02d}.jsonl"
    central_path = run_directory / "records.jsonl"
    assigned = [record for index, record in enumerate(manifest) if index % world_size == rank]
    assigned_ids = {manifest_sample_id(record) for record in assigned}
    previous = load_previous([central_path, shard_path])
    shard_records = {
        sample_id: record
        for sample_id, record in previous.items()
        if sample_id in assigned_ids
    }
    pending = [
        record
        for record in assigned
        if args.overwrite
        or not record_is_resumable(
            shard_records.get(manifest_sample_id(record)), record, args, lora_path
        )
    ]
    resumed = len(assigned) - len(pending)
    LOGGER.info(
        "rank_start rank=%d/%d assigned=%d resumed=%d pending=%d device=%s",
        rank,
        world_size,
        len(assigned),
        resumed,
        len(pending),
        accelerator.device,
    )

    pipeline = ZImagePipeline.from_pretrained(
        str(model_path),
        torch_dtype=torch.bfloat16,
        local_files_only=True,
    )
    if lora_path is not None:
        adapter_name = "dopcd_teacher" if args.adapter_role == "ema_teacher" else "dopcd_student"
        pipeline.load_lora_weights(
            str(lora_path.parent),
            weight_name=lora_path.name,
            adapter_name=adapter_name,
            local_files_only=True,
        )
        pipeline.set_adapters(adapter_name, adapter_weights=args.lora_scale)
    pipeline.to(accelerator.device)
    pipeline.set_progress_bar_config(disable=True)

    progress = tqdm(
        total=len(assigned),
        initial=resumed,
        desc=f"D-OPCD {args.benchmark} rank {rank}",
        unit="img",
        dynamic_ncols=True,
        disable=not accelerator.is_local_main_process,
    )
    started = time.monotonic()
    for processed, source in enumerate(pending, start=1):
        sample_id = manifest_sample_id(source)
        prompt = str(source["prompt"])
        seed = record_seed(source, args)
        generation = generation_config(args, seed)
        image_path = run_directory / "images" / image_name(sample_id)
        sample_started = time.time()
        generator = torch.Generator(device=accelerator.device).manual_seed(seed)
        with torch.inference_mode():
            image = pipeline(
                prompt=prompt,
                width=args.width,
                height=args.height,
                num_inference_steps=args.num_inference_steps,
                guidance_scale=args.guidance_scale,
                max_sequence_length=args.max_sequence_length,
                generator=generator,
            ).images[0]
        write_image_atomic(image, image_path)
        shard_records[sample_id] = {
            "sample_id": sample_id,
            "benchmark": args.benchmark,
            "split": args.split,
            "method": args.method,
            "status": "complete",
            "original_prompt": str(source.get("original_query", prompt)),
            "generation_prompt": prompt,
            "prompt_source": args.prompt_source,
            "student_lora_path": None if lora_path is None else str(lora_path),
            "adapter_role": args.adapter_role,
            "generation": generation,
            "image_path": str(image_path),
            "started_at_unix": sample_started,
            "finished_at_unix": time.time(),
            "input_record": source,
        }
        write_jsonl(
            shard_path,
            [
                shard_records[manifest_sample_id(record)]
                for record in assigned
                if manifest_sample_id(record) in shard_records
            ],
        )
        progress.update(1)
        if processed % args.log_every == 0 or processed == len(pending):
            elapsed = max(time.monotonic() - started, 1e-9)
            LOGGER.info(
                "rank_progress rank=%d processed=%d/%d complete=%d/%d rate=%.3f_img_s last=%s",
                rank,
                processed,
                len(pending),
                resumed + processed,
                len(assigned),
                processed / elapsed,
                sample_id,
            )
    progress.close()
    accelerator.wait_for_everyone()

    if accelerator.is_main_process:
        merged: dict[str, dict[str, Any]] = {}
        for shard_rank in range(world_size):
            path = run_directory / f"records.rank-{shard_rank:02d}.jsonl"
            if not path.is_file():
                raise RuntimeError(f"Missing completed inference shard: {path}")
            for record in read_jsonl(path):
                merged[str(record["sample_id"])] = record
        missing = [
            manifest_sample_id(record)
            for record in manifest
            if manifest_sample_id(record) not in merged
        ]
        if missing:
            raise RuntimeError(f"Incomplete inference; missing {len(missing)} samples: {missing[:10]}")
        ordered = [merged[manifest_sample_id(record)] for record in manifest]
        write_jsonl(central_path, ordered)
        plan.update(
            {
                "world_size": world_size,
                "inference_executed": True,
                "completed_count": len(ordered),
                "records_path": str(central_path),
            }
        )
        write_json(run_directory / "inference_summary.json", plan)
        LOGGER.info("inference_complete total=%d records=%s", len(ordered), central_path)
    accelerator.wait_for_everyone()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

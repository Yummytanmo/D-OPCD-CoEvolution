"""Generate and stage checkpoint images without a shell launcher."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys

if not __package__:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from runners.common import ROOT, digest, executable, path, read_json, read_jsonl, run_logged, write_json
sys.path.insert(1, str(Path(__file__).resolve().parents[2]))
from evaluation.benchmarks.common import stage_checkpoint


DEFAULT_GENERATION = {
    "model_path": os.environ.get("Z_IMAGE_MODEL_PATH", "models/Z-Image-Turbo"),
    "accelerate": ".venv/bin/accelerate",
    "width": 1024,
    "height": 1024,
    "num_inference_steps": 8,
    "guidance_scale": 0.0,
    "max_sequence_length": 512,
    "lora_scale": 1.0,
}


def generation_command(task: dict, *, port: int = 29501) -> list[str]:
    settings = DEFAULT_GENERATION | task.get("generation", {})
    processes = int(settings.get("num_processes", 1))
    if processes < 1:
        raise ValueError("num_processes must be positive")
    accelerate = executable(settings["accelerate"])
    accelerate_config = path(
        "configs/accelerate_single_gpu.yaml" if processes == 1
        else "configs/accelerate_ddp.yaml"
    )
    return [
        str(accelerate), "launch", "--config_file", str(accelerate_config),
        "--num_processes", str(processes),
        "--main_process_port", str(port),
        str(ROOT / "scripts/infer_geneval_test.py"),
        "--model-path", str(path(settings["model_path"])),
        "--lora-path", str(Path(task["checkpoint"])),
        "--benchmark", task["benchmark"],
        "--data-path", task["manifest"],
        "--run-directory", str(Path(task["evaluation_dir"]) / "generation"),
        "--split", task["split"],
        "--method", f"dopcd-{task['run_id']}-checkpoint-{task['step']}",
        "--expected-count", str(task["expected_count"]),
        "--width", str(settings.get("width", 1024)),
        "--height", str(settings.get("height", 1024)),
        "--num-inference-steps", str(settings.get("num_inference_steps", 8)),
        "--guidance-scale", str(settings.get("guidance_scale", 0.0)),
        "--max-sequence-length", str(settings.get("max_sequence_length", 512)),
        "--lora-scale", str(settings.get("lora_scale", 1.0)),
        "--execute",
    ]


def stage_direct(task: dict) -> None:
    """WISE and R2I use the sealed source order and no GEMS staging schema."""
    dest = Path(task["evaluation_dir"])
    manifest = read_jsonl(Path(task["manifest"]))
    records = read_jsonl(dest / "generation" / "records.jsonl")
    count = task["expected_count"]
    if len(manifest) != count or len(records) != count:
        raise ValueError("Generation/manifest coverage mismatch")
    by_id = {str(row["sample_id"]): row for row in records}
    expected_ids = [str(row.get("source_sample_id") or row.get("sample_id") or "")
                    for row in manifest]
    if "" in expected_ids or set(by_id) != set(expected_ids) or len(by_id) != count:
        raise ValueError("Generated sample IDs do not match manifest")
    images = dest / "images"
    images.mkdir(parents=True, exist_ok=True)
    for index, row in enumerate(manifest):
        record = by_id[expected_ids[index]]
        source = Path(record["image_path"]).resolve()
        if (record.get("status") != "complete" or
                record.get("original_prompt") != row.get("prompt") or
                not source.is_file() or source.stat().st_size == 0):
            raise ValueError(f"Invalid generated image for {expected_ids[index]}")
        if task["benchmark"] == "wise":
            name = str(row["source_metadata"]["prompt_id"]) + ".png"
        else:
            name = f"{index:05d}.png"
        target = images / name
        if target.is_symlink():
            if target.resolve() != source:
                raise ValueError(f"Conflicting staged image: {target}")
        elif target.exists():
            if digest(target) != digest(source):
                raise ValueError(f"Conflicting staged image: {target}")
        else:
            target.symlink_to(source)
    if len(list(images.glob("*.png"))) != count:
        raise ValueError("Staged image count mismatch")


def run(task: dict, *, env: dict[str, str] | None = None, port: int = 29501) -> None:
    dest = Path(task["evaluation_dir"])
    checkpoint = Path(task["checkpoint"])
    if digest(checkpoint) != task["checkpoint_sha256"]:
        raise ValueError("Checkpoint changed before generation")
    if digest(Path(task["manifest"])) != task["manifest_sha256"]:
        raise ValueError("Manifest changed before generation")
    environment = (env or os.environ).copy()
    for key in ("WORLD_SIZE", "RANK", "LOCAL_RANK", "LOCAL_WORLD_SIZE",
                "MASTER_ADDR", "MASTER_PORT"):
        environment.pop(key, None)
    environment.update(DOPCD_ROOT=str(ROOT), HF_HUB_OFFLINE="1",
                       TRANSFORMERS_OFFLINE="1", TOKENIZERS_PARALLELISM="false")
    run_logged(generation_command(task, port=port), log=dest / "metadata" / "generation.log",
               env=environment)
    if task["benchmark"] in {"geneval", "geneval2"}:
        plan = stage_checkpoint(task)
        with (dest / "metadata" / "staging.log").open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(plan, ensure_ascii=False, indent=2, sort_keys=True) + "\n")
    else:
        stage_direct(task)
    records = read_jsonl(dest / "generation" / "records.jsonl")
    if len(records) != task["expected_count"]:
        raise ValueError("Generated record coverage is incomplete")
    images = {}
    image_root = (dest / "generation" / "images").resolve()
    for record in records:
        source = Path(record["image_path"]).resolve()
        if not source.is_relative_to(image_root) or not source.is_file():
            raise ValueError(f"Generated image escapes or is missing: {source}")
        images[str(record["sample_id"])] = {"path": str(source), "sha256": digest(source)}
    if len(images) != task["expected_count"]:
        raise ValueError("Generated sample IDs are duplicated")
    write_json(dest / "metadata" / "generation-stage.json",
               {"status": "completed", "sample_count": task["expected_count"],
                "checkpoint_sha256": task["checkpoint_sha256"],
                "manifest_sha256": task["manifest_sha256"], "images": images})


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task", type=Path, required=True)
    args = parser.parse_args()
    run(read_json(args.task))


if __name__ == "__main__":
    main()

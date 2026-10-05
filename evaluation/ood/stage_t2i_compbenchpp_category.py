#!/usr/bin/env python3
"""Validate one generated T2I-CompBench++ category and hard-link official inputs."""

from __future__ import annotations

import argparse
from hashlib import sha256
import json
import os
from pathlib import Path


def digest(path: Path) -> str:
    value = sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            value.update(chunk)
    return value.hexdigest()


def rows(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def write_jsonl(path: Path, values: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text("".join(json.dumps(v, sort_keys=True, separators=(",", ":")) + "\n" for v in values))
    temporary.replace(path)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--category", required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--generation-dir", type=Path, required=True)
    parser.add_argument("--official-dir", type=Path, required=True)
    parser.add_argument("--metadata", type=Path, required=True)
    args = parser.parse_args()
    manifest = rows(args.manifest.resolve(strict=True))
    generated = rows((args.generation_dir / "records.jsonl").resolve(strict=True))
    if len(manifest) != 3000 or len(generated) != 3000:
        raise ValueError("A category must contain exactly 300 prompts x 10 images")
    generated_by_id = {row["sample_id"]: row for row in generated}
    if len(generated_by_id) != len(generated):
        raise ValueError("Generated sample IDs are not unique")
    samples = args.official_dir / "samples"
    if args.official_dir.exists():
        raise ValueError(f"Refusing to overwrite official staging: {args.official_dir}")
    samples.mkdir(parents=True)
    staged = []
    for expected_index, source in enumerate(manifest):
        if source["category"] != args.category:
            raise ValueError("Category manifest contains another category")
        sample_id = source["source_sample_id"]
        record = generated_by_id.get(sample_id)
        if record is None or record.get("status") != "complete":
            raise ValueError(f"Missing complete generation record: {sample_id}")
        prompt = source["prompt"]
        seed = int(source["inference_seed"])
        if (
            record.get("original_prompt") != prompt
            or record.get("generation_prompt") != prompt
            or int((record.get("generation") or {}).get("seed", -1)) != seed
            or record.get("input_record") != source
        ):
            raise ValueError(f"Generation provenance mismatch: {sample_id}")
        image = Path(record["image_path"]).resolve(strict=True)
        if image.stat().st_size == 0:
            raise ValueError(f"Empty generated image: {image}")
        if any(char in prompt for char in ("/", "\\", "_", "\n", "\r")):
            raise ValueError(f"Prompt is unsafe for the official filename protocol: {prompt!r}")
        official_name = f"{prompt}_{expected_index:06d}.png"
        official_path = samples / official_name
        os.link(image, official_path)
        staged.append({
            "category": args.category,
            "source_prompt_id": source["source_prompt_id"],
            "source_sample_id": sample_id,
            "prompt": prompt,
            "prompt_index": int(source["prompt_index"]),
            "sample_index": int(source["sample_index"]),
            "seed": seed,
            "question_id": expected_index,
            "generated_image_path": str(image),
            "generated_image_sha256": digest(image),
            "official_image_path": str(official_path),
            "retained_after_evaluation": int(source["sample_index"]) == 0,
        })
    write_jsonl(args.metadata, staged)
    print(json.dumps({"category": args.category, "staged": len(staged), "status": "completed"}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

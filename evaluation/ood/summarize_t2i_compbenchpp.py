#!/usr/bin/env python3
"""Validate official T2I-CompBench++ raw outputs and write durable summaries."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import shutil


RAW = {
    "color": "annotation_blip/vqa_result.json",
    "shape": "annotation_blip/vqa_result.json",
    "texture": "annotation_blip/vqa_result.json",
    "spatial": "labels/annotation_obj_detection_2d/vqa_result.json",
    "3d_spatial": "labels/annotation_obj_detection_3d/vqa_result.json",
    "numeracy": "annotation_num/vqa_result.json",
    "non_spatial": "annotation_clip/vqa_result.json",
    "complex": "annotation_3_in_1/vqa_result.json",
}


def write(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)


def summarize_category(category: str, official_dir: Path, output: Path) -> None:
    raw_path = official_dir / RAW[category]
    values = json.loads(raw_path.read_text(encoding="utf-8"))
    if not isinstance(values, list) or len(values) != 3000:
        raise ValueError(f"{category} raw result must contain 3000 entries")
    answers = []
    for index, row in enumerate(values):
        if int(row.get("question_id", -1)) != index:
            raise ValueError(f"{category} question IDs are not ordered 0..2999")
        value = float(row["answer"])
        if not math.isfinite(value):
            raise ValueError(f"{category} contains a non-finite score")
        answers.append(value)
    raw_out = output / "raw" / "vqa_result.json"
    raw_out.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(raw_path, raw_out)
    write(output / "metrics.json", {
        "schema_version": 1,
        "status": "completed",
        "benchmark": "t2i-compbench-plusplus",
        "category": category,
        "prompt_count": 300,
        "image_count": 3000,
        "images_per_prompt": 10,
        "seed_range": [42, 51],
        "score": sum(answers) / len(answers),
        "raw_result": str(raw_out),
    })


def summarize_model(model_dir: Path) -> None:
    category_scores = {}
    for category in RAW:
        metrics = json.loads((model_dir / "categories" / category / "scores" / "metrics.json").read_text())
        if metrics.get("status") != "completed" or int(metrics.get("image_count", 0)) != 3000:
            raise ValueError(f"Incomplete category metrics: {category}")
        category_scores[category] = float(metrics["score"])
    write(model_dir / "scores" / "metrics.json", {
        "schema_version": 1,
        "status": "completed",
        "benchmark": "t2i-compbench-plusplus",
        "official_split": "validation",
        "use_as": "held-out-ood-test",
        "prompt_count": 2400,
        "image_count": 24000,
        "images_per_prompt": 10,
        "category_scores": category_scores,
        "mean_8_nonpaper_aggregate": sum(category_scores.values()) / len(category_scores),
    })


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="mode", required=True)
    category = sub.add_parser("category")
    category.add_argument("--category", choices=tuple(RAW), required=True)
    category.add_argument("--official-dir", type=Path, required=True)
    category.add_argument("--output-dir", type=Path, required=True)
    model = sub.add_parser("model")
    model.add_argument("--model-dir", type=Path, required=True)
    args = parser.parse_args()
    if args.mode == "category":
        summarize_category(args.category, args.official_dir, args.output_dir)
    else:
        summarize_model(args.model_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

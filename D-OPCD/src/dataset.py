from __future__ import annotations

import hashlib
import math
from pathlib import Path
from typing import Any

import torch
from PIL import Image
from torch.utils.data import Dataset

from common import (
    DEFAULT_CONTEXT_LABEL,
    TEACHER_CONTEXT_MODES,
    compose_teacher_context,
    read_jsonl,
)

IMAGE_MANIFEST_FIELDS = {"sample_id", "target_image", "target_prompt_role", "target_sha256"}


def load_sample_weights(path: str | Path) -> dict[int, float]:
    source = Path(path).expanduser().resolve()
    rows = read_jsonl(source)
    weights: dict[int, float] = {}
    for line_number, row in enumerate(rows, start=1):
        sample_id = row.get("sample_id")
        value = row.get("sample_weight")
        if not isinstance(sample_id, int) or sample_id < 0 or sample_id in weights:
            raise ValueError(f"{source}:{line_number}: invalid/duplicate sample_id")
        if not isinstance(value, (int, float)) or not math.isfinite(float(value)) or value <= 0:
            raise ValueError(f"{source}:{line_number}: sample_weight must be finite and > 0")
        weights[sample_id] = float(value)
    if not weights:
        raise ValueError(f"Sample-weight file is empty: {source}")
    return weights


class PromptContextDataset(Dataset):
    """Student sees q; teacher sees p, [q;p], q, or a VLM text+image context."""

    def __init__(
        self,
        jsonl_path: str | Path,
        context_label: str = DEFAULT_CONTEXT_LABEL,
        sample_weights_jsonl: str | Path | None = None,
        teacher_context_mode: str = "p_only",
        image_manifest_jsonl: str | Path | None = None,
        resolution: int = 1024,
    ):
        self.jsonl_path = Path(jsonl_path).expanduser().resolve()
        self.context_label = str(context_label).strip()
        if teacher_context_mode not in TEACHER_CONTEXT_MODES:
            raise ValueError(f"Unknown teacher context mode: {teacher_context_mode}")
        self.teacher_context_mode = teacher_context_mode
        self.resolution = int(resolution)
        self.rows = read_jsonl(self.jsonl_path)
        if not self.rows:
            raise ValueError(f"Dataset is empty: {self.jsonl_path}")
        self.images: dict[str, Path] = {}
        if teacher_context_mode in {"vlm_q_image", "vlm_q_p_image"}:
            if image_manifest_jsonl is None:
                raise ValueError(f"{teacher_context_mode} requires image_manifest_jsonl")
            manifest_path = Path(image_manifest_jsonl).expanduser().resolve()
            manifest_rows = read_jsonl(manifest_path)
            context_ids = {str(row["sample_id"]) for row in self.rows}
            manifest_ids = {str(row["sample_id"]) for row in manifest_rows}
            if context_ids != manifest_ids:
                missing = sorted(context_ids - manifest_ids)
                extra = sorted(manifest_ids - context_ids)
                raise ValueError(
                    "Image manifest must match the context dataset exactly; "
                    f"missing={missing[:5]}, extra={extra[:5]}"
                )
            for row in manifest_rows:
                if set(row) != IMAGE_MANIFEST_FIELDS:
                    raise ValueError(
                        f"Image manifest row fields={sorted(row)}, "
                        f"expected={sorted(IMAGE_MANIFEST_FIELDS)}"
                    )
                if row["target_prompt_role"] != "privileged_prompt":
                    raise ValueError(
                        f"sample_id={row['sample_id']}: target_prompt_role must be "
                        f"'privileged_prompt', got {row['target_prompt_role']!r}"
                    )
                image_path = Path(row["target_image"])
                if not image_path.is_absolute():
                    image_path = manifest_path.parent / image_path
                image_path = image_path.resolve()
                if not image_path.is_file():
                    raise ValueError(f"Missing target image: {image_path}")
                digest = hashlib.sha256(image_path.read_bytes()).hexdigest()
                if digest != row["target_sha256"]:
                    raise ValueError(f"Target image hash mismatch: {image_path}")
                self.images[str(row["sample_id"])] = image_path
        elif image_manifest_jsonl is not None:
            raise ValueError("image_manifest_jsonl is only valid for VLM image modes")
        self.sample_weights_path = (
            Path(sample_weights_jsonl).expanduser().resolve()
            if sample_weights_jsonl is not None
            else None
        )
        if self.sample_weights_path is None:
            self.sample_weights = {int(row["sample_id"]): 1.0 for row in self.rows}
        else:
            self.sample_weights = load_sample_weights(self.sample_weights_path)
            dataset_ids = {int(row["sample_id"]) for row in self.rows}
            weight_ids = set(self.sample_weights)
            if dataset_ids != weight_ids:
                missing = sorted(dataset_ids - weight_ids)
                extra = sorted(weight_ids - dataset_ids)
                raise ValueError(
                    "Sample-weight IDs do not match the dataset; "
                    f"missing={missing[:5]}, extra={extra[:5]}"
                )

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int) -> dict[str, Any]:
        row = self.rows[index]
        query = str(row["original_query"]).strip()
        privileged = str(row["privileged_prompt"]).strip()
        if not query or not privileged:
            raise ValueError(f"sample_id={row['sample_id']}: q and p must be non-empty")
        teacher_prompt = {
            "p_only": lambda: privileged,
            "q_plus_p": lambda: compose_teacher_context(query, privileged, self.context_label),
            "q_only": lambda: query,
            "vlm_q_image": lambda: query,
            "vlm_q_p_image": lambda: compose_teacher_context(query, privileged, self.context_label),
        }[self.teacher_context_mode]()
        item = {
            "sample_id": int(row["sample_id"]),
            "tag": str(row["tag"]),
            "student_prompt": query,
            "privileged_prompt": privileged,
            "teacher_prompt": teacher_prompt,
            "sample_weight": self.sample_weights[int(row["sample_id"])],
        }
        if self.teacher_context_mode in {"vlm_q_image", "vlm_q_p_image"}:
            image_path = self.images[str(row["sample_id"])]
            with Image.open(image_path) as image:
                image = image.convert("RGB").resize(
                    (self.resolution, self.resolution), Image.Resampling.LANCZOS
                )
                pixels = torch.ByteTensor(torch.ByteStorage.from_buffer(image.tobytes()))
            pixels = pixels.reshape(self.resolution, self.resolution, 3).permute(2, 0, 1)
            item["teacher_image"] = pixels.float().div(255.0).mul(2.0).sub(1.0)
        return item


def collate_prompt_contexts(examples: list[dict[str, Any]]) -> dict[str, Any]:
    batch = {
        "sample_ids": [example["sample_id"] for example in examples],
        "tags": [example["tag"] for example in examples],
        "student_prompts": [example["student_prompt"] for example in examples],
        "privileged_prompts": [example["privileged_prompt"] for example in examples],
        "teacher_prompts": [example["teacher_prompt"] for example in examples],
        "sample_weights": [example["sample_weight"] for example in examples],
    }
    if "teacher_image" in examples[0]:
        batch["teacher_images"] = torch.stack([example["teacher_image"] for example in examples])
    return batch

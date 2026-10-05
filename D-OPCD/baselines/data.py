from __future__ import annotations

import math
import random
import sys
from pathlib import Path
from typing import Any, Literal

import numpy as np
import torch
from PIL import Image, ImageOps
from torch.utils.data import Dataset


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from common import read_jsonl, sha256_file  # noqa: E402


BaselineMethod = Literal["sft", "flow_dpo"]
CONTEXT_FIELDS = {
    "sample_id",
    "source_sample_id",
    "tag",
    "original_query",
    "privileged_prompt",
}
SFT_MANIFEST_FIELDS = {
    "sample_id",
    "target_image",
    "target_sha256",
    "target_prompt_role",
}
DPO_MANIFEST_FIELDS = {
    "sample_id",
    "chosen_image",
    "chosen_sha256",
    "rejected_image",
    "rejected_sha256",
    "chosen_prompt_role",
    "rejected_prompt_role",
}


def _sample_id(row: dict[str, Any], source: Path, line_number: int) -> int:
    value = row.get("sample_id")
    if not isinstance(value, int) or isinstance(value, bool) or value < 0:
        raise ValueError(f"{source}:{line_number}: sample_id must be a non-negative integer")
    return value


def _nonempty_text(row: dict[str, Any], field: str, source: Path, line_number: int) -> str:
    value = row.get(field)
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{source}:{line_number}: {field} must be a non-empty string")
    return value.strip()


def _index_rows(path: Path, required_fields: set[str]) -> tuple[list[int], dict[int, dict[str, Any]]]:
    rows = read_jsonl(path)
    if not rows:
        raise ValueError(f"Dataset is empty: {path}")
    order: list[int] = []
    indexed: dict[int, dict[str, Any]] = {}
    for line_number, row in enumerate(rows, start=1):
        missing = sorted(required_fields - row.keys())
        if missing:
            raise ValueError(f"{path}:{line_number}: missing fields {missing}")
        sample_id = _sample_id(row, path, line_number)
        if sample_id in indexed:
            raise ValueError(f"{path}:{line_number}: duplicate sample_id={sample_id}")
        order.append(sample_id)
        indexed[sample_id] = row
    return order, indexed


def load_sample_weights(path: str | Path, expected_ids: set[int]) -> dict[int, float]:
    source = Path(path).expanduser().resolve()
    rows = read_jsonl(source)
    weights: dict[int, float] = {}
    for line_number, row in enumerate(rows, start=1):
        sample_id = _sample_id(row, source, line_number)
        value = row.get("sample_weight")
        if sample_id in weights:
            raise ValueError(f"{source}:{line_number}: duplicate sample_id={sample_id}")
        if not isinstance(value, (int, float)) or isinstance(value, bool):
            raise ValueError(f"{source}:{line_number}: sample_weight must be numeric")
        weight = float(value)
        if not math.isfinite(weight) or weight <= 0:
            raise ValueError(f"{source}:{line_number}: sample_weight must be finite and > 0")
        weights[sample_id] = weight
    if set(weights) != expected_ids:
        missing = sorted(expected_ids - set(weights))
        extra = sorted(set(weights) - expected_ids)
        raise ValueError(
            f"Sample-weight IDs do not match the context dataset; missing={missing[:5]}, extra={extra[:5]}"
        )
    return weights


def _resolve_image(path_value: str, manifest_path: Path, image_root: Path) -> Path:
    candidate = Path(path_value).expanduser()
    if not candidate.is_absolute():
        candidate = manifest_path.parent / candidate
    resolved = candidate.resolve()
    root = image_root.resolve()
    if resolved != root and root not in resolved.parents:
        raise ValueError(f"Baseline image must stay inside {root}: {resolved}")
    if not resolved.is_file():
        raise FileNotFoundError(f"Baseline image does not exist: {resolved}")
    return resolved


def _verify_image_hash(
    image_path: Path,
    expected_value: Any,
    *,
    sample_id: int,
    field: str,
) -> None:
    expected = str(expected_value).strip()
    if (
        expected != expected.lower()
        or len(expected) != 64
        or any(character not in "0123456789abcdef" for character in expected)
    ):
        raise ValueError(f"sample_id={sample_id}: {field} must be a lowercase SHA256 hex digest")
    actual = sha256_file(image_path)
    if actual != expected:
        raise ValueError(
            f"sample_id={sample_id}: {field} mismatch for {image_path}; "
            f"expected={expected}, actual={actual}"
        )


def load_joined_rows(
    context_jsonl: str | Path,
    image_manifest_jsonl: str | Path,
    method: BaselineMethod,
    *,
    sample_weights_jsonl: str | Path | None = None,
    image_root: str | Path | None = None,
) -> list[dict[str, Any]]:
    """Join immutable D-OPCD contexts to image supervision by exact sample_id."""
    context_path = Path(context_jsonl).expanduser().resolve()
    manifest_path = Path(image_manifest_jsonl).expanduser().resolve()
    required = SFT_MANIFEST_FIELDS if method == "sft" else DPO_MANIFEST_FIELDS
    context_order, contexts = _index_rows(context_path, CONTEXT_FIELDS)
    _, images = _index_rows(manifest_path, required)

    context_ids = set(contexts)
    image_ids = set(images)
    if context_ids != image_ids:
        missing = sorted(context_ids - image_ids)
        extra = sorted(image_ids - context_ids)
        raise ValueError(
            "Image manifest must match the context dataset exactly; "
            f"missing={missing[:5]}, extra={extra[:5]}"
        )

    if image_root is None:
        image_root_path = manifest_path.parent
    else:
        image_root_path = Path(image_root).expanduser().resolve()
    weights = (
        load_sample_weights(sample_weights_jsonl, context_ids)
        if sample_weights_jsonl is not None
        else {sample_id: 1.0 for sample_id in context_ids}
    )

    joined: list[dict[str, Any]] = []
    for context_line_number, sample_id in enumerate(context_order, start=1):
        context = contexts[sample_id]
        manifest = images[sample_id]
        query = _nonempty_text(context, "original_query", context_path, context_line_number)
        privileged = _nonempty_text(
            context, "privileged_prompt", context_path, context_line_number
        )
        source_sample_id = _nonempty_text(
            context, "source_sample_id", context_path, context_line_number
        )
        tag = _nonempty_text(context, "tag", context_path, context_line_number)
        row: dict[str, Any] = {
            "sample_id": sample_id,
            "source_sample_id": source_sample_id,
            "tag": tag,
            "student_prompt": query,
            "privileged_prompt": privileged,
            "sample_weight": weights[sample_id],
        }
        if method == "sft":
            role = _nonempty_text(manifest, "target_prompt_role", manifest_path, 1)
            if role != "privileged_prompt":
                raise ValueError(
                    f"sample_id={sample_id}: SFT target_prompt_role must be 'privileged_prompt', got {role!r}"
                )
            target = _nonempty_text(manifest, "target_image", manifest_path, 1)
            row["target_image"] = _resolve_image(target, manifest_path, image_root_path)
            _verify_image_hash(
                row["target_image"],
                manifest["target_sha256"],
                sample_id=sample_id,
                field="target_sha256",
            )
        else:
            chosen_role = _nonempty_text(manifest, "chosen_prompt_role", manifest_path, 1)
            rejected_role = _nonempty_text(manifest, "rejected_prompt_role", manifest_path, 1)
            if (chosen_role, rejected_role) != ("privileged_prompt", "original_query"):
                raise ValueError(
                    f"sample_id={sample_id}: flow-DPO roles must be "
                    "chosen=privileged_prompt and rejected=original_query"
                )
            chosen = _nonempty_text(manifest, "chosen_image", manifest_path, 1)
            rejected = _nonempty_text(manifest, "rejected_image", manifest_path, 1)
            row["chosen_image"] = _resolve_image(chosen, manifest_path, image_root_path)
            row["rejected_image"] = _resolve_image(rejected, manifest_path, image_root_path)
            if row["chosen_image"] == row["rejected_image"]:
                raise ValueError(f"sample_id={sample_id}: chosen and rejected images must differ")
            _verify_image_hash(
                row["chosen_image"],
                manifest["chosen_sha256"],
                sample_id=sample_id,
                field="chosen_sha256",
            )
            _verify_image_hash(
                row["rejected_image"],
                manifest["rejected_sha256"],
                sample_id=sample_id,
                field="rejected_sha256",
            )
        joined.append(row)
    return joined


def _load_rgb(path: Path) -> Image.Image:
    with Image.open(path) as source:
        return ImageOps.exif_transpose(source).convert("RGB")


def _resize_and_crop_pair(
    images: list[Image.Image], resolution: int, horizontal_flip: bool
) -> list[Image.Image]:
    if resolution <= 0:
        raise ValueError("resolution must be positive")
    width, height = images[0].size
    if width <= 0 or height <= 0:
        raise ValueError("image has invalid dimensions")
    images = [image if image.size == (width, height) else image.resize((width, height)) for image in images]
    scale = resolution / min(width, height)
    resized_width = max(resolution, round(width * scale))
    resized_height = max(resolution, round(height * scale))
    left = (resized_width - resolution) // 2
    top = (resized_height - resolution) // 2
    box = (left, top, left + resolution, top + resolution)
    output = [
        image.resize((resized_width, resized_height), Image.Resampling.BILINEAR).crop(box)
        for image in images
    ]
    if horizontal_flip:
        output = [ImageOps.mirror(image) for image in output]
    return output


def _to_normalized_tensor(image: Image.Image) -> torch.Tensor:
    array = np.asarray(image, dtype=np.float32).copy()
    tensor = torch.from_numpy(array).permute(2, 0, 1).contiguous()
    return tensor.div_(127.5).sub_(1.0)


class ContextImageDataset(Dataset):
    """SFT/DPO supervision joined to the unchanged D-OPCD (q, p) context rows.

    The model-facing prompt is always q. The p field is retained only so the
    sidecar can prove where chosen/target supervision came from.
    """

    def __init__(
        self,
        context_jsonl: str | Path,
        image_manifest_jsonl: str | Path,
        method: BaselineMethod,
        resolution: int,
        *,
        sample_weights_jsonl: str | Path | None = None,
        image_root: str | Path | None = None,
        random_horizontal_flip: bool = False,
    ):
        self.method = method
        self.resolution = resolution
        self.random_horizontal_flip = random_horizontal_flip
        self.rows = load_joined_rows(
            context_jsonl,
            image_manifest_jsonl,
            method,
            sample_weights_jsonl=sample_weights_jsonl,
            image_root=image_root,
        )

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int) -> dict[str, Any]:
        row = self.rows[index]
        flip = self.random_horizontal_flip and random.random() < 0.5
        if self.method == "sft":
            images = _resize_and_crop_pair([_load_rgb(row["target_image"])], self.resolution, flip)
            pixel_values = _to_normalized_tensor(images[0])
        else:
            images = _resize_and_crop_pair(
                [_load_rgb(row["chosen_image"]), _load_rgb(row["rejected_image"])],
                self.resolution,
                flip,
            )
            pixel_values = torch.stack([_to_normalized_tensor(image) for image in images])
        return {
            "sample_id": row["sample_id"],
            "prompt": row["student_prompt"],
            "pixel_values": pixel_values,
            "sample_weight": row["sample_weight"],
        }


def collate_context_images(examples: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "sample_ids": [example["sample_id"] for example in examples],
        "prompts": [example["prompt"] for example in examples],
        "pixel_values": torch.stack([example["pixel_values"] for example in examples])
        .to(memory_format=torch.contiguous_format)
        .float(),
        "sample_weights": torch.tensor(
            [example["sample_weight"] for example in examples], dtype=torch.float32
        ),
    }

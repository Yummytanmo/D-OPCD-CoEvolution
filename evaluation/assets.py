"""Model cache and GenEval detector weights for local feedback services."""

from __future__ import annotations

import os
import shutil
import tempfile
import urllib.request
from pathlib import Path

from evaluation.model_config import load_project_config


GENEVAL_WEIGHT_NAME = "mask2former_swin-s-p4-w7-224_lsj_8x2_50e_coco.pth"
GENEVAL_WEIGHT_URL = (
    "https://download.openmmlab.com/mmdetection/v2.0/mask2former/"
    "mask2former_swin-s-p4-w7-224_lsj_8x2_50e_coco/"
    "mask2former_swin-s-p4-w7-224_lsj_8x2_50e_coco_20220504_001756-743b7d99.pth"
)


def evaluation_root() -> Path:
    return Path(__file__).resolve().parent


def default_model_root() -> Path:
    configured = load_project_config().get("model_root") or os.environ.get("EVALUATION_MODEL_ROOT")
    if configured:
        path = Path(str(configured)).expanduser()
        return (path if path.is_absolute() else evaluation_root().parent / path).resolve()
    return evaluation_root() / ".assets" / "models"


def configure_cache_environment(model_root: Path | None = None) -> Path:
    root = (model_root or default_model_root()).expanduser().resolve()
    root.mkdir(parents=True, exist_ok=True)
    for name, directory in (("HF_HOME", "huggingface"), ("TORCH_HOME", "torch")):
        cache = root / directory
        cache.mkdir(parents=True, exist_ok=True)
        os.environ.setdefault(name, str(cache))
    return root


def ensure_geneval_weights(model_dir: str | Path) -> Path:
    destination = Path(model_dir).expanduser().resolve() / GENEVAL_WEIGHT_NAME
    if destination.is_file() and destination.stat().st_size > 0:
        return destination
    destination.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(dir=destination.parent, suffix=".download", delete=False) as handle:
        temporary = Path(handle.name)
    try:
        request = urllib.request.Request(GENEVAL_WEIGHT_URL, headers={"User-Agent": "D-OPCD-CoEvolution/1.0"})
        with urllib.request.urlopen(request, timeout=120) as response, temporary.open("wb") as output:
            shutil.copyfileobj(response, output, length=1024 * 1024)
        if temporary.stat().st_size == 0:
            raise RuntimeError("Downloaded empty GenEval detector checkpoint")
        os.replace(temporary, destination)
    finally:
        temporary.unlink(missing_ok=True)
    return destination

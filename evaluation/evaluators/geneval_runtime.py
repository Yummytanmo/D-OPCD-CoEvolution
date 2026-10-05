"""Persistent in-process runtime around the official GenEval evaluator."""

from __future__ import annotations

import importlib.util
import os
import sys
import threading
from pathlib import Path
from types import SimpleNamespace
from typing import Any

from evaluation.assets import ensure_geneval_weights, evaluation_root


ORIGINAL_MODULE = Path(os.environ.get(
    "GENEVAL_EVALUATOR_SCRIPT",
    Path(__file__).resolve().parents[2]
    / "evaluation/.assets/geneval/evaluation/evaluate_images.py",
)).expanduser().resolve()


class GenEvalRuntime:
    """Load Mask2Former/OpenCLIP once and reuse the official image evaluator."""

    def __init__(
        self,
        *,
        model_path: str | Path,
        open_clip_cache_dir: str | Path,
        model_config: str | Path | None = None,
        device: str = "cuda",
    ) -> None:
        self.model_path = Path(model_path).expanduser().resolve()
        self.open_clip_cache_dir = Path(open_clip_cache_dir).expanduser().resolve()
        self.model_config = (
            Path(model_config).expanduser().resolve() if model_config else None
        )
        self.device = device
        self._module = None
        self._load_lock = threading.Lock()
        self._evaluation_lock = threading.Lock()

    def _resolve_model_config(self, module: Any) -> Path:
        if self.model_config is not None:
            return self.model_config
        workspace_config = (
            Path(__file__).resolve().parents[2]
            / "mmdetection"
            / "configs"
            / "mask2former"
            / "mask2former_swin-s-p4-w7-224_lsj_8x2_50e_coco.py"
        )
        if workspace_config.is_file():
            return workspace_config
        return (
            Path(module.mmdet.__file__).resolve().parent
            / ".."
            / "configs"
            / "mask2former"
            / "mask2former_swin-s-p4-w7-224_lsj_8x2_50e_coco.py"
        ).resolve()

    def load(self) -> None:
        if self._module is not None:
            return
        with self._load_lock:
            if self._module is not None:
                return
            if not ORIGINAL_MODULE.is_file():
                raise FileNotFoundError(
                    f"Install the official GenEval evaluator or set GENEVAL_EVALUATOR_SCRIPT: {ORIGINAL_MODULE}"
                )
            self.model_path = ensure_geneval_weights(self.model_path).parent
            try:
                self.open_clip_cache_dir.mkdir(parents=True, exist_ok=True)
            except OSError:
                self.open_clip_cache_dir = (
                    evaluation_root() / ".assets" / "models" / "open_clip"
                )
                self.open_clip_cache_dir.mkdir(parents=True, exist_ok=True)
            local_mmdetection = Path(__file__).resolve().parents[2] / "mmdetection"
            if local_mmdetection.is_dir() and str(local_mmdetection) not in sys.path:
                sys.path.insert(0, str(local_mmdetection))
            spec = importlib.util.spec_from_file_location(
                f"evaluation_copied_geneval_{id(self)}",
                ORIGINAL_MODULE,
            )
            if spec is None or spec.loader is None:
                raise RuntimeError(f"Cannot import GenEval evaluator: {ORIGINAL_MODULE}")
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
            module.DEVICE = self.device

            options: dict[str, str] = {}
            args = SimpleNamespace(
                model_config=str(self._resolve_model_config(module)),
                model_path=str(self.model_path),
                options=options,
            )
            original_create = module.open_clip.create_model_and_transforms

            def create_with_local_cache(*values, **kwargs):
                kwargs.setdefault("cache_dir", str(self.open_clip_cache_dir))
                local_clip = Path(
                    os.environ.get(
                        "GENEVAL_CLIP_WEIGHTS",
                        "",
                    )
                )
                if kwargs.get("pretrained") == "openai" and local_clip.is_file():
                    kwargs["pretrained"] = str(local_clip)
                return original_create(*values, **kwargs)

            module.open_clip.create_model_and_transforms = create_with_local_cache
            try:
                detector, clip_components, classnames = module.load_models(args)
            finally:
                module.open_clip.create_model_and_transforms = original_create

            module.args = args
            module.object_detector = detector
            module.clip_model, module.transform, module.tokenizer = clip_components
            module.classnames = classnames
            module.THRESHOLD = 0.3
            module.COUNTING_THRESHOLD = 0.9
            module.MAX_OBJECTS = 16
            module.NMS_THRESHOLD = 1.0
            module.POSITION_THRESHOLD = 0.1
            self._module = module

    def evaluate_image(self, image_path: str | Path, metadata: dict[str, Any]) -> dict[str, Any]:
        self.load()
        assert self._module is not None
        with self._evaluation_lock:
            return dict(self._module.evaluate_image(str(image_path), dict(metadata)))

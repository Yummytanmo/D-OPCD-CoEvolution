from __future__ import annotations

import hashlib
import json
import os
import re
import tempfile
import unicodedata
from pathlib import Path
from typing import Any, Iterable


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONTEXT_LABEL = "Privileged generation context:"
TEACHER_CONTEXT_MODES = ("p_only", "q_plus_p", "q_only", "vlm_q_image", "vlm_q_p_image")
TRAIN_FIELDS = {
    "sample_id",
    "source_sample_id",
    "tag",
    "original_query",
    "privileged_prompt",
}


def normalize_text(value: str) -> str:
    value = unicodedata.normalize("NFKC", str(value))
    return re.sub(r"\s+", " ", value).strip().casefold()


def compose_teacher_context(
    original_query: str,
    privileged_prompt: str,
    label: str = DEFAULT_CONTEXT_LABEL,
) -> str:
    query = str(original_query).strip()
    context = str(privileged_prompt).strip()
    context_label = str(label).strip()
    if not query:
        raise ValueError("original_query must be non-empty")
    if not context:
        raise ValueError("privileged_prompt must be non-empty")
    if not context_label:
        raise ValueError("context label must be non-empty")
    return f"{query}\n\n{context_label}\n{context}"


def teacher_context_description(mode: str) -> str:
    if mode not in TEACHER_CONTEXT_MODES:
        raise ValueError(f"Unknown teacher context mode: {mode}")
    return {
        "p_only": "p", "q_plus_p": "[q;p]", "q_only": "q",
        "vlm_q_image": "VLM(q+image)",
        "vlm_q_p_image": "VLM([q;p]+image)",
    }[mode]


def resolve_teacher_context_mode(
    requested: str | None, args_path: str | Path, resume_requested: bool = False
) -> str:
    """Use p_only for bare new runs; preserve the routing of existing runs."""
    if requested is not None and requested not in TEACHER_CONTEXT_MODES:
        raise ValueError(f"Unknown teacher context mode: {requested}")
    path = Path(args_path)
    previous = None
    if path.is_file():
        saved = read_json(path)
        previous = saved.get("teacher_context_mode")
        if previous is None:
            previous = {"[q;p]": "q_plus_p", "p": "p_only", "q": "q_only"}.get(saved.get("teacher_context"))
        if previous not in TEACHER_CONTEXT_MODES:
            raise ValueError(f"Cannot determine teacher context mode from existing {path}")
    mode = requested or (previous if resume_requested else None) or "p_only"
    if previous is not None and mode != previous:
        raise ValueError(
            f"Teacher context mode mismatch for {path}: existing={previous}, requested={mode}. "
            "Use a new run directory or resume with the original mode."
        )
    return mode


def read_json(path: str | Path) -> Any:
    with Path(path).open("r", encoding="utf-8") as handle:
        return json.load(handle)


def read_jsonl(path: str | Path) -> list[dict[str, Any]]:
    source = Path(path)
    rows: list[dict[str, Any]] = []
    with source.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                value = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON at {source}:{line_number}: {exc}") from exc
            if not isinstance(value, dict):
                raise ValueError(f"Expected an object at {source}:{line_number}")
            rows.append(value)
    return rows


def _atomic_write(path: Path, payload: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        "w",
        encoding="utf-8",
        newline="\n",
        dir=path.parent,
        prefix=f".{path.name}.",
        suffix=".tmp",
        delete=False,
    ) as handle:
        handle.write(payload)
        temporary = Path(handle.name)
    os.replace(temporary, path)


def write_json(path: str | Path, value: Any) -> None:
    payload = json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False)
    _atomic_write(Path(path), f"{payload}\n")


def write_jsonl(path: str | Path, rows: Iterable[dict[str, Any]]) -> None:
    payload = "".join(
        json.dumps(row, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False)
        + "\n"
        for row in rows
    )
    _atomic_write(Path(path), payload)


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def stable_integer_id(dataset: str, split: str, source_id: str) -> int:
    payload = f"{dataset}:{split}:{source_id}".encode("utf-8")
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], "big") & ((1 << 63) - 1)


def ensure_within(path: str | Path, root: str | Path = PROJECT_ROOT) -> Path:
    resolved = Path(path).expanduser().resolve()
    boundary = Path(root).expanduser().resolve()
    if resolved != boundary and boundary not in resolved.parents:
        raise ValueError(f"Path must stay inside {boundary}: {resolved}")
    return resolved


def resolve_project_path(path: str | Path, root: str | Path = PROJECT_ROOT) -> Path:
    value = Path(path).expanduser()
    if not value.is_absolute():
        value = Path(root) / value
    return value.resolve()

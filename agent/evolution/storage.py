"""Atomic file primitives shared by evolution domains."""

from __future__ import annotations

import hashlib
import json
import os
import re
import threading
import uuid
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterator


STATE_FORMAT_VERSION = 1


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _record_name(key: str) -> str:
    prefix = re.sub(r"[^A-Za-z0-9._-]+", "-", key).strip("-.")[:80] or "record"
    digest = hashlib.sha1(key.encode("utf-8")).hexdigest()[:12]
    return f"{prefix}-{digest}.json"


class StateStore:
    """File-backed state with atomic writes and an in-process writer lock."""

    def __init__(
        self,
        path: str | Path,
        *,
        run_id: str | None = None,
        generator_model: str = "unknown",
        round_id: int = 0,
    ) -> None:
        self.path = Path(path).expanduser().resolve()
        self.path.mkdir(parents=True, exist_ok=True)
        self.run_id = str(run_id or self.path.parent.name)
        self.generator_model = str(generator_model)
        self.round_id = int(round_id)
        self._lock = threading.RLock()

        metadata = self.read_data("metadata", {})
        stored_run = str(metadata.get("run_id") or "")
        if stored_run and stored_run != self.run_id:
            raise ValueError(
                "Evolution state belongs to a different run_id. "
                "Use that run_id or a separate state directory."
            )
        history = [str(item) for item in metadata.get("generator_model_history") or []]
        if self.generator_model not in history:
            history.append(self.generator_model)
        metadata.update(
            {
                "format_version": STATE_FORMAT_VERSION,
                "run_id": self.run_id,
                "generator_model": self.generator_model,
                "generator_model_history": history,
                "round_id": self.round_id,
                "updated_at": utc_now(),
            }
        )
        metadata.setdefault("created_at", utc_now())
        self.write_data("metadata", metadata)

    @contextmanager
    def locked(self) -> Iterator[None]:
        with self._lock:
            yield

    @staticmethod
    def load_json(path: Path, default: Any = None) -> Any:
        if not path.is_file():
            return default
        return json.loads(path.read_text(encoding="utf-8"))

    @staticmethod
    def save_json(path: Path, value: Any) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_name(f".{path.name}.{os.getpid()}.{uuid.uuid4().hex}.tmp")
        temporary.write_text(
            json.dumps(value, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        temporary.replace(path)

    @staticmethod
    def save_text(path: Path, value: str) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_name(f".{path.name}.{os.getpid()}.{uuid.uuid4().hex}.tmp")
        temporary.write_text(value, encoding="utf-8")
        temporary.replace(path)

    def read_data(self, name: str, default: Any = None) -> Any:
        with self._lock:
            return self.load_json(self.path / f"{name}.json", default)

    def write_data(self, name: str, value: Any) -> None:
        with self._lock:
            self.save_json(self.path / f"{name}.json", value)

    def update_data(self, name: str, default: Any, update) -> Any:
        with self._lock:
            value = self.load_json(self.path / f"{name}.json", default)
            result = update(value)
            self.save_json(self.path / f"{name}.json", result)
            return result

    def record_path(self, collection: str, key: str) -> Path:
        return self.path / collection / _record_name(key)

    def read_record(self, collection: str, key: str) -> dict[str, Any] | None:
        with self._lock:
            value = self.load_json(self.record_path(collection, key))
        return dict(value) if isinstance(value, dict) else None

    def create_record(
        self,
        collection: str,
        key: str,
        value: dict[str, Any],
    ) -> bool:
        with self._lock:
            path = self.record_path(collection, key)
            if path.exists():
                return False
            self.save_json(path, value)
            return True

    def write_record(
        self,
        collection: str,
        key: str,
        value: dict[str, Any],
    ) -> None:
        with self._lock:
            self.save_json(self.record_path(collection, key), value)

    def update_record(
        self,
        collection: str,
        key: str,
        update,
    ) -> dict[str, Any] | None:
        with self._lock:
            path = self.record_path(collection, key)
            value = self.load_json(path)
            if not isinstance(value, dict):
                return None
            result = dict(update(dict(value)))
            self.save_json(path, result)
            return result

    def records(self, collection: str) -> list[dict[str, Any]]:
        directory = self.path / collection
        if not directory.is_dir():
            return []
        with self._lock:
            values = [self.load_json(path) for path in sorted(directory.glob("*.json"))]
        return [dict(value) for value in values if isinstance(value, dict)]

    def record_count(self, collection: str) -> int:
        directory = self.path / collection
        return len(list(directory.glob("*.json"))) if directory.is_dir() else 0

    def get_metadata(self, key: str, default: Any = None) -> Any:
        return self.read_data("metadata", {}).get(key, default)

    def set_metadata(self, key: str, value: Any) -> None:
        def apply(metadata: dict[str, Any]) -> dict[str, Any]:
            metadata[key] = value
            metadata["updated_at"] = utc_now()
            return metadata

        self.update_data("metadata", {}, apply)

    def episode_key(self, task_id: str) -> str:
        return f"{self.round_id}:{task_id}"

    def progress(self, task_id: str) -> dict[str, Any] | None:
        return self.read_record("progress", self.episode_key(task_id))

    def mark_progress(
        self,
        task_id: str,
        status: str,
        *,
        image_path: str | None = None,
        result_path: str | None = None,
        feedback_error: str | None = None,
    ) -> None:
        key = self.episode_key(task_id)
        with self._lock:
            value = self.read_record("progress", key) or {
                "task_id": task_id,
                "round_id": self.round_id,
                "run_id": self.run_id,
            }
            value.update(status=status, feedback_error=feedback_error, updated_at=utc_now())
            if image_path is not None:
                value["image_path"] = image_path
            if result_path is not None:
                value["result_path"] = result_path
            self.write_record("progress", key, value)

    def is_committed(self, task_id: str) -> bool:
        value = self.progress(task_id)
        return value is not None and value.get("status") == "committed"


__all__ = ["STATE_FORMAT_VERSION", "StateStore", "utc_now"]

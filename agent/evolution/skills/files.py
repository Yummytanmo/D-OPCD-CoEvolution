"""File layout and projections for versioned skills."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Iterable

from agent.evolution.skills.document import (
    SKILL_EXECUTION_STAGE,
    SKILL_INPUT_CONTRACT,
    SKILL_KIND,
    SKILL_OUTPUT_CONTRACT,
    SkillDocument,
)
from agent.evolution.storage import StateStore, utc_now


class SkillFiles:
    RETIRED_DIRECTORY_NAME = "retired"

    def __init__(self, state: StateStore) -> None:
        self.state = state
        self.root = state.path / "skills"

    @property
    def retired_root(self) -> Path:
        return self.root / self.RETIRED_DIRECTORY_NAME

    @property
    def active_library_path(self) -> Path:
        return self.root / "active_library.json"

    def active_library(self) -> dict[str, Any] | None:
        value = self.state.load_json(self.active_library_path)
        if not isinstance(value, dict) or not isinstance(value.get("skills"), dict):
            return None
        return dict(value)

    def active_versions(self) -> dict[str, int]:
        """Read the single atomic visibility pointer for the Skill library."""
        library = self.active_library()
        if library is not None:
            return {
                str(skill_id): int(version)
                for skill_id, version in library["skills"].items()
            }
        # Legacy fallback is used only while initializing the first pointer.
        active: dict[str, int] = {}
        if not self.root.is_dir():
            return active
        for directory in sorted(self.root.iterdir()):
            if (
                not directory.is_dir()
                or directory.name == self.RETIRED_DIRECTORY_NAME
            ):
                continue
            manifest = self.load_manifest(directory.name)
            if isinstance(manifest, dict) and manifest.get("active_version") is not None:
                active[directory.name] = int(manifest["active_version"])
        return active

    def ensure_active_library(self) -> None:
        if self.active_library() is not None:
            return
        active = self.active_versions()
        self.state.save_json(
            self.active_library_path,
            {
                "schema_version": 1,
                "revision": 0,
                "skills": active,
                "updated_at": utc_now(),
            },
        )

    def save_active_library(
        self,
        active_versions: dict[str, int],
        *,
        expected_versions: dict[str, int] | None = None,
    ) -> dict[str, Any]:
        """Atomically expose a completely prepared set of immutable versions."""
        current = self.active_versions()
        if expected_versions is not None and current != expected_versions:
            raise ValueError("active Skill versions changed during staged transaction")
        previous = self.active_library() or {}
        value = {
            "schema_version": 1,
            "revision": int(previous.get("revision", 0)) + 1,
            "skills": {
                str(skill_id): int(version)
                for skill_id, version in sorted(active_versions.items())
            },
            "updated_at": utc_now(),
        }
        self.state.save_json(self.active_library_path, value)
        return value

    def active_skill_dir(self, skill_id: str) -> Path:
        return self.root / skill_id

    def retired_skill_dir(self, skill_id: str) -> Path:
        return self.retired_root / skill_id

    def known_skill_ids(self) -> list[str]:
        """Return active and retired IDs without treating the archive as a Skill."""
        skill_ids: set[str] = set()
        if self.root.is_dir():
            skill_ids.update(
                directory.name
                for directory in self.root.iterdir()
                if directory.is_dir()
                and directory.name != self.RETIRED_DIRECTORY_NAME
            )
        if self.retired_root.is_dir():
            skill_ids.update(
                directory.name
                for directory in self.retired_root.iterdir()
                if directory.is_dir()
            )
        return sorted(skill_ids)

    def skill_dir(self, skill_id: str) -> Path:
        """Resolve an active Skill first, then its archived retirement history."""
        active = self.active_skill_dir(skill_id)
        if active.exists():
            return active
        retired = self.retired_skill_dir(skill_id)
        return retired if retired.exists() else active

    def version_dir(self, skill_id: str, version: int) -> Path:
        return self.skill_dir(skill_id) / f"v{version}"

    def load_manifest(self, skill_id: str) -> dict[str, Any] | None:
        value = self.state.load_json(self.skill_dir(skill_id) / "manifest.json")
        return dict(value) if isinstance(value, dict) else None

    def save_manifest(self, value: dict[str, Any]) -> None:
        path = self.skill_dir(str(value["skill_id"])) / "manifest.json"
        self.state.save_json(path, value)

    def archive_retired(self, skill_ids: Iterable[str]) -> list[str]:
        """Move newly retired Skill histories out of the active directory listing."""
        archived = []
        for raw_skill_id in sorted(set(str(item) for item in skill_ids)):
            source = self.active_skill_dir(raw_skill_id)
            destination = self.retired_skill_dir(raw_skill_id)
            if destination.exists():
                if source.exists():
                    raise RuntimeError(
                        f"both active and retired Skill directories exist: {raw_skill_id}"
                    )
                continue
            if not source.is_dir():
                raise FileNotFoundError(
                    f"retired Skill directory is missing: {raw_skill_id}"
                )
            self.retired_root.mkdir(parents=True, exist_ok=True)
            source.replace(destination)
            archived.append(raw_skill_id)
        return archived

    def migrate_retired_layout(self) -> list[str]:
        """Archive legacy root-level directories whose manifests are inactive."""
        if not self.root.is_dir():
            return []
        active_ids = set(self.active_versions())
        retired = []
        for directory in sorted(self.root.iterdir()):
            if (
                not directory.is_dir()
                or directory.name == self.RETIRED_DIRECTORY_NAME
                or directory.name in active_ids
            ):
                continue
            manifest = self.state.load_json(directory / "manifest.json")
            if not isinstance(manifest, dict):
                continue
            if manifest.get("active_version") is None:
                retired.append(directory.name)
        return self.archive_retired(retired)

    def _usage_counts(self) -> dict[tuple[str, int], dict[str, Any]]:
        """Derive mutable utility from append-only task records.

        Skill Markdown and version metadata stay stable; deleting or replaying one task
        cannot leave a stale counter embedded in the skill itself.
        """
        counts: dict[tuple[str, int], dict[str, Any]] = {}
        for item in self.state.records("skill_usage"):
            key = (str(item.get("skill_id") or ""), int(item.get("version", -1)))
            value = counts.setdefault(
                key,
                {
                    "uses": 0,
                    "first_attempt_count": 0,
                    "first_attempt_sum": 0.0,
                    "final_reward_count": 0,
                    "final_reward_sum": 0.0,
                },
            )
            value["uses"] += 1
            first_attempt_score = item.get("first_attempt_score")
            if isinstance(first_attempt_score, (int, float)) and not isinstance(
                first_attempt_score, bool
            ):
                value["first_attempt_count"] += 1
                value["first_attempt_sum"] += float(first_attempt_score)
            final_reward = item.get("final_reward")
            if isinstance(final_reward, (int, float)) and not isinstance(
                final_reward, bool
            ):
                value["final_reward_count"] += 1
                value["final_reward_sum"] += float(final_reward)
        return counts

    def load_version(
        self,
        skill_id: str,
        version: int,
        usage_counts: dict[tuple[str, int], dict[str, Any]] | None = None,
    ) -> dict[str, Any] | None:
        directory = self.version_dir(skill_id, version)
        meta = self.state.load_json(directory / "meta.json")
        markdown_path = directory / "SKILL.md"
        if not isinstance(meta, dict) or not markdown_path.is_file():
            return None
        document = SkillDocument.from_markdown(
            markdown_path.read_text(encoding="utf-8"), skill_id
        )
        counts = (usage_counts if usage_counts is not None else self._usage_counts()).get(
            (skill_id, version),
            {
                "uses": 0,
                "first_attempt_count": 0,
                "first_attempt_sum": 0.0,
                "final_reward_count": 0,
                "final_reward_sum": 0.0,
            },
        )
        value = {
            "skill_id": skill_id,
            "version": version,
            **dict(meta),
            **document.runtime_fields(),
            "uses": counts["uses"],
            "first_attempt_count": counts["first_attempt_count"],
            "final_reward_count": counts["final_reward_count"],
        }
        if counts["first_attempt_count"]:
            value["mean_first_attempt_score"] = round(
                counts["first_attempt_sum"] / counts["first_attempt_count"],
                6,
            )
        if counts["final_reward_count"]:
            value["mean_final_reward"] = round(
                counts["final_reward_sum"] / counts["final_reward_count"], 6
            )
        return value

    def save_version(
        self,
        *,
        skill_id: str,
        version: int,
        status: str,
        markdown: str,
        evidence_ids: list[str],
        parent_version: int | None,
        extra_metadata: dict[str, Any] | None = None,
    ) -> None:
        directory = self.version_dir(skill_id, version)
        now = utc_now()
        self.state.save_text(directory / "SKILL.md", markdown)
        meta = {
            "status": status,
            "kind": SKILL_KIND,
            "execution_stage": SKILL_EXECUTION_STAGE,
            "input_contract": SKILL_INPUT_CONTRACT,
            "output_contract": SKILL_OUTPUT_CONTRACT,
            "evidence_ids": list(dict.fromkeys(evidence_ids)),
            "parent_version": parent_version,
            "created_at": now,
            "updated_at": now,
            **dict(extra_metadata or {}),
        }
        self.state.save_json(directory / "meta.json", meta)

    def runtime_skills(self) -> list[dict[str, Any]]:
        """Load exactly the active version of each logical Skill."""
        usage_counts = self._usage_counts()
        skills = []
        for skill_id, version in sorted(self.active_versions().items()):
            value = self.load_version(
                skill_id,
                int(version),
                usage_counts,
            )
            if value is None:
                raise RuntimeError(
                    f"active Skill pointer references a missing version: {skill_id}@{version}"
                )
            # Version metadata is historical projection. Selection by the atomic pointer
            # is authoritative even if a crash interrupted later metadata housekeeping.
            value["status"] = "active"
            if (
                value.get("kind", SKILL_KIND) == SKILL_KIND
                and value.get("execution_stage", SKILL_EXECUTION_STAGE)
                == SKILL_EXECUTION_STAGE
            ):
                skills.append(value)
        return skills

__all__ = ["SkillFiles"]

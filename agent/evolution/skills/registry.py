"""Versioned high-level initial-prompt Skills with direct activation."""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any, Iterable

from agent.evolution.skills.document import (
    SKILL_EXECUTION_STAGE,
    SKILL_KIND,
    SkillDocument,
)
from agent.evolution.skills.files import SkillFiles
from agent.evolution.storage import StateStore, utc_now


_ID_RE = re.compile(r"[^a-z0-9_]+")
_TOKEN_RE = re.compile(r"[a-z0-9_]+", re.IGNORECASE)
PROPOSED_SKILL_TOKEN = "__proposed_skill__"


def _skill_id(value: str) -> str:
    normalized = _ID_RE.sub("_", value.strip().casefold()).strip("_")[:64]
    return "" if normalized == SkillFiles.RETIRED_DIRECTORY_NAME else normalized


def _tokens(value: str) -> set[str]:
    return {token.casefold() for token in _TOKEN_RE.findall(value) if len(token) > 1}


class SkillRegistry:
    def __init__(
        self,
        state: StateStore,
        *,
        max_skills: int | None = None,
    ) -> None:
        self.state = state
        self.files = SkillFiles(state)
        self.max_skills = int(max_skills) if max_skills is not None else None
        if self.max_skills is not None and self.max_skills <= 0:
            raise ValueError("max_skills must be positive")

    def _over_capacity(self, count: int) -> bool:
        return self.max_skills is not None and int(count) > self.max_skills

    def initialize(self, mode: str, seed_dir: str | Path = "agent/skills") -> None:
        mode = mode.strip().casefold()
        if mode not in {"empty", "seed"}:
            raise ValueError("skill initialization must be 'empty' or 'seed'")
        if self.state.get_metadata("skill_registry_initialized", False):
            self._activate_legacy_candidates()
            self.files.ensure_active_library()
            with self.state.locked():
                self.files.migrate_retired_layout()
            self._validate_active_capacity()
            return
        with self.state.locked():
            if mode == "seed":
                self._import_seed_skills(Path(seed_dir))
            self.files.ensure_active_library()
            self.files.migrate_retired_layout()
            self.state.set_metadata("skill_initialization", mode)
            self.state.set_metadata("skill_registry_initialized", True)
        self._validate_active_capacity()

    def _validate_active_capacity(self) -> None:
        active_count = len(self.files.active_versions())
        if self._over_capacity(active_count):
            raise ValueError(
                f"active Skill library has {active_count} skills, above "
                f"max_skills={self.max_skills}; explicitly consolidate or retire "
                "Skills before resuming with this limit"
            )

    def _activate_legacy_candidates(self) -> None:
        """Migrate candidate versions written before direct activation was adopted."""
        if not self.files.root.is_dir():
            return
        with self.state.locked():
            self.files.ensure_active_library()
            baseline_active = self.files.active_versions()
            next_active = dict(baseline_active)
            for directory in sorted(self.files.root.iterdir()):
                if (
                    not directory.is_dir()
                    or directory.name == self.files.RETIRED_DIRECTORY_NAME
                ):
                    continue
                manifest = self.files.load_manifest(directory.name)
                if not isinstance(manifest, dict):
                    continue
                latest = manifest.get("latest_version")
                if latest is None:
                    continue
                latest = int(latest)
                meta_path = self.files.version_dir(directory.name, latest) / "meta.json"
                meta = self.state.load_json(meta_path)
                if not isinstance(meta, dict) or meta.get("status") != "candidate":
                    continue
                active_version = manifest.get("active_version")
                if active_version is not None and int(active_version) != latest:
                    old_path = (
                        self.files.version_dir(directory.name, int(active_version))
                        / "meta.json"
                    )
                    old = self.state.load_json(old_path)
                    if isinstance(old, dict):
                        old.update(
                            status="deprecated",
                            superseded_by_version=latest,
                            updated_at=utc_now(),
                        )
                        self.state.save_json(old_path, old)
                meta.update(status="active", updated_at=utc_now())
                manifest.update(active_version=latest, updated_at=utc_now())
                self.state.save_json(meta_path, meta)
                self.files.save_manifest(manifest)
                next_active[directory.name] = latest
            if next_active != baseline_active:
                self.files.save_active_library(
                    next_active,
                    expected_versions=baseline_active,
                )

    def _import_seed_skills(self, seed_dir: Path) -> None:
        if not seed_dir.exists():
            return
        directories = [
            directory
            for directory in sorted(seed_dir.iterdir())
            if (directory / "SKILL.md").is_file()
        ]
        if self._over_capacity(len(directories)):
            raise ValueError(
                f"seed library has {len(directories)} skills, above "
                f"max_skills={self.max_skills}"
            )
        for directory in directories:
            skill_id = _skill_id(directory.name)
            if not skill_id or self.files.load_manifest(skill_id) is not None:
                continue
            document = SkillDocument.from_markdown(
                (directory / "SKILL.md").read_text(encoding="utf-8"), skill_id
            )
            now = utc_now()
            self.files.save_version(
                skill_id=skill_id,
                version=0,
                status="active",
                markdown=document.to_markdown(),
                evidence_ids=[],
                parent_version=None,
            )
            self.files.save_manifest(
                {
                    "skill_id": skill_id,
                    "active_version": 0,
                    "latest_version": 0,
                    "created_at": now,
                    "updated_at": now,
                }
            )

    def runtime_skills(self) -> list[dict[str, Any]]:
        return [
            item
            for item in self.files.runtime_skills()
            if item.get("kind", SKILL_KIND) == SKILL_KIND
            and item.get("execution_stage", SKILL_EXECUTION_STAGE)
            == SKILL_EXECUTION_STAGE
        ]

    def retrieve(self, prompt: str, *, top_k: int = 3) -> list[dict[str, Any]]:
        prompt_tokens = _tokens(prompt)
        ranked: list[tuple[float, dict[str, Any]]] = []
        for skill in self.runtime_skills():
            searchable = f"{skill.get('name', '')} {skill.get('description', '')}"
            skill_tokens = _tokens(searchable)
            lexical = len(prompt_tokens & skill_tokens) / max(1, len(prompt_tokens))
            quality = (
                float(skill["mean_first_attempt_score"])
                if int(skill.get("first_attempt_count", 0))
                else 0.5
            )
            ranked.append((0.85 * lexical + 0.15 * quality, skill))
        ranked.sort(key=lambda pair: pair[0], reverse=True)
        return [item for _, item in ranked[: int(top_k)]]

    def skill_count(self) -> int:
        return len(self.runtime_skills())

    def can_create(self) -> bool:
        return self.max_skills is None or self.skill_count() < self.max_skills

    @staticmethod
    def manifest(skills: Iterable[dict[str, Any]]) -> str:
        return "\n".join(
            "- SKILL_ID: {skill_id}\n  NAME: {name}\n"
            "  DESCRIPTION: {description}".format(
                skill_id=skill["skill_id"],
                name=skill.get("skill_name") or skill["skill_id"],
                description=skill.get("description") or "",
            )
            for skill in skills
        )

    def summaries(self) -> list[dict[str, Any]]:
        summaries = []
        for skill in self.runtime_skills():
            value = {
                "skill_id": skill["skill_id"],
                "version": skill["version"],
                "status": skill["status"],
                "kind": skill["kind"],
                "execution_stage": skill["execution_stage"],
                "title": skill.get("name") or "",
                "description": skill.get("description") or "",
                "instructions": skill.get("instructions") or "",
                "source_insight_ids": list(skill.get("evidence_ids") or []),
                "uses": int(skill.get("uses", 0)),
                "first_attempt_count": int(skill.get("first_attempt_count", 0)),
            }
            if int(skill.get("first_attempt_count", 0)):
                value["mean_first_attempt_score"] = float(
                    skill["mean_first_attempt_score"]
                )
            summaries.append(value)
        return summaries

    def get(self, skill_id: str) -> dict[str, Any] | None:
        normalized = _skill_id(skill_id)
        return next(
            (item for item in self.runtime_skills() if item["skill_id"] == normalized),
            None,
        )

    def apply_tool_transaction(
        self,
        *,
        plan: dict[str, Any],
        batch_id: int,
    ) -> dict[str, Any] | None:
        """Commit a staged multi-Skill plan by switching one active-library pointer."""
        if str(plan.get("operation") or "").upper() != "SKILL_TRANSACTION":
            return None
        expected_versions = {
            _skill_id(str(skill_id)): int(version)
            for skill_id, version in dict(
                plan.get("expected_active_versions") or {}
            ).items()
        }
        raw_results = plan.get("result_skills") or []
        raw_retired = plan.get("retired_skills") or []
        if not isinstance(raw_results, list) or not isinstance(raw_retired, list):
            return None

        with self.state.locked():
            baseline_versions = self.files.active_versions()
            if baseline_versions != expected_versions:
                return None
            active = {item["skill_id"]: item for item in self.runtime_skills()}
            active_ids = set(active)

            retired_by_id: dict[str, list[str]] = {}
            for raw in raw_retired:
                if not isinstance(raw, dict):
                    return None
                skill_id = _skill_id(str(raw.get("skill_id") or ""))
                raw_replacements = raw.get("replacement_skill_ids")
                is_split = raw_replacements is not None
                if raw_replacements is None:
                    raw_replacements = [raw.get("replacement_skill_id")]
                if not isinstance(raw_replacements, list):
                    return None
                replacements = [
                    _skill_id(str(item or "")) for item in raw_replacements
                ]
                if (
                    not skill_id
                    or not replacements
                    or (is_split and len(replacements) != 2)
                    or len(replacements) != len(set(replacements))
                    or any(not item or item == skill_id for item in replacements)
                    or skill_id not in active_ids
                    or skill_id in retired_by_id
                ):
                    return None
                retired_by_id[skill_id] = replacements

            prepared = []
            result_ids: set[str] = set()
            for raw in raw_results:
                if not isinstance(raw, dict):
                    return None
                skill_id = _skill_id(str(raw.get("skill_id") or ""))
                if not skill_id or skill_id in result_ids or skill_id in retired_by_id:
                    return None
                result_ids.add(skill_id)
                expected_version = raw.get("expected_version")
                is_new = expected_version is None
                if is_new:
                    if skill_id in active_ids or self.files.load_manifest(skill_id) is not None:
                        return None
                elif (
                    skill_id not in active_ids
                    or int(expected_version) != baseline_versions[skill_id]
                ):
                    return None

                document = SkillDocument.from_markdown(
                    str(raw.get("markdown") or ""), skill_id
                )
                markdown = document.to_markdown()
                if not document.description or not document.instructions:
                    return None
                document.warn_if_not_initial_only()
                inherited = list(
                    dict.fromkeys(
                        _skill_id(str(item))
                        for item in raw.get("inherited_skill_ids", [])
                        if _skill_id(str(item))
                    )
                )
                if any(item not in active_ids for item in inherited):
                    return None
                if not is_new and skill_id not in inherited:
                    return None
                if any(
                    source != skill_id
                    and skill_id not in retired_by_id.get(source, [])
                    for source in inherited
                ):
                    return None
                new_evidence = list(
                    dict.fromkeys(
                        str(item)
                        for item in raw.get("new_source_insight_ids", [])
                        if str(item)
                    )
                )
                if is_new and not new_evidence:
                    return None
                evidence = list(new_evidence)
                for source in inherited:
                    evidence.extend(active[source].get("evidence_ids") or [])
                evidence = list(dict.fromkeys(str(item) for item in evidence))

                manifest = self.files.load_manifest(skill_id)
                if is_new:
                    version = 1
                    parent_version = None
                    now = utc_now()
                    manifest = {
                        "skill_id": skill_id,
                        "active_version": None,
                        "latest_version": version,
                        "created_at": now,
                        "updated_at": now,
                    }
                else:
                    if not isinstance(manifest, dict):
                        return None
                    parent_version = baseline_versions[skill_id]
                    version = int(manifest.get("latest_version", parent_version)) + 1
                    manifest = dict(manifest)
                prepared.append(
                    {
                        "skill_id": skill_id,
                        "operation": str(raw.get("operation") or "MODIFY").upper(),
                        "version": version,
                        "parent_version": parent_version,
                        "manifest": manifest,
                        "markdown": markdown,
                        "evidence_ids": evidence,
                        "new_source_insight_ids": new_evidence,
                        "inherited_skill_ids": inherited,
                    }
                )

            final_ids = (active_ids - set(retired_by_id)) | result_ids
            if (
                not final_ids
                or self._over_capacity(len(final_ids))
                or any(
                    value not in final_ids
                    for replacements in retired_by_id.values()
                    for value in replacements
                )
                or any(
                    value not in result_ids
                    for replacements in retired_by_id.values()
                    for value in replacements
                )
            ):
                return None
            if not prepared and not retired_by_id:
                return {
                    "operation": "NOOP",
                    "status": "active",
                    "active_skill_count": len(active_ids),
                }

            now = utc_now()
            prepared_by_id = {item["skill_id"]: item for item in prepared}

            # New immutable versions and projection metadata are written first. They are
            # not visible to inference until active_library.json is replaced below.
            for item in prepared:
                self.files.save_version(
                    skill_id=item["skill_id"],
                    version=item["version"],
                    status="active",
                    markdown=item["markdown"],
                    evidence_ids=item["evidence_ids"],
                    parent_version=item["parent_version"],
                    extra_metadata={
                        "evolution_operation": item["operation"],
                        "proposed_batch": int(batch_id),
                        "inherited_skill_ids": item["inherited_skill_ids"],
                        "new_source_insight_ids": item[
                            "new_source_insight_ids"
                        ],
                    },
                )

            for skill_id in sorted(active_ids):
                if skill_id not in prepared_by_id and skill_id not in retired_by_id:
                    continue
                version = baseline_versions[skill_id]
                meta_path = self.files.version_dir(skill_id, version) / "meta.json"
                meta = self.state.load_json(meta_path, {})
                if skill_id in retired_by_id:
                    replacements = retired_by_id[skill_id]
                    meta.update(status="deprecated", updated_at=now)
                    if len(replacements) == 1:
                        meta["merged_into"] = replacements[0]
                    else:
                        meta["split_into"] = replacements
                else:
                    meta.update(
                        status="deprecated",
                        superseded_by_version=prepared_by_id[skill_id]["version"],
                        updated_at=now,
                    )
                self.state.save_json(meta_path, meta)

            for skill_id, replacements in retired_by_id.items():
                manifest = self.files.load_manifest(skill_id)
                if not isinstance(manifest, dict):
                    return None
                manifest.update(active_version=None, updated_at=now)
                if len(replacements) == 1:
                    manifest["merged_into"] = replacements[0]
                else:
                    manifest["split_into"] = replacements
                self.files.save_manifest(manifest)

            for item in prepared:
                manifest = item["manifest"]
                manifest.update(
                    active_version=item["version"],
                    latest_version=item["version"],
                    updated_at=now,
                )
                self.files.save_manifest(manifest)

            next_versions = {
                skill_id: version
                for skill_id, version in baseline_versions.items()
                if skill_id not in retired_by_id
            }
            next_versions.update(
                {item["skill_id"]: item["version"] for item in prepared}
            )
            self.files.save_active_library(
                next_versions,
                expected_versions=baseline_versions,
            )
            self.files.archive_retired(retired_by_id)

        changes = [
            {
                "operation": item["operation"],
                "skill_id": item["skill_id"],
                "version": item["version"],
                "source_insight_ids": item["new_source_insight_ids"],
                "inherited_skill_ids": item["inherited_skill_ids"],
            }
            for item in prepared
        ]
        return {
            "operation": "SKILL_TRANSACTION",
            "changes": changes,
            "retired_skills": [
                (
                    {
                        "skill_id": skill_id,
                        "replacement_skill_id": replacements[0],
                    }
                    if len(replacements) == 1
                    else {
                        "skill_id": skill_id,
                        "replacement_skill_ids": replacements,
                    }
                )
                for skill_id, replacements in sorted(retired_by_id.items())
            ],
            "active_skill_count": len(final_ids),
            "status": "active",
            "execution_stage": SKILL_EXECUTION_STAGE,
        }

    def apply_evolution(
        self,
        *,
        proposal: dict[str, Any],
        batch_id: int,
        evidence_ids: list[str],
    ) -> dict[str, Any] | None:
        """Apply one validated library change and activate the resulting version."""
        operation = str(proposal.get("operation") or "").upper()
        operation = {"CREATE_SKILL": "CREATE", "MODIFY_SKILL": "MODIFY"}.get(
            operation, operation
        )
        if operation not in {"CREATE", "MODIFY", "MERGE", "REPLACE"}:
            return None
        requested_id = _skill_id(
            str(proposal.get("skill_id") or proposal.get("target_skill_id") or "")
        )
        raw_markdown = str(proposal.get("markdown") or "").strip()
        evidence_ids = list(dict.fromkeys(str(item) for item in evidence_ids))
        if (
            not requested_id
            or not raw_markdown
            or (operation in {"CREATE", "REPLACE"} and not evidence_ids)
        ):
            return None
        document = SkillDocument.from_markdown(raw_markdown, requested_id)
        markdown = document.to_markdown()
        if not document.description or not document.instructions:
            return None
        document.warn_if_not_initial_only()

        with self.state.locked():
            now = utc_now()
            baseline_active_versions = self.files.active_versions()
            existing = self.files.load_manifest(requested_id)
            merged_skill_ids = list(
                dict.fromkeys(
                    _skill_id(str(item))
                    for item in proposal.get("merged_skill_ids", [])
                    if _skill_id(str(item))
                )
            )
            replaced_skill_id = _skill_id(
                str(proposal.get("replaced_skill_id") or "")
            )

            def active_value(skill_id: str) -> tuple[dict[str, Any], dict[str, Any]] | None:
                manifest = self.files.load_manifest(skill_id)
                active_version = baseline_active_versions.get(skill_id)
                if not isinstance(manifest, dict) or active_version is None:
                    return None
                value = self.files.load_version(
                    skill_id,
                    int(active_version),
                )
                if value is None:
                    return None
                manifest = dict(manifest)
                manifest["active_version"] = int(active_version)
                return manifest, value

            def deprecate_active(
                skill_id: str,
                manifest: dict[str, Any],
                **metadata: Any,
            ) -> None:
                version = int(manifest["active_version"])
                meta_path = self.files.version_dir(skill_id, version) / "meta.json"
                meta = self.state.load_json(meta_path, {})
                meta.update(status="deprecated", updated_at=now, **metadata)
                self.state.save_json(meta_path, meta)

            source_evidence = list(evidence_ids)
            if operation == "CREATE":
                if existing is not None or not self.can_create():
                    return None
                version = 1
                parent_version = None
                manifest = {
                    "skill_id": requested_id,
                    "active_version": None,
                    "latest_version": version,
                    "created_at": now,
                    "updated_at": now,
                }
            elif operation in {"MODIFY", "MERGE"}:
                active = active_value(requested_id)
                if active is None:
                    return None
                manifest, current = active
                if operation == "MERGE":
                    if not merged_skill_ids or requested_id in merged_skill_ids:
                        return None
                    merged_values = []
                    for skill_id in merged_skill_ids:
                        merged = active_value(skill_id)
                        if merged is None:
                            return None
                        merged_values.append((skill_id, *merged))
                    for _, _, value in merged_values:
                        source_evidence.extend(value.get("evidence_ids") or [])
                else:
                    merged_values = []
                source_evidence.extend(current.get("evidence_ids") or [])
                source_evidence = list(dict.fromkeys(source_evidence))
                parent_version = int(manifest["active_version"])
                version = int(existing.get("latest_version", parent_version)) + 1
                deprecate_active(
                    requested_id,
                    manifest,
                    superseded_by_version=version,
                )
                if operation == "MERGE":
                    for skill_id, merged_manifest, _ in merged_values:
                        deprecate_active(
                            skill_id,
                            merged_manifest,
                            merged_into=requested_id,
                        )
                        merged_manifest.update(
                            active_version=None,
                            merged_into=requested_id,
                            updated_at=now,
                        )
                        self.files.save_manifest(merged_manifest)
                manifest = dict(manifest)
                manifest.update(latest_version=version, updated_at=now)
            else:
                if self.can_create() or existing is not None:
                    return None
                replaced = active_value(replaced_skill_id)
                if not replaced or replaced_skill_id == requested_id:
                    return None
                replaced_manifest, _ = replaced
                deprecate_active(
                    replaced_skill_id,
                    replaced_manifest,
                    replaced_by=requested_id,
                )
                replaced_manifest.update(
                    active_version=None,
                    replaced_by=requested_id,
                    updated_at=now,
                )
                self.files.save_manifest(replaced_manifest)
                version = 1
                parent_version = None
                manifest = {
                    "skill_id": requested_id,
                    "active_version": None,
                    "latest_version": version,
                    "replaces": replaced_skill_id,
                    "created_at": now,
                    "updated_at": now,
                }

            self.files.save_version(
                skill_id=requested_id,
                version=version,
                status="active",
                markdown=markdown,
                evidence_ids=source_evidence,
                parent_version=parent_version,
                extra_metadata={
                    "evolution_operation": operation,
                    "proposed_batch": int(batch_id),
                    **(
                        {"merged_from": merged_skill_ids}
                        if operation == "MERGE"
                        else {}
                    ),
                    **(
                        {"replaces": replaced_skill_id}
                        if operation == "REPLACE"
                        else {}
                    ),
                },
            )
            manifest.update(active_version=version, latest_version=version, updated_at=now)
            self.files.save_manifest(manifest)
            next_active_versions = dict(baseline_active_versions)
            next_active_versions[requested_id] = version
            if operation == "MERGE":
                for skill_id in merged_skill_ids:
                    next_active_versions.pop(skill_id, None)
            if operation == "REPLACE":
                next_active_versions.pop(replaced_skill_id, None)
            self.files.save_active_library(
                next_active_versions,
                expected_versions=baseline_active_versions,
            )
            if operation == "MERGE":
                self.files.archive_retired(merged_skill_ids)
            elif operation == "REPLACE":
                self.files.archive_retired([replaced_skill_id])
        result = {
            "operation": operation,
            "skill_id": requested_id,
            "version": version,
            "status": "active",
            "execution_stage": SKILL_EXECUTION_STAGE,
            "source_insight_ids": list(dict.fromkeys(evidence_ids)),
        }
        if operation == "MERGE":
            result["merged_skill_ids"] = merged_skill_ids
        if operation == "REPLACE":
            result["replaced_skill_id"] = replaced_skill_id
        return result

    def apply_reorganization(
        self,
        *,
        plan: dict[str, Any],
        proposed_skill: dict[str, Any],
        batch_id: int,
    ) -> dict[str, Any] | None:
        """Activate a validated full-library reorganization after a full-capacity CREATE."""
        if self.max_skills is None:
            return None
        if str(plan.get("operation") or "").upper() != "REORGANIZE":
            return None
        if str(proposed_skill.get("operation") or "").upper() != "CREATE":
            return None
        proposed_id = _skill_id(str(proposed_skill.get("skill_id") or ""))
        proposed_evidence = list(
            dict.fromkeys(
                str(item) for item in proposed_skill.get("source_insight_ids", [])
            )
        )
        if not proposed_id or not proposed_evidence:
            return None

        kept_ids = [_skill_id(str(item)) for item in plan.get("kept_skill_ids", [])]
        retired = list(plan.get("retired_skills", []))
        results = list(plan.get("result_skills", []))
        expected_versions = {
            _skill_id(str(skill_id)): int(version)
            for skill_id, version in dict(
                plan.get("expected_active_versions", {})
            ).items()
        }
        if not results or not expected_versions:
            return None

        with self.state.locked():
            baseline_active_versions = self.files.active_versions()
            active = {item["skill_id"]: item for item in self.runtime_skills()}
            active_versions = {
                skill_id: int(item["version"])
                for skill_id, item in active.items()
            }
            if (
                len(active) < self.max_skills
                or active_versions != expected_versions
                or self.files.load_manifest(proposed_id) is not None
            ):
                return None

            active_ids = set(active)
            assigned = [*kept_ids]
            retired_by_id: dict[str, str] = {}
            for item in retired:
                if not isinstance(item, dict):
                    return None
                skill_id = _skill_id(str(item.get("skill_id") or ""))
                replacement_id = _skill_id(
                    str(item.get("replacement_skill_id") or "")
                )
                if not skill_id or not replacement_id:
                    return None
                assigned.append(skill_id)
                retired_by_id[skill_id] = replacement_id

            prepared = []
            result_target_ids = []
            source_to_target: dict[str, str] = {}
            for item in results:
                if not isinstance(item, dict):
                    return None
                target_id = _skill_id(str(item.get("skill_id") or ""))
                raw_markdown = str(item.get("markdown") or "").strip()
                raw_sources = list(item.get("source_skill_ids", []))
                existing_sources = [
                    _skill_id(str(source))
                    for source in raw_sources
                    if str(source) != PROPOSED_SKILL_TOKEN
                ]
                includes_proposed = PROPOSED_SKILL_TOKEN in raw_sources
                if (
                    not target_id
                    or not raw_markdown
                    or not raw_sources
                    or any(source not in active_ids for source in existing_sources)
                ):
                    return None
                document = SkillDocument.from_markdown(raw_markdown, target_id)
                markdown = document.to_markdown()
                if not document.description or not document.instructions:
                    return None
                document.warn_if_not_initial_only()
                if target_id in active_ids:
                    if target_id not in existing_sources:
                        return None
                    manifest = self.files.load_manifest(target_id)
                    if not isinstance(manifest, dict):
                        return None
                    parent_version = int(manifest["active_version"])
                    version = int(manifest.get("latest_version", parent_version)) + 1
                    manifest = dict(manifest)
                else:
                    if self.files.load_manifest(target_id) is not None:
                        return None
                    parent_version = None
                    version = 1
                    now = utc_now()
                    manifest = {
                        "skill_id": target_id,
                        "active_version": None,
                        "latest_version": version,
                        "created_at": now,
                        "updated_at": now,
                    }

                evidence = []
                if includes_proposed:
                    evidence.extend(proposed_evidence)
                for source in existing_sources:
                    evidence.extend(active[source].get("evidence_ids") or [])
                    assigned.append(source)
                    source_to_target[source] = target_id
                evidence = list(dict.fromkeys(str(value) for value in evidence))
                prepared.append(
                    {
                        "skill_id": target_id,
                        "version": version,
                        "parent_version": parent_version,
                        "manifest": manifest,
                        "markdown": markdown,
                        "evidence_ids": evidence,
                        "source_skill_ids": raw_sources,
                        "includes_proposed": includes_proposed,
                    }
                )
                result_target_ids.append(target_id)

            if (
                len(assigned) != len(set(assigned))
                or set(assigned) != active_ids
                or len(result_target_ids) != len(set(result_target_ids))
                or set(result_target_ids).intersection(
                    set(kept_ids) | set(retired_by_id)
                )
            ):
                return None
            resulting_ids = set(kept_ids) | set(result_target_ids)
            if (
                not resulting_ids
                or self._over_capacity(len(resulting_ids))
                or any(
                    replacement not in resulting_ids
                    for replacement in retired_by_id.values()
                )
            ):
                return None

            now = utc_now()
            for item in prepared:
                self.files.save_version(
                    skill_id=item["skill_id"],
                    version=item["version"],
                    status="active",
                    markdown=item["markdown"],
                    evidence_ids=item["evidence_ids"],
                    parent_version=item["parent_version"],
                    extra_metadata={
                        "evolution_operation": "REORGANIZE",
                        "proposed_batch": int(batch_id),
                        "source_skill_ids": item["source_skill_ids"],
                        "includes_proposed_skill": item["includes_proposed"],
                        "proposed_skill_id": proposed_id,
                    },
                )

            for skill_id in sorted(active_ids - set(kept_ids)):
                manifest = self.files.load_manifest(skill_id)
                if not isinstance(manifest, dict):
                    return None
                version = int(manifest["active_version"])
                meta_path = self.files.version_dir(skill_id, version) / "meta.json"
                meta = self.state.load_json(meta_path, {})
                if skill_id in retired_by_id:
                    replacement_id = retired_by_id[skill_id]
                    meta.update(
                        status="deprecated",
                        replaced_by=replacement_id,
                        reorganization_batch=int(batch_id),
                        updated_at=now,
                    )
                    manifest.update(
                        active_version=None,
                        replaced_by=replacement_id,
                        updated_at=now,
                    )
                else:
                    target_id = source_to_target[skill_id]
                    target_item = next(
                        item for item in prepared if item["skill_id"] == target_id
                    )
                    if skill_id == target_id:
                        meta.update(
                            status="deprecated",
                            superseded_by_version=target_item["version"],
                            reorganization_batch=int(batch_id),
                            updated_at=now,
                        )
                    else:
                        meta.update(
                            status="deprecated",
                            merged_into=target_id,
                            reorganization_batch=int(batch_id),
                            updated_at=now,
                        )
                        manifest.update(
                            active_version=None,
                            merged_into=target_id,
                            updated_at=now,
                        )
                self.state.save_json(meta_path, meta)
                self.files.save_manifest(manifest)

            applied_results = []
            for item in prepared:
                manifest = item["manifest"]
                manifest.update(
                    active_version=item["version"],
                    latest_version=item["version"],
                    updated_at=now,
                )
                self.files.save_manifest(manifest)
                applied_results.append(
                    {
                        "skill_id": item["skill_id"],
                        "version": item["version"],
                        "status": "active",
                        "source_skill_ids": list(item["source_skill_ids"]),
                        "source_insight_ids": list(item["evidence_ids"]),
                    }
                )

            next_active_versions = {
                skill_id: baseline_active_versions[skill_id]
                for skill_id in kept_ids
            }
            next_active_versions.update(
                {
                    str(item["skill_id"]): int(item["version"])
                    for item in prepared
                }
            )
            self.files.save_active_library(
                next_active_versions,
                expected_versions=baseline_active_versions,
            )
            self.files.archive_retired(active_ids - resulting_ids)

        return {
            "operation": "REORGANIZE",
            "proposed_skill_id": proposed_id,
            "proposed_skill_disposition": plan["proposed_skill_disposition"],
            "kept_skill_ids": kept_ids,
            "retired_skills": retired,
            "result_skills": applied_results,
            "active_skill_count": len(resulting_ids),
            "status": "active",
            "execution_stage": SKILL_EXECUTION_STAGE,
        }

    def record_usage(
        self,
        *,
        task_id: str,
        selections: Iterable[dict[str, Any]],
        first_attempt_score: float | None,
        final_reward: float | None = None,
    ) -> None:
        """Record direct-active Skill performance for later library evolution."""
        if first_attempt_score is not None and not 0.0 <= float(first_attempt_score) <= 1.0:
            raise ValueError("first_attempt_score must be between 0 and 1")
        if final_reward is not None and not 0.0 <= float(final_reward) <= 1.0:
            raise ValueError("final_reward must be between 0 and 1")
        with self.state.locked():
            for selection in selections:
                if str(selection.get("execution_stage") or SKILL_EXECUTION_STAGE) != (
                    SKILL_EXECUTION_STAGE
                ):
                    continue
                skill_id = _skill_id(str(selection["skill_id"]))
                version = int(selection["version"])
                if not (
                    self.files.version_dir(skill_id, version) / "meta.json"
                ).is_file():
                    continue
                value: dict[str, Any] = {
                    "task_id": task_id,
                    "round_id": self.state.round_id,
                    "skill_id": skill_id,
                    "version": version,
                    "execution_stage": SKILL_EXECUTION_STAGE,
                    "created_at": utc_now(),
                }
                if first_attempt_score is not None:
                    value["first_attempt_score"] = float(first_attempt_score)
                if final_reward is not None:
                    value["final_reward"] = float(final_reward)
                self.state.create_record(
                    "skill_usage",
                    f"{self.state.episode_key(task_id)}:{skill_id}:{version}",
                    value,
                )


__all__ = ["PROPOSED_SKILL_TOKEN", "SkillRegistry"]

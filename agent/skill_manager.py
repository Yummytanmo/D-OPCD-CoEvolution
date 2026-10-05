from __future__ import annotations

from pathlib import Path
from typing import Any

from agent.evolution.skills.document import SkillDocument


class SkillManager:
    def __init__(self, skills_dir="agent/skills", registry=None):
        self.skills_dir = Path(skills_dir)
        self.registry = registry
        self.skills = self._load_skills() if registry is None else {}

    def _load_skills(self) -> dict[str, dict[str, Any]]:
        skills: dict[str, dict[str, Any]] = {}
        if not self.skills_dir.exists():
            return skills
        for directory in sorted(self.skills_dir.iterdir()):
            path = directory / "SKILL.md"
            if not path.is_file():
                continue
            skill_id = directory.name
            document = SkillDocument.from_markdown(
                path.read_text(encoding="utf-8"),
                skill_id,
            )
            skills[skill_id] = {
                "id": skill_id,
                "skill_id": skill_id,
                "version": 0,
                "status": "active",
                **document.runtime_fields(),
            }
        return skills

    def active_skills(self) -> list[dict[str, Any]]:
        """Return the complete active library for LLM routing."""
        if self.registry is not None:
            return self.registry.runtime_skills()
        return list(self.skills.values())

    def get_skill(self, skill_id: str) -> dict[str, Any] | None:
        if self.registry is not None:
            return self.registry.get(skill_id)
        return self.skills.get(skill_id)

    def get_skill_manifest(self) -> str:
        candidates = self.active_skills()
        if self.registry is not None:
            return self.registry.manifest(candidates)
        return "\n".join(
            "- SKILL_ID: {skill_id}\n  NAME: {name}\n"
            "  DESCRIPTION: {description}".format(
                skill_id=skill["skill_id"],
                name=skill.get("skill_name") or skill["skill_id"],
                description=skill.get("description") or "",
            )
            for skill in candidates
        )

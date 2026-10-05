"""Render natural-language insights and skills into agent prompts."""

from __future__ import annotations

from typing import Any, Iterable

from agent.evolution.memory.insights import InsightStore
from agent.evolution.skills.document import (
    SKILL_EXECUTION_STAGE,
    SKILL_KIND,
    SkillDocument,
)


class PromptCompiler:
    @staticmethod
    def initial_skill_rules(skill: dict[str, Any] | None) -> str | None:
        """Return instructions for the one pre-generation prompt rewrite."""
        if skill is None:
            return None
        if str(skill.get("kind") or SKILL_KIND) != SKILL_KIND:
            return None
        if str(skill.get("execution_stage") or SKILL_EXECUTION_STAGE) != (
            SKILL_EXECUTION_STAGE
        ):
            return None
        document = SkillDocument.from_markdown(
            str(skill.get("markdown") or ""),
            str(skill.get("skill_id") or "skill"),
        )
        instructions = document.instructions.strip()
        return instructions or None

    @classmethod
    def initial_skills_rules(
        cls,
        skills: Iterable[dict[str, Any]],
    ) -> str | None:
        """Combine every selected initial-prompt Skill for one coherent rewrite."""
        sections = []
        for index, skill in enumerate(skills, 1):
            instructions = cls.initial_skill_rules(skill)
            if instructions:
                sections.append(
                    f"### Selected Skill {index}\n{instructions}"
                )
        return "\n\n".join(sections) or None

    @staticmethod
    def insights(insights: Iterable[dict[str, Any]]) -> str | None:
        return InsightStore.render(insights)

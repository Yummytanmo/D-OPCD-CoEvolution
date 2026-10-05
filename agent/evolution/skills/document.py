"""Standard SKILL.md representation shared by static and evolved Skills."""

from __future__ import annotations

import json
import re
import warnings
from dataclasses import dataclass
from typing import Any

from agent.evolution.soft_constraints import (
    SoftConstraintWarning,
    warn_soft_limit,
    warn_soft_range,
)


SKILL_KIND = "initial_prompt_rewriter"
SKILL_EXECUTION_STAGE = "before_first_generation"
SKILL_INPUT_CONTRACT = "original_user_prompt"
SKILL_OUTPUT_CONTRACT = "enhanced_image_generation_prompt"


def _strip_fence(value: str) -> str:
    text = value.strip()
    match = re.fullmatch(r"```(?:markdown|md)?\s*\n(.*?)\n```", text, re.DOTALL)
    return match.group(1).strip() if match else text


def _yaml_scalar(value: str) -> str:
    """Read the single-line scalar form emitted by :meth:`to_markdown`."""
    text = value.strip()
    if text.startswith('"') and text.endswith('"'):
        try:
            parsed = json.loads(text)
            return str(parsed)
        except json.JSONDecodeError:
            pass
    if text.startswith("'") and text.endswith("'"):
        return text[1:-1].replace("''", "'")
    return text


def _split_frontmatter(markdown: str) -> tuple[dict[str, str], str]:
    match = re.match(
        r"\A---[ \t]*\n(.*?)\n---[ \t]*(?:\n|\Z)",
        markdown,
        re.DOTALL,
    )
    if not match:
        return {}, markdown
    metadata: dict[str, str] = {}
    for line in match.group(1).splitlines():
        key, separator, raw_value = line.partition(":")
        if separator and key.strip():
            metadata[key.strip()] = _yaml_scalar(raw_value)
    return metadata, markdown[match.end() :].lstrip()


def _standard_name(value: str, fallback: str) -> str:
    candidate = str(value or "").strip().casefold()
    if candidate.startswith("skill-"):
        candidate = candidate.removeprefix("skill-")
    if (
        re.fullmatch(r"[a-z0-9]+(?:-[a-z0-9]+)*", candidate)
        and len(candidate) <= 64
    ):
        return candidate
    normalized = re.sub(
        r"[^a-z0-9]+",
        "-",
        candidate or fallback.casefold(),
    ).strip("-")
    if normalized.startswith("skill-"):
        normalized = normalized.removeprefix("skill-")
    normalized = normalized[:64].rstrip("-")
    return normalized or "prompt-rewriter"


def _human_title(value: str) -> str:
    """Keep the heading human-readable without repeating the document type."""
    return re.sub(r"^skill\s+", "", str(value).strip(), flags=re.IGNORECASE)


def _deduplicate_routing_text(description: str, instructions: str) -> str:
    """Migrate older documents that copied their routing paragraph into the body."""
    if instructions.startswith(description):
        remainder = instructions[len(description) :].lstrip(" \t\r\n-")
        if remainder:
            return remainder
    return instructions


def _section(markdown: str, title: str) -> str:
    match = re.search(
        rf"^##\s+{re.escape(title)}\s*$\n(.*?)(?=^##\s+|\Z)",
        markdown,
        re.IGNORECASE | re.MULTILINE | re.DOTALL,
    )
    return match.group(1).strip() if match else ""


def _first_paragraph(markdown: str) -> str:
    body = re.sub(r"^#.*$", "", markdown, count=1, flags=re.MULTILINE)
    body = re.split(r"^##\s+", body, maxsplit=1, flags=re.MULTILINE)[0]
    for paragraph in re.split(r"\n\s*\n", body):
        value = paragraph.strip().strip("-").strip()
        if value and value != "---":
            return value
    return ""


def _natural_body(markdown: str) -> str:
    """Preserve a free-form authored Skill when no fixed sections were requested."""
    return re.sub(r"^#\s+.*$", "", markdown, count=1, flags=re.MULTILINE).strip()


@dataclass(frozen=True)
class SkillDocument:
    title: str
    description: str
    instructions: str
    output_format: str = "Return ONLY the final enhanced prompt text."
    skill_name: str = ""

    @classmethod
    def from_markdown(cls, value: str, fallback_id: str) -> "SkillDocument":
        markdown = _strip_fence(str(value or ""))
        metadata, body = _split_frontmatter(markdown)
        title_match = re.search(
            r"^#\s+(?:Skill:\s*)?(.+?)\s*$",
            body,
            re.IGNORECASE | re.MULTILINE,
        )
        title = _human_title(
            title_match.group(1).strip()
            if title_match
            else fallback_id.replace("_", " ").title()
        )
        description = (
            metadata.get("description")
            or _section(body, "Description")
            or _first_paragraph(body)
        )
        instructions = _section(body, "Instructions")
        if not instructions:
            instructions = (
                _natural_body(body)
                if not re.search(r"^##\s+", body, re.MULTILINE)
                else description
            )
        instructions = _deduplicate_routing_text(description, instructions)
        output_format = _section(body, "Output Format")
        return cls(
            title=title,
            description=description.strip(),
            instructions=instructions.strip(),
            output_format=(
                output_format.strip()
                or "Return ONLY the final enhanced prompt text."
            ),
            skill_name=_standard_name(
                metadata.get("name", ""),
                fallback_id or title,
            ),
        )

    @classmethod
    def from_structured(cls, value: dict[str, Any]) -> "SkillDocument" | None:
        title = _human_title(str(value.get("title") or ""))
        description = str(value.get("description") or "").strip()
        triggers = [
            str(item).strip()
            for item in value.get("trigger_when", [])
            if str(item).strip()
        ]
        anti_triggers = [
            str(item).strip()
            for item in value.get("do_not_trigger_when", [])
            if str(item).strip()
        ]
        actions = [
            str(item).strip()
            for item in value.get("instructions", [])
            if str(item).strip()
        ]
        if not title or not description or not triggers or not anti_triggers or not actions:
            return None
        warning_name = f"Skill principles ({title})"
        warn_soft_range(
            warning_name,
            actual=len(actions),
            preferred_min=4,
            preferred_max=10,
            unit="principles",
        )
        routing = (
            f"{description}\n"
            f"- **Trigger when**: {'; '.join(triggers)}.\n"
            f"- **Do NOT trigger when**: {'; '.join(anti_triggers)}."
        )
        instructions = "\n".join(
            f"{index}. {action}" for index, action in enumerate(actions, 1)
        )
        document = cls(
            title=title,
            description=routing,
            instructions=instructions,
            output_format="Return ONLY the final enhanced prompt text.",
            skill_name=_standard_name(
                str(value.get("suggested_id") or ""),
                title,
            ),
        )
        warn_soft_limit(
            f"Skill document size ({title})",
            actual=len(document.to_markdown()),
            preferred_max=4_500,
            unit="characters",
        )
        return document

    def to_markdown(self) -> str:
        skill_name = _standard_name(self.skill_name, self.title)
        description = " ".join(self.description.split())
        return (
            "---\n"
            f"name: {skill_name}\n"
            f"description: {json.dumps(description, ensure_ascii=False)}\n"
            "---\n\n"
            f"# Skill: {self.title}\n\n"
            f"## Instructions\n{self.instructions}\n\n"
            f"## Output Format\n{self.output_format}\n"
        )

    def runtime_fields(self) -> dict[str, str]:
        return {
            "name": self.title,
            "skill_name": _standard_name(self.skill_name, self.title),
            "description": self.description,
            "instructions": self.instructions,
            "output_format": self.output_format,
            "markdown": self.to_markdown(),
            "kind": SKILL_KIND,
            "execution_stage": SKILL_EXECUTION_STAGE,
            "input_contract": SKILL_INPUT_CONTRACT,
            "output_contract": SKILL_OUTPUT_CONTRACT,
        }

    def is_initial_only(self) -> bool:
        """Detect whether a document appears independent of post-generation state."""
        text = f"{self.description}\n{self.instructions}".casefold()
        forbidden_phrases = (
            "generated image",
            "previous image",
            "first image",
            "image result",
            "attempt history",
            "previous attempt",
            "failed check",
            "check result",
            "verification result",
            "evaluator",
            "reward",
            "feedback",
            "retry",
            "refinement loop",
            "during refinement",
            "next attempt",
        )
        forbidden_words = re.search(
            r"\b(attempts?|retries|retrying|refinement|rewards?|feedback|evaluators?)\b",
            text,
        )
        return forbidden_words is None and not any(
            phrase in text for phrase in forbidden_phrases
        )

    def warn_if_not_initial_only(self) -> None:
        """Report a likely stage mismatch without rejecting the Skill document."""
        if self.is_initial_only():
            return
        warnings.warn(
            f"Soft constraint missed for Skill execution scope ({self.title}): "
            "content may depend on generation attempts, evaluation, or refinement; "
            "retained the Skill document.",
            SoftConstraintWarning,
            stacklevel=1,
        )


__all__ = [
    "SKILL_EXECUTION_STAGE",
    "SKILL_INPUT_CONTRACT",
    "SKILL_KIND",
    "SKILL_OUTPUT_CONTRACT",
    "SkillDocument",
]

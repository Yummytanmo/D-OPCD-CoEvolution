"""Staged exact-edit workspace for atomic high-level Skill management."""

from __future__ import annotations

import copy
from dataclasses import dataclass, field
from difflib import SequenceMatcher
from typing import Annotated, Any, Literal

from pydantic import Field

from agent.evolution.skills.document import SkillDocument
from agent.evolution.skills.registry import _skill_id
from agent.evolution.soft_constraints import warn_soft_limit
from agent.tooling import FunctionTool, ToolArguments


DEFAULT_MAX_SKILL_CHARACTERS = 4_500


class SkillCapacityError(ValueError):
    """The staged transaction would exceed the active Skill limit."""

    def __init__(self, final_count: int, max_skills: int) -> None:
        self.final_count = int(final_count)
        self.max_skills = int(max_skills)
        super().__init__(
            f"staged library has {self.final_count} Skills; "
            f"consolidate it to at most {self.max_skills}"
        )


class SkillLengthError(ValueError):
    """One or more staged active Skills exceed the configured character limit."""

    def __init__(self, over_limit: dict[str, int], max_characters: int) -> None:
        self.over_limit = dict(over_limit)
        self.max_characters = int(max_characters)
        details = ", ".join(
            f"{skill_id}={characters}"
            for skill_id, characters in sorted(self.over_limit.items())
        )
        super().__init__(
            f"staged Skills exceed {self.max_characters} characters: {details}"
        )


class _InspectArgs(ToolArguments):
    skill_id: str = Field(min_length=1)


class _CreateArgs(ToolArguments):
    suggested_id: str = Field(min_length=1)
    content: str = Field(min_length=1)
    source_insight_refs: list[str] = Field(min_length=1)


class _EditArgs(ToolArguments):
    skill_id: str = Field(min_length=1)
    source_insight_refs: list[str] = Field(default_factory=list)


class _ReplaceArgs(_EditArgs):
    old_text: str = Field(min_length=1)
    new_text: str


class _InsertArgs(_EditArgs):
    anchor: str = Field(min_length=1)
    text: str = Field(min_length=1)
    position: Literal["before", "after"]


class _DeleteArgs(_EditArgs):
    text: str = Field(min_length=1)


class _RenameArgs(_EditArgs):
    new_title: str = Field(min_length=1)


class _SummaryArgs(_EditArgs):
    content: str = Field(min_length=1)


class _SplitSkill(ToolArguments):
    suggested_id: str = Field(min_length=1)
    content: str = Field(min_length=1)


class _SplitArgs(_EditArgs):
    new_skills: list[_SplitSkill] = Field(min_length=2, max_length=2)


class _MergeArgs(ToolArguments):
    target_skill_id: str = Field(min_length=1)
    merged_skill_ids: list[str] = Field(min_length=1)
    content: str = Field(min_length=1)
    source_insight_refs: list[str] = Field(default_factory=list)


class _RetireArgs(ToolArguments):
    skill_id: str = Field(min_length=1)
    replacement_skill_id: str = Field(min_length=1)


class _FinishArgs(ToolArguments):
    pass


class _CreateOperation(ToolArguments):
    op: Literal["CREATE"]
    suggested_id: str = Field(min_length=1)
    content: str = Field(min_length=1)
    source_insight_refs: list[str] = Field(min_length=1)


class _ReplaceOperation(ToolArguments):
    op: Literal["REPLACE_TEXT"]
    skill_id: str = Field(min_length=1)
    old_text: str = Field(min_length=1)
    new_text: str
    source_insight_refs: list[str] = Field(default_factory=list)


class _InsertOperation(ToolArguments):
    op: Literal["INSERT_TEXT"]
    skill_id: str = Field(min_length=1)
    anchor: str = Field(min_length=1)
    text: str = Field(min_length=1)
    position: Literal["before", "after"]
    source_insight_refs: list[str] = Field(default_factory=list)


class _DeleteOperation(ToolArguments):
    op: Literal["DELETE_TEXT"]
    skill_id: str = Field(min_length=1)
    text: str = Field(min_length=1)
    source_insight_refs: list[str] = Field(default_factory=list)


class _RenameOperation(ToolArguments):
    op: Literal["RENAME"]
    skill_id: str = Field(min_length=1)
    new_title: str = Field(min_length=1)
    source_insight_refs: list[str] = Field(default_factory=list)


class _SummaryOperation(ToolArguments):
    op: Literal["SUMMARY"]
    skill_id: str = Field(min_length=1)
    content: str = Field(min_length=1)
    source_insight_refs: list[str] = Field(default_factory=list)


class _SplitOperation(ToolArguments):
    op: Literal["SPLIT"]
    skill_id: str = Field(min_length=1)
    new_skills: list[_SplitSkill] = Field(min_length=2, max_length=2)
    source_insight_refs: list[str] = Field(default_factory=list)


class _MergeOperation(ToolArguments):
    op: Literal["MERGE"]
    target_skill_id: str = Field(min_length=1)
    merged_skill_ids: list[str] = Field(min_length=1)
    content: str = Field(min_length=1)
    source_insight_refs: list[str] = Field(default_factory=list)


class _RetireOperation(ToolArguments):
    op: Literal["RETIRE"]
    skill_id: str = Field(min_length=1)
    replacement_skill_id: str = Field(min_length=1)


_SkillOperation = Annotated[
    _CreateOperation
    | _ReplaceOperation
    | _InsertOperation
    | _DeleteOperation
    | _RenameOperation
    | _SummaryOperation
    | _SplitOperation
    | _MergeOperation
    | _RetireOperation,
    Field(discriminator="op"),
]


class _ApplySkillOperationsArgs(ToolArguments):
    operations: list[_SkillOperation] = Field(default_factory=list)


@dataclass
class _Draft:
    skill_id: str
    base_version: int | None
    markdown: str
    existing_evidence_ids: list[str] = field(default_factory=list)
    new_source_insight_ids: set[str] = field(default_factory=set)
    inherited_skill_ids: set[str] = field(default_factory=set)


@dataclass(frozen=True)
class _PatchMatch:
    start: int
    end: int
    similarity: float


class SkillWorkspace:
    """Apply model edits only to a private copy and export one commit plan."""

    def __init__(
        self,
        *,
        active_skills: list[dict[str, Any]],
        reserved_skill_ids: set[str],
        insight_id_by_ref: dict[str, str],
        required_source_insight_ids: set[str],
        max_skills: int | None,
    ) -> None:
        self.baseline_versions = {
            str(item["skill_id"]): int(item["version"]) for item in active_skills
        }
        self.drafts = {
            str(item["skill_id"]): _Draft(
                skill_id=str(item["skill_id"]),
                base_version=int(item["version"]),
                markdown=str(item.get("markdown") or ""),
                existing_evidence_ids=list(item.get("evidence_ids") or []),
                inherited_skill_ids={str(item["skill_id"])},
            )
            for item in active_skills
        }
        self.reserved_skill_ids = set(reserved_skill_ids) | set(self.drafts)
        self.insight_id_by_ref = dict(insight_id_by_ref)
        self.required_source_insight_ids = set(required_source_insight_ids)
        self.max_skills = int(max_skills) if max_skills is not None else None
        self.retired: dict[str, tuple[str, ...]] = {}
        self.touched: set[str] = set()
        self.operation_log: list[dict[str, Any]] = []
        self.finished = False

    @staticmethod
    def _canonical(content: str, skill_id: str) -> str:
        document = SkillDocument.from_markdown(str(content).strip(), skill_id)
        if not document.description or not document.instructions:
            raise ValueError("Skill must contain a description and instructions")
        document.warn_if_not_initial_only()
        markdown = document.to_markdown()
        warn_soft_limit(
            f"Skill document size ({document.title})",
            actual=len(markdown),
            preferred_max=DEFAULT_MAX_SKILL_CHARACTERS,
            unit="characters",
        )
        return markdown

    def _draft(self, raw_skill_id: str) -> _Draft:
        skill_id = _skill_id(raw_skill_id)
        if skill_id in self.retired:
            raise ValueError(f"Skill is already staged for retirement: {skill_id}")
        try:
            return self.drafts[skill_id]
        except KeyError as error:
            raise ValueError(f"unknown active Skill: {raw_skill_id}") from error

    def _source_ids(self, references: list[str]) -> set[str]:
        refs = list(dict.fromkeys(str(item) for item in references))
        unknown = [item for item in refs if item not in self.insight_id_by_ref]
        if unknown:
            raise ValueError(f"unknown source_insight_refs: {unknown}")
        return {self.insight_id_by_ref[item] for item in refs}

    def _stage_markdown(
        self,
        draft: _Draft,
        markdown: str,
        source_refs: list[str],
        operation: str,
    ) -> dict[str, Any]:
        canonical = self._canonical(markdown, draft.skill_id)
        draft.markdown = canonical
        draft.new_source_insight_ids.update(self._source_ids(source_refs))
        self.touched.add(draft.skill_id)
        self.operation_log.append(
            {"operation": operation, "skill_id": draft.skill_id}
        )
        document = SkillDocument.from_markdown(draft.markdown, draft.skill_id)
        return {
            "staged": True,
            "skill_id": draft.skill_id,
            "title": document.title,
            "characters": len(draft.markdown),
        }

    @staticmethod
    def _locate_patch(text: str, old: str) -> _PatchMatch:
        count = text.count(old)
        if count == 1:
            start = text.index(old)
            return _PatchMatch(
                start=start,
                end=start + len(old),
                similarity=1.0,
            )
        if count > 1:
            raise ValueError(f"exact patch requires one match; found {count}")

        # A short fuzzy target is too easy to match accidentally. Skill Manager
        # patches are normally copied sentences or sections, so keep the fallback
        # limited to substantial, nearly identical text.
        if len(old) < 32:
            raise ValueError(
                "exact patch requires one match; found 0; fuzzy fallback "
                "requires at least 32 characters"
            )
        if len(old) > len(text):
            raise ValueError(
                "exact patch requires one match; found 0; fuzzy fallback "
                "found no safely aligned candidate"
            )

        alignment = SequenceMatcher(None, old, text, autojunk=False)
        blocks = [block for block in alignment.get_matching_blocks() if block.size]
        minimum_anchor = max(12, (len(old) + 3) // 4)
        useful_blocks = [block for block in blocks if block.size >= minimum_anchor]
        if not useful_blocks:
            raise ValueError(
                "exact patch requires one match; found 0; fuzzy fallback "
                "found no safely aligned candidate"
            )

        candidate_starts: set[int] = set()
        for block in useful_blocks:
            anchor = old[block.a : block.a + block.size]
            position = text.find(anchor)
            while position >= 0:
                start = position - block.a
                if 0 <= start <= len(text) - len(old):
                    candidate_starts.add(start)
                position = text.find(anchor, position + 1)

        candidates = sorted(
            (
                _PatchMatch(
                    start=start,
                    end=start + len(old),
                    similarity=SequenceMatcher(
                        None,
                        old,
                        text[start : start + len(old)],
                        autojunk=False,
                    ).ratio(),
                )
                for start in candidate_starts
            ),
            key=lambda item: item.similarity,
            reverse=True,
        )
        if not candidates or candidates[0].similarity < 0.97:
            best = candidates[0].similarity if candidates else 0.0
            raise ValueError(
                "exact patch requires one match; found 0; fuzzy fallback best "
                f"similarity {best:.4f} is below 0.9700"
            )

        # Several matching blocks from one local edit can produce starts a few
        # characters apart. Collapse those before comparing distinct regions.
        distinct: list[_PatchMatch] = []
        local_distance = max(2, min(16, len(old) // 50))
        for candidate in candidates:
            if any(
                abs(candidate.start - existing.start) <= local_distance
                for existing in distinct
            ):
                continue
            distinct.append(candidate)

        best = distinct[0]
        if len(distinct) > 1 and best.similarity - distinct[1].similarity < 0.02:
            raise ValueError(
                "exact patch requires one match; found 0; fuzzy fallback is "
                f"ambiguous (best={best.similarity:.4f}, "
                f"second={distinct[1].similarity:.4f})"
            )
        return best

    @classmethod
    def _replace_once(cls, text: str, old: str, new: str) -> str:
        match = cls._locate_patch(text, old)
        return f"{text[:match.start]}{new}{text[match.end:]}"

    def inspect(self, arguments: _InspectArgs) -> dict[str, Any]:
        draft = self._draft(arguments.skill_id)
        return {
            "skill_id": draft.skill_id,
            "content": draft.markdown,
            "staged_changes": draft.skill_id in self.touched,
        }

    def create(self, arguments: _CreateArgs) -> dict[str, Any]:
        skill_id = _skill_id(arguments.suggested_id)
        if not skill_id:
            raise ValueError("suggested_id does not contain a valid Skill identifier")
        if skill_id in self.reserved_skill_ids:
            raise ValueError(f"Skill identifier already exists or is reserved: {skill_id}")
        source_ids = self._source_ids(arguments.source_insight_refs)
        if not source_ids:
            raise ValueError("a new Skill requires supporting Insight references")
        markdown = self._canonical(arguments.content, skill_id)
        self.drafts[skill_id] = _Draft(
            skill_id=skill_id,
            base_version=None,
            markdown=markdown,
            new_source_insight_ids=source_ids,
        )
        self.reserved_skill_ids.add(skill_id)
        self.touched.add(skill_id)
        self.operation_log.append({"operation": "CREATE", "skill_id": skill_id})
        return {
            "staged": True,
            "skill_id": skill_id,
            "active_skill_count_if_finished": self._final_count(),
        }

    def replace(self, arguments: _ReplaceArgs) -> dict[str, Any]:
        draft = self._draft(arguments.skill_id)
        updated = self._replace_once(
            draft.markdown, arguments.old_text, arguments.new_text
        )
        return self._stage_markdown(
            draft, updated, arguments.source_insight_refs, "REPLACE_TEXT"
        )

    def insert(self, arguments: _InsertArgs) -> dict[str, Any]:
        draft = self._draft(arguments.skill_id)
        match = self._locate_patch(draft.markdown, arguments.anchor)
        matched_anchor = draft.markdown[match.start : match.end]
        replacement = (
            f"{arguments.text}{matched_anchor}"
            if arguments.position == "before"
            else f"{matched_anchor}{arguments.text}"
        )
        updated = (
            f"{draft.markdown[:match.start]}{replacement}"
            f"{draft.markdown[match.end:]}"
        )
        return self._stage_markdown(
            draft, updated, arguments.source_insight_refs, "INSERT_TEXT"
        )

    def delete(self, arguments: _DeleteArgs) -> dict[str, Any]:
        draft = self._draft(arguments.skill_id)
        updated = self._replace_once(draft.markdown, arguments.text, "")
        return self._stage_markdown(
            draft, updated, arguments.source_insight_refs, "DELETE_TEXT"
        )

    def rename(self, arguments: _RenameArgs) -> dict[str, Any]:
        draft = self._draft(arguments.skill_id)
        document = SkillDocument.from_markdown(draft.markdown, draft.skill_id)
        updated = SkillDocument(
            title=arguments.new_title.strip(),
            description=document.description,
            instructions=document.instructions,
            output_format=document.output_format,
            skill_name=document.skill_name,
        ).to_markdown()
        return self._stage_markdown(
            draft, updated, arguments.source_insight_refs, "RENAME"
        )

    def summary(self, arguments: _SummaryArgs) -> dict[str, Any]:
        draft = self._draft(arguments.skill_id)
        current = SkillDocument.from_markdown(draft.markdown, draft.skill_id)
        canonical = self._canonical(arguments.content, draft.skill_id)
        summarized = SkillDocument.from_markdown(canonical, draft.skill_id)
        if (
            summarized.title != current.title
            or summarized.skill_name != current.skill_name
        ):
            raise ValueError("SUMMARY must preserve the Skill title and frontmatter name")
        if len(canonical) >= len(draft.markdown):
            raise ValueError("SUMMARY content must be shorter than the current Skill")
        return self._stage_markdown(
            draft, canonical, arguments.source_insight_refs, "SUMMARY"
        )

    def split(self, arguments: _SplitArgs) -> dict[str, Any]:
        source = self._draft(arguments.skill_id)
        source_ids = set(source.new_source_insight_ids)
        source_ids.update(self._source_ids(arguments.source_insight_refs))

        prepared: list[tuple[str, str]] = []
        for new_skill in arguments.new_skills:
            skill_id = _skill_id(new_skill.suggested_id)
            if not skill_id:
                raise ValueError(
                    "a SPLIT suggested_id does not contain a valid Skill identifier"
                )
            if skill_id in self.reserved_skill_ids:
                raise ValueError(
                    f"Skill identifier already exists or is reserved: {skill_id}"
                )
            prepared.append((skill_id, self._canonical(new_skill.content, skill_id)))
        new_ids = [skill_id for skill_id, _ in prepared]
        if len(set(new_ids)) != 2:
            raise ValueError("SPLIT must create two distinct Skill identifiers")

        for skill_id, markdown in prepared:
            self.drafts[skill_id] = _Draft(
                skill_id=skill_id,
                base_version=None,
                markdown=markdown,
                new_source_insight_ids=set(source_ids),
                inherited_skill_ids=set(source.inherited_skill_ids),
            )
            self.reserved_skill_ids.add(skill_id)
            self.touched.add(skill_id)
        for retired_id, replacements in list(self.retired.items()):
            if source.skill_id not in replacements:
                continue
            updated: list[str] = []
            for replacement in replacements:
                updated.extend(new_ids if replacement == source.skill_id else [replacement])
            self.retired[retired_id] = tuple(dict.fromkeys(updated))
        if source.base_version is None:
            self.drafts.pop(source.skill_id)
            self.touched.discard(source.skill_id)
        else:
            self.retired[source.skill_id] = tuple(new_ids)
        self.operation_log.append(
            {
                "operation": "SPLIT",
                "skill_id": source.skill_id,
                "new_skill_ids": new_ids,
            }
        )
        return {
            "staged": True,
            "split_skill_id": source.skill_id,
            "new_skill_ids": new_ids,
            "active_skill_count_if_finished": self._final_count(),
        }

    def merge(self, arguments: _MergeArgs) -> dict[str, Any]:
        target = self._draft(arguments.target_skill_id)
        merged_ids = list(
            dict.fromkeys(_skill_id(item) for item in arguments.merged_skill_ids)
        )
        if target.skill_id in merged_ids:
            raise ValueError("merge target cannot also be a merged source")
        inherited = set(target.inherited_skill_ids)
        for skill_id in merged_ids:
            source = self._draft(skill_id)
            inherited.update(source.inherited_skill_ids)
            target.new_source_insight_ids.update(source.new_source_insight_ids)
            for retired_id, replacements in list(self.retired.items()):
                if skill_id not in replacements:
                    continue
                updated = [
                    target.skill_id if replacement == skill_id else replacement
                    for replacement in replacements
                ]
                self.retired[retired_id] = tuple(dict.fromkeys(updated))
            if source.base_version is None:
                self.drafts.pop(skill_id)
                self.touched.discard(skill_id)
            else:
                self.retired[skill_id] = (target.skill_id,)
        target.inherited_skill_ids = inherited
        result = self._stage_markdown(
            target,
            arguments.content,
            arguments.source_insight_refs,
            "MERGE",
        )
        result["merged_skill_ids"] = merged_ids
        result["active_skill_count_if_finished"] = self._final_count()
        return result

    def retire(self, arguments: _RetireArgs) -> dict[str, Any]:
        source = self._draft(arguments.skill_id)
        replacement = self._draft(arguments.replacement_skill_id)
        if source.skill_id == replacement.skill_id:
            raise ValueError("a Skill cannot replace itself")
        if source.base_version is None:
            raise ValueError("discard a new draft by not creating it; it cannot be retired")
        replacement.inherited_skill_ids.update(source.inherited_skill_ids)
        self.retired[source.skill_id] = (replacement.skill_id,)
        self.touched.add(replacement.skill_id)
        self.operation_log.append(
            {
                "operation": "RETIRE",
                "skill_id": source.skill_id,
                "replacement_skill_id": replacement.skill_id,
            }
        )
        return {
            "staged": True,
            "retired_skill_id": source.skill_id,
            "replacement_skill_id": replacement.skill_id,
            "active_skill_count_if_finished": self._final_count(),
        }

    def _final_count(self) -> int:
        return len([skill_id for skill_id in self.drafts if skill_id not in self.retired])

    def active_skill_snapshot(self) -> list[dict[str, Any]]:
        """Return complete staged active documents for maintenance prompts."""
        values = []
        for skill_id in sorted(self.drafts):
            if skill_id in self.retired:
                continue
            draft = self.drafts[skill_id]
            document = SkillDocument.from_markdown(draft.markdown, skill_id)
            values.append(
                {
                    "skill_id": skill_id,
                    "title": document.title,
                    "description": document.description,
                    "characters": len(draft.markdown),
                    "content": draft.markdown,
                }
            )
        return values

    def over_limit_skills(self, max_characters: int) -> dict[str, int]:
        limit = int(max_characters)
        return {
            skill_id: len(draft.markdown)
            for skill_id, draft in self.drafts.items()
            if skill_id not in self.retired and len(draft.markdown) > limit
        }

    def validate_skill_lengths(self, max_characters: int) -> None:
        over_limit = self.over_limit_skills(max_characters)
        if over_limit:
            raise SkillLengthError(over_limit, max_characters)

    def finish(self, _: _FinishArgs) -> dict[str, Any]:
        final_ids = {skill_id for skill_id in self.drafts if skill_id not in self.retired}
        if not final_ids and (self.touched or self.retired):
            raise ValueError("the active Skill library cannot be empty")
        if self.max_skills is not None and len(final_ids) > self.max_skills:
            raise SkillCapacityError(len(final_ids), self.max_skills)
        if any(
            replacement not in final_ids
            for replacements in self.retired.values()
            for replacement in replacements
        ):
            raise ValueError("every retired Skill must name an active replacement")
        consumed = {
            insight_id
            for skill_id in self.touched
            for insight_id in self.drafts[skill_id].new_source_insight_ids
        }
        if self.touched and self.required_source_insight_ids and not (
            consumed & self.required_source_insight_ids
        ):
            raise ValueError(
                "a library change must incorporate at least one new_or_changed Insight"
            )
        self.finished = True
        return {
            "ready_to_commit": True,
            "changed_skill_count": len(self.touched),
            "retired_skill_count": len(self.retired),
            "final_skill_count": len(final_ids),
        }

    def plan(self) -> dict[str, Any]:
        if not self.finished:
            raise ValueError("Skill workspace has not been finished")
        results = []
        for skill_id in sorted(self.touched - set(self.retired)):
            draft = self.drafts[skill_id]
            inherited = sorted(draft.inherited_skill_ids)
            operation = (
                "SPLIT"
                if draft.base_version is None and inherited
                else "CREATE"
                if draft.base_version is None
                else "MERGE"
                if len(inherited) > 1
                else "MODIFY"
            )
            results.append(
                {
                    "operation": operation,
                    "skill_id": skill_id,
                    "expected_version": draft.base_version,
                    "markdown": draft.markdown,
                    "new_source_insight_ids": sorted(
                        draft.new_source_insight_ids
                    ),
                    "inherited_skill_ids": inherited,
                }
            )
        return {
            "operation": "SKILL_TRANSACTION",
            "expected_active_versions": dict(self.baseline_versions),
            "result_skills": results,
            "retired_skills": [
                (
                    {
                        "skill_id": skill_id,
                        "replacement_skill_id": replacements[0],
                    }
                    if len(replacements) == 1
                    else {
                        "skill_id": skill_id,
                        "replacement_skill_ids": list(replacements),
                    }
                )
                for skill_id, replacements in sorted(self.retired.items())
            ],
            "tool_operations": list(self.operation_log),
        }

    def apply_plan(
        self,
        arguments: _ApplySkillOperationsArgs,
    ) -> dict[str, Any]:
        """Apply one complete ordered edit plan and validate its final library."""
        snapshot = {
            "drafts": copy.deepcopy(self.drafts),
            "reserved_skill_ids": set(self.reserved_skill_ids),
            "retired": dict(self.retired),
            "touched": set(self.touched),
            "operation_log": list(self.operation_log),
            "finished": self.finished,
        }
        results: list[dict[str, Any]] = []
        try:
            for operation in arguments.operations:
                if isinstance(operation, _CreateOperation):
                    result = self.create(
                        _CreateArgs(
                            suggested_id=operation.suggested_id,
                            content=operation.content,
                            source_insight_refs=operation.source_insight_refs,
                        )
                    )
                elif isinstance(operation, _ReplaceOperation):
                    result = self.replace(
                        _ReplaceArgs(
                            skill_id=operation.skill_id,
                            old_text=operation.old_text,
                            new_text=operation.new_text,
                            source_insight_refs=operation.source_insight_refs,
                        )
                    )
                elif isinstance(operation, _InsertOperation):
                    result = self.insert(
                        _InsertArgs(
                            skill_id=operation.skill_id,
                            anchor=operation.anchor,
                            text=operation.text,
                            position=operation.position,
                            source_insight_refs=operation.source_insight_refs,
                        )
                    )
                elif isinstance(operation, _DeleteOperation):
                    result = self.delete(
                        _DeleteArgs(
                            skill_id=operation.skill_id,
                            text=operation.text,
                            source_insight_refs=operation.source_insight_refs,
                        )
                    )
                elif isinstance(operation, _RenameOperation):
                    result = self.rename(
                        _RenameArgs(
                            skill_id=operation.skill_id,
                            new_title=operation.new_title,
                            source_insight_refs=operation.source_insight_refs,
                        )
                    )
                elif isinstance(operation, _SummaryOperation):
                    result = self.summary(
                        _SummaryArgs(
                            skill_id=operation.skill_id,
                            content=operation.content,
                            source_insight_refs=operation.source_insight_refs,
                        )
                    )
                elif isinstance(operation, _SplitOperation):
                    result = self.split(
                        _SplitArgs(
                            skill_id=operation.skill_id,
                            new_skills=operation.new_skills,
                            source_insight_refs=operation.source_insight_refs,
                        )
                    )
                elif isinstance(operation, _MergeOperation):
                    result = self.merge(
                        _MergeArgs(
                            target_skill_id=operation.target_skill_id,
                            merged_skill_ids=operation.merged_skill_ids,
                            content=operation.content,
                            source_insight_refs=operation.source_insight_refs,
                        )
                    )
                else:
                    result = self.retire(
                        _RetireArgs(
                            skill_id=operation.skill_id,
                            replacement_skill_id=operation.replacement_skill_id,
                        )
                    )
                results.append({"op": operation.op, **result})
            finish_result = self.finish(_FinishArgs())
        except ValueError:
            self.drafts = snapshot["drafts"]
            self.reserved_skill_ids = snapshot["reserved_skill_ids"]
            self.retired = snapshot["retired"]
            self.touched = snapshot["touched"]
            self.operation_log = snapshot["operation_log"]
            self.finished = snapshot["finished"]
            raise
        return {
            **finish_result,
            "operation_results": results,
        }

    def apply_serialized_plan(self, value: dict[str, Any]) -> dict[str, Any]:
        """Validate and execute a JSON operation chain produced without tool calling."""
        arguments = _ApplySkillOperationsArgs.model_validate(value)
        return self.apply_plan(arguments)

    def plan_tool(self) -> FunctionTool:
        return FunctionTool(
            "apply_skill_operations",
            (
                "Apply the complete ordered Skill-library update once. Operations "
                "execute serially in one private transaction; use an empty list for "
                "a deliberate no-op."
            ),
            _ApplySkillOperationsArgs,
            self.apply_plan,
        )

    def tools(self) -> list[FunctionTool]:
        return [
            FunctionTool(
                "inspect_skill",
                "Read the current complete text of one active Skill before editing it.",
                _InspectArgs,
                self.inspect,
            ),
            FunctionTool(
                "create_skill",
                "Stage a distinct high-level Skill as complete natural-language content.",
                _CreateArgs,
                self.create,
            ),
            FunctionTool(
                "replace_skill_text",
                "Replace an exact uniquely matching passage in one Skill.",
                _ReplaceArgs,
                self.replace,
            ),
            FunctionTool(
                "insert_skill_text",
                "Insert text immediately before or after an exact unique anchor in one Skill.",
                _InsertArgs,
                self.insert,
            ),
            FunctionTool(
                "delete_skill_text",
                "Delete an exact uniquely matching passage from one Skill.",
                _DeleteArgs,
                self.delete,
            ),
            FunctionTool(
                "rename_skill",
                "Change a Skill title while preserving its stable skill_id and body.",
                _RenameArgs,
                self.rename,
            ),
            FunctionTool(
                "summarize_skill",
                (
                    "Replace one overgrown Skill with complete shorter content while "
                    "preserving its identity."
                ),
                _SummaryArgs,
                self.summary,
            ),
            FunctionTool(
                "split_skill",
                (
                    "Replace one over-broad Skill with exactly two complete, "
                    "independently routeable Skills."
                ),
                _SplitArgs,
                self.split,
            ),
            FunctionTool(
                "merge_skills",
                "Merge overlapping active Skills into one target using complete resulting content.",
                _MergeArgs,
                self.merge,
            ),
            FunctionTool(
                "retire_skill",
                "Retire a redundant Skill in favor of another active replacement.",
                _RetireArgs,
                self.retire,
            ),
            FunctionTool(
                "finish_skill_update",
                "Validate and finish the staged library; call with no edits for a deliberate no-op.",
                _FinishArgs,
                self.finish,
                terminal=True,
            ),
        ]


__all__ = [
    "DEFAULT_MAX_SKILL_CHARACTERS",
    "SkillCapacityError",
    "SkillLengthError",
    "SkillWorkspace",
]

"""Staged executor for converting one Episode batch into Insight operations."""

from __future__ import annotations

from typing import Annotated, Any, Literal

from pydantic import Field

from agent.tooling import FunctionTool, ToolArguments


class _AddInsightArgs(ToolArguments):
    text: str = Field(min_length=1)
    episode_refs: list[str] = Field(min_length=1)


class _ExistingInsightArgs(ToolArguments):
    insight_ref: str = Field(min_length=1)
    episode_refs: list[str] = Field(min_length=1)


class _ReviseInsightArgs(_ExistingInsightArgs):
    text: str = Field(min_length=1)


class _ContradictInsightArgs(_ExistingInsightArgs):
    reason: str = Field(min_length=1)


class _FinishArgs(ToolArguments):
    pass


class _AddOperation(ToolArguments):
    op: Literal["ADD"]
    text: str = Field(min_length=1)
    episode_refs: list[str] = Field(min_length=1)


class _SupportOperation(ToolArguments):
    op: Literal["SUPPORT"]
    insight_ref: str = Field(min_length=1)
    episode_refs: list[str] = Field(min_length=1)


class _ReviseOperation(_SupportOperation):
    op: Literal["REVISE"]
    text: str = Field(min_length=1)


class _ContradictOperation(_SupportOperation):
    op: Literal["CONTRADICT"]
    reason: str = Field(min_length=1)


class _MergeOperation(ToolArguments):
    op: Literal["MERGE"]
    target_insight_ref: str = Field(min_length=1)
    merged_insight_refs: list[str] = Field(min_length=1)
    text: str = Field(min_length=1)
    episode_refs: list[str] = Field(min_length=1)


class _ArchiveOperation(_SupportOperation):
    op: Literal["ARCHIVE"]
    reason: str = Field(min_length=1)
    replacement_insight_ref: str | None = None


_InsightOperation = Annotated[
    _AddOperation
    | _SupportOperation
    | _ReviseOperation
    | _ContradictOperation
    | _MergeOperation
    | _ArchiveOperation,
    Field(discriminator="op"),
]


class _ApplyInsightOperationsArgs(ToolArguments):
    operations: list[_InsightOperation] = Field(default_factory=list)


class InsightWorkspace:
    """Validate model actions against frozen local references before persistence."""

    def __init__(
        self,
        *,
        episode_id_by_ref: dict[str, str],
        insight_id_by_ref: dict[str, str],
    ) -> None:
        self.episode_id_by_ref = dict(episode_id_by_ref)
        self.insight_id_by_ref = dict(insight_id_by_ref)
        self._operations: list[dict[str, Any]] = []
        self._directions: dict[tuple[str, str], str] = {}
        self._revised: set[str] = set()
        self._added_texts: set[str] = set()
        self._archived: set[str] = set()
        self.finished = False

    @staticmethod
    def _text_key(value: str) -> str:
        return " ".join(str(value).casefold().split())

    def _episode_ids(self, references: list[str]) -> list[str]:
        refs = list(dict.fromkeys(str(item) for item in references))
        unknown = [item for item in refs if item not in self.episode_id_by_ref]
        if unknown:
            raise ValueError(f"unknown episode_refs: {unknown}")
        return [self.episode_id_by_ref[item] for item in refs]

    def _insight_id(self, reference: str) -> str:
        try:
            return self.insight_id_by_ref[str(reference)]
        except KeyError as error:
            raise ValueError(f"unknown insight_ref: {reference}") from error

    def _active_insight_id(self, reference: str) -> str:
        insight_id = self._insight_id(reference)
        if insight_id in self._archived:
            raise ValueError("an archived Insight cannot be edited again in this batch")
        return insight_id

    def _check_direction(
        self,
        insight_id: str,
        episode_ids: list[str],
        direction: str,
    ) -> None:
        for episode_id in episode_ids:
            previous = self._directions.get((insight_id, episode_id))
            if previous is not None and previous != direction:
                raise ValueError(
                    "one episode cannot both support and contradict one insight"
                )

    def _record_direction(
        self,
        insight_id: str,
        episode_ids: list[str],
        direction: str,
    ) -> None:
        for episode_id in episode_ids:
            self._directions[(insight_id, episode_id)] = direction

    def _remove_staged_support(
        self,
        insight_id: str,
        episode_ids: list[str],
    ) -> None:
        """Let a later REVISE carry evidence already staged as bare SUPPORT."""
        cited = set(episode_ids)
        retained = []
        for operation in self._operations:
            if (
                operation.get("op") == "SUPPORT"
                and operation.get("insight_id") == insight_id
            ):
                remaining = [
                    episode_id
                    for episode_id in operation.get("episode_ids", [])
                    if episode_id not in cited
                ]
                if remaining:
                    retained.append({**operation, "episode_ids": remaining})
            else:
                retained.append(operation)
        self._operations = retained

    def add(self, arguments: _AddInsightArgs) -> dict[str, Any]:
        episode_ids = self._episode_ids(arguments.episode_refs)
        text = arguments.text.strip()
        text_key = self._text_key(text)
        if text_key in self._added_texts:
            return {"staged": False, "duplicate": True}
        operation = {
            "op": "ADD",
            "text": text,
            "episode_ids": episode_ids,
        }
        self._operations.append(operation)
        self._added_texts.add(text_key)
        return {"staged": True, "operation_index": len(self._operations) - 1}

    def support(self, arguments: _ExistingInsightArgs) -> dict[str, Any]:
        insight_id = self._active_insight_id(arguments.insight_ref)
        episode_ids = self._episode_ids(arguments.episode_refs)
        self._check_direction(insight_id, episode_ids, "support")
        new_episode_ids = [
            episode_id
            for episode_id in episode_ids
            if self._directions.get((insight_id, episode_id)) != "support"
        ]
        if not new_episode_ids:
            return {"staged": False, "duplicate": True}
        operation = {
            "op": "SUPPORT",
            "insight_id": insight_id,
            "episode_ids": new_episode_ids,
        }
        self._operations.append(operation)
        self._record_direction(insight_id, new_episode_ids, "support")
        return {"staged": True, "operation_index": len(self._operations) - 1}

    def revise(self, arguments: _ReviseInsightArgs) -> dict[str, Any]:
        insight_id = self._active_insight_id(arguments.insight_ref)
        if insight_id in self._revised:
            raise ValueError("an insight may be revised at most once per batch")
        episode_ids = self._episode_ids(arguments.episode_refs)
        self._check_direction(insight_id, episode_ids, "support")
        self._remove_staged_support(insight_id, episode_ids)
        operation = {
            "op": "REVISE",
            "insight_id": insight_id,
            "text": arguments.text.strip(),
            "episode_ids": episode_ids,
        }
        self._operations.append(operation)
        self._revised.add(insight_id)
        self._record_direction(insight_id, episode_ids, "support")
        return {"staged": True, "operation_index": len(self._operations) - 1}

    def contradict(self, arguments: _ContradictInsightArgs) -> dict[str, Any]:
        insight_id = self._active_insight_id(arguments.insight_ref)
        episode_ids = self._episode_ids(arguments.episode_refs)
        self._check_direction(insight_id, episode_ids, "contradict")
        new_episode_ids = [
            episode_id
            for episode_id in episode_ids
            if self._directions.get((insight_id, episode_id)) != "contradict"
        ]
        if not new_episode_ids:
            return {"staged": False, "duplicate": True}
        operation = {
            "op": "CONTRADICT",
            "insight_id": insight_id,
            "episode_ids": new_episode_ids,
            "reason": arguments.reason.strip(),
        }
        self._operations.append(operation)
        self._record_direction(insight_id, new_episode_ids, "contradict")
        return {"staged": True, "operation_index": len(self._operations) - 1}

    def merge(self, arguments: _MergeOperation) -> dict[str, Any]:
        target_id = self._active_insight_id(arguments.target_insight_ref)
        if target_id in self._revised:
            raise ValueError("an insight may be revised or merged at most once per batch")
        merged_ids = list(
            dict.fromkeys(
                self._active_insight_id(item)
                for item in arguments.merged_insight_refs
            )
        )
        if target_id in merged_ids:
            raise ValueError("MERGE target cannot also be a source")
        episode_ids = self._episode_ids(arguments.episode_refs)
        self._check_direction(target_id, episode_ids, "support")
        operation = {
            "op": "MERGE",
            "target_insight_id": target_id,
            "merged_insight_ids": merged_ids,
            "text": arguments.text.strip(),
            "episode_ids": episode_ids,
        }
        self._operations.append(operation)
        self._archived.update(merged_ids)
        self._revised.add(target_id)
        self._record_direction(target_id, episode_ids, "support")
        return {"staged": True, "operation_index": len(self._operations) - 1}

    def archive(self, arguments: _ArchiveOperation) -> dict[str, Any]:
        insight_id = self._active_insight_id(arguments.insight_ref)
        replacement_id = (
            self._active_insight_id(arguments.replacement_insight_ref)
            if arguments.replacement_insight_ref
            else None
        )
        if replacement_id == insight_id:
            raise ValueError("ARCHIVE replacement must be a different Insight")
        operation = {
            "op": "ARCHIVE",
            "insight_id": insight_id,
            "episode_ids": self._episode_ids(arguments.episode_refs),
            "reason": arguments.reason.strip(),
            **(
                {"replacement_insight_id": replacement_id}
                if replacement_id is not None
                else {}
            ),
        }
        self._operations.append(operation)
        self._archived.add(insight_id)
        return {"staged": True, "operation_index": len(self._operations) - 1}

    def finish(self, _: _FinishArgs) -> dict[str, Any]:
        self.finished = True
        return {"ready_to_commit": True, "operation_count": len(self._operations)}

    def operations(self) -> list[dict[str, Any]]:
        return [dict(item) for item in self._operations]

    def apply_plan(
        self,
        arguments: _ApplyInsightOperationsArgs,
    ) -> dict[str, Any]:
        """Apply one model-authored operation list serially to this private workspace."""
        before_operations = list(self._operations)
        before_directions = dict(self._directions)
        before_revised = set(self._revised)
        before_added_texts = set(self._added_texts)
        before_archived = set(self._archived)
        before_finished = self.finished
        results: list[dict[str, Any]] = []
        try:
            for operation in arguments.operations:
                if isinstance(operation, _AddOperation):
                    result = self.add(
                        _AddInsightArgs(
                            text=operation.text,
                            episode_refs=operation.episode_refs,
                        )
                    )
                elif isinstance(operation, _ReviseOperation):
                    result = self.revise(
                        _ReviseInsightArgs(
                            insight_ref=operation.insight_ref,
                            episode_refs=operation.episode_refs,
                            text=operation.text,
                        )
                    )
                elif isinstance(operation, _ContradictOperation):
                    result = self.contradict(
                        _ContradictInsightArgs(
                            insight_ref=operation.insight_ref,
                            episode_refs=operation.episode_refs,
                            reason=operation.reason,
                        )
                    )
                elif isinstance(operation, _MergeOperation):
                    result = self.merge(operation)
                elif isinstance(operation, _ArchiveOperation):
                    result = self.archive(operation)
                else:
                    result = self.support(
                        _ExistingInsightArgs(
                            insight_ref=operation.insight_ref,
                            episode_refs=operation.episode_refs,
                        )
                    )
                results.append({"op": operation.op, **result})
            self.finished = True
        except ValueError:
            self._operations = before_operations
            self._directions = before_directions
            self._revised = before_revised
            self._added_texts = before_added_texts
            self._archived = before_archived
            self.finished = before_finished
            raise
        return {
            "ready_to_commit": True,
            "operation_count": len(self._operations),
            "operation_results": results,
        }

    def apply_serialized_plan(self, value: dict[str, Any]) -> dict[str, Any]:
        """Validate and execute a JSON operation chain produced without tool calling."""
        arguments = _ApplyInsightOperationsArgs.model_validate(value)
        return self.apply_plan(arguments)

    def plan_tool(self) -> FunctionTool:
        return FunctionTool(
            "apply_insight_operations",
            (
                "Apply the complete ordered Insight update once. Operations execute "
                "serially; use an empty list for a deliberate no-op."
            ),
            _ApplyInsightOperationsArgs,
            self.apply_plan,
        )

    def tools(self) -> list[FunctionTool]:
        return [
            FunctionTool(
                "add_insight",
                "Stage a new reusable situational refinement Insight supported by Episodes.",
                _AddInsightArgs,
                self.add,
            ),
            FunctionTool(
                "support_insight",
                "Attach supporting Episodes to an existing Insight without changing its text.",
                _ExistingInsightArgs,
                self.support,
            ),
            FunctionTool(
                "revise_insight",
                "Refine the scope or boundary of an existing Insight and attach support.",
                _ReviseInsightArgs,
                self.revise,
            ),
            FunctionTool(
                "contradict_insight",
                "Record Episode evidence that opposes an existing Insight.",
                _ContradictInsightArgs,
                self.contradict,
            ),
            FunctionTool(
                "finish_insight_update",
                "Finish after all useful Insight operations have been staged; zero operations is a valid no-op.",
                _FinishArgs,
                self.finish,
                terminal=True,
            ),
        ]


__all__ = ["InsightWorkspace"]

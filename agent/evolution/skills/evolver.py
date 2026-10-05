"""Evolve a high-level Skill library when new evidence reaches a threshold."""

from __future__ import annotations

import copy
import json
import re
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

from agent.evolution.management_call import (
    ManagementCallError,
    run_management_call,
)
from agent.evolution.memory.insights import InsightStore
from agent.evolution.skills.document import SkillDocument
from agent.evolution.skills.history import describe_changes, recent_history
from agent.evolution.skills.workspace import (
    DEFAULT_MAX_SKILL_CHARACTERS,
    SkillCapacityError,
    SkillLengthError,
    SkillWorkspace,
)
from agent.evolution.soft_constraints import warn_soft_limit
from agent.evolution.skills.registry import (
    PROPOSED_SKILL_TOKEN,
    SkillRegistry,
    _skill_id,
)
from agent.evolution.storage import utc_now
from agent.evolution.tracing import EvolutionTraceStore
from agent.prompts import PromptManager


@dataclass(frozen=True)
class _EvidenceProjection:
    records: list[dict[str, Any]]
    prompt_candidates: list[dict[str, Any]]
    visible_new: list[dict[str, Any]]
    all_new_ids: set[str]
    insight_id_by_ref: dict[str, str]


class SkillEvolver:
    def __init__(
        self,
        think: Callable[[str], str],
        *,
        min_support: int = 4,
        min_batches: int = 3,
        min_mean_reward: float = 0.5,
        min_mature_insights: int = 5,
        prompt_manager: PromptManager | None = None,
        trace_store: EvolutionTraceStore | None = None,
        serialized_operations: bool = False,
        max_attempts: int = 3,
        max_skill_characters: int = DEFAULT_MAX_SKILL_CHARACTERS,
        min_batches_between_evolutions: int = 0,
    ) -> None:
        self.think = think
        self.min_support = int(min_support)
        self.min_batches = int(min_batches)
        self.min_mean_reward = float(min_mean_reward)
        self.min_mature_insights = int(min_mature_insights)
        self.prompt_manager = prompt_manager or PromptManager()
        self.trace_store = trace_store
        self.serialized_operations = bool(serialized_operations)
        self.max_attempts = int(max_attempts)
        self.max_skill_characters = int(max_skill_characters)
        self.min_batches_between_evolutions = int(min_batches_between_evolutions)
        if min(
            self.min_support,
            self.min_batches,
            self.min_mature_insights,
            self.max_attempts,
            self.max_skill_characters,
        ) <= 0:
            raise ValueError("skill evolution thresholds and attempts must be positive")
        if not 0.0 <= self.min_mean_reward <= 1.0:
            raise ValueError("min_mean_reward must be between 0 and 1")
        if self.min_batches_between_evolutions < 0:
            raise ValueError("min_batches_between_evolutions must be non-negative")

    def process_batch(
        self,
        *,
        batch_id: int,
        insights: InsightStore,
        registry: SkillRegistry,
    ) -> list[dict[str, Any]]:
        """Evolve when enough unreviewed mature Insights are ready."""
        batch_id = int(batch_id)
        if batch_id <= 0:
            return []
        cycle_key = f"{registry.state.round_id}:{batch_id}"
        existing = registry.state.read_record("skill_cycles", cycle_key)
        if existing is not None:
            change = existing.get("change") or existing.get("candidate")
            return [dict(change)] if isinstance(change, dict) else []

        mature = insights.mature(
            min_support=self.min_support,
            min_batches=self.min_batches,
            min_mean_reward=self.min_mean_reward,
        )
        reviewed = registry.state.read_data("skill_reviewed_insights", {})
        skills = registry.summaries()
        projection = self._project_evidence(
            mature=mature,
            skills=skills,
            reviewed=reviewed,
        )
        pending_mature_count = len(projection.all_new_ids)
        if pending_mature_count < self.min_mature_insights:
            self._record_cycle(
                registry,
                cycle_key,
                batch_id=batch_id,
                outcome="waiting_for_skill_evidence",
                considered=projection.visible_new,
                pending_mature_count=pending_mature_count,
                trigger_mature_insights=self.min_mature_insights,
            )
            return []

        last_evolution_batch = self._last_evolution_batch(registry)
        if (
            last_evolution_batch is not None
            and batch_id - last_evolution_batch
            < self.min_batches_between_evolutions
        ):
            self._record_cycle(
                registry,
                cycle_key,
                batch_id=batch_id,
                outcome="waiting_for_evolution_cooldown",
                considered=projection.visible_new,
                pending_mature_count=pending_mature_count,
                trigger_mature_insights=self.min_mature_insights,
                last_evolution_batch=last_evolution_batch,
                min_batches_between_evolutions=self.min_batches_between_evolutions,
            )
            return []

        visible_pending_ids = {
            str(item["insight_id"]) for item in projection.visible_new
        }

        if self.serialized_operations:
            return self._process_serialized_operations(
                batch_id=batch_id,
                cycle_key=cycle_key,
                registry=registry,
                skills=skills,
                projection=projection,
                visible_pending_ids=visible_pending_ids,
            )

        prompt = self.prompt_manager.skill_evolution(
            insight_candidates=projection.prompt_candidates,
            skills=self._prompt_skills(skills),
            history=recent_history(registry),
            max_skill_count=registry.max_skills,
            max_skill_characters=self.max_skill_characters,
        )
        trace_id = (
            self.trace_store.start(
                stage="skill_evolution",
                prompt=prompt,
                batch_id=batch_id,
            )
            if self.trace_store is not None
            else None
        )
        try:
            response = self.think(prompt).strip()
        except BaseException as error:
            if trace_id is not None:
                self.trace_store.fail(trace_id, error)
            raise
        value = self._json_object(response)
        decision = self._decision(
            value,
            insight_id_by_ref=projection.insight_id_by_ref,
            required_source_insight_ids=visible_pending_ids,
            registry=registry,
        )
        proposal = None
        if decision is not None:
            if decision["operation"] == "NOOP":
                proposal = decision
            elif isinstance((value or {}).get("skill"), dict):
                # Compatibility for older persisted structured responses.
                proposal = self._proposal(
                    value,
                    insight_id_by_ref=projection.insight_id_by_ref,
                    required_source_insight_ids=visible_pending_ids,
                    registry=registry,
                )
            else:
                proposal = self._proposal_from_text(
                    decision,
                    skill_text=str((value or {}).get("content") or ""),
                    registry=registry,
                )
        if trace_id is not None:
            self.trace_store.complete(
                trace_id,
                raw_response=response,
                parsed_response=value,
                normalized_output=proposal,
            )
        if decision is None or proposal is None:
            self._record_cycle(
                registry,
                cycle_key,
                batch_id=batch_id,
                outcome=(
                    "invalid_proposal"
                    if decision is None
                    else "invalid_skill_text"
                ),
                considered=projection.records,
            )
            return []

        if proposal["operation"] == "NOOP":
            # A valid NOOP is still a completed review. Advance the Insight
            # cursors so unchanged mature evidence does not immediately
            # retrigger the manager on the next batch. If an Insight changes
            # later, its newer updated_batch will make it visible again.
            self._mark_reviewed(registry, projection.visible_new)
            self._record_cycle(
                registry,
                cycle_key,
                batch_id=batch_id,
                outcome="noop",
                considered=projection.records,
            )
            return []

        if proposal["operation"] == "CREATE" and not registry.can_create():
            proposed_source_ids = set(proposal["source_insight_ids"])
            proposed_skill_insights = [
                item
                for item in mature
                if str(item["insight_id"]) in proposed_source_ids
            ]
            prompt = self.prompt_manager.skill_reorganization(
                max_skill_count=int(registry.max_skills),
                active_skills=self._prompt_skills(skills),
                proposed_skill=self._proposed_skill_context(proposal),
                proposed_skill_insights=[
                    self._prompt_insight(item) for item in proposed_skill_insights
                ],
            )
            trace_id = (
                self.trace_store.start(
                    stage="skill_reorganization",
                    prompt=prompt,
                    batch_id=batch_id,
                )
                if self.trace_store is not None
                else None
            )
            try:
                response = self.think(prompt).strip()
            except BaseException as error:
                if trace_id is not None:
                    self.trace_store.fail(trace_id, error)
                raise
            parsed_reorganization = self._json_object(response)
            plan = self._reorganization_plan(
                parsed_reorganization,
                active_skills=skills,
                proposed_skill=proposal,
                max_skill_count=int(registry.max_skills),
                registry=registry,
            )
            if trace_id is not None:
                self.trace_store.complete(
                    trace_id,
                    raw_response=response,
                    parsed_response=parsed_reorganization,
                    normalized_output=plan,
                )
            if plan is None:
                self._record_cycle(
                    registry,
                    cycle_key,
                    batch_id=batch_id,
                    outcome="invalid_reorganization",
                    considered=projection.records,
                )
                return []
            if plan["proposed_skill_disposition"] == "DISCARD":
                self._mark_reviewed(registry, projection.visible_new)
                self._record_cycle(
                    registry,
                    cycle_key,
                    batch_id=batch_id,
                    outcome="reorganization_noop",
                    considered=projection.records,
                )
                return []
            change = registry.apply_reorganization(
                plan=plan,
                proposed_skill=proposal,
                batch_id=batch_id,
            )
        else:
            change = registry.apply_evolution(
                proposal=proposal,
                batch_id=batch_id,
                evidence_ids=list(proposal["source_insight_ids"]),
            )
        if change is None:
            self._record_cycle(
                registry,
                cycle_key,
                batch_id=batch_id,
                outcome="change_rejected",
                considered=projection.records,
            )
            return []

        # Provenance remains limited to the Insight IDs used by the applied
        # proposal, but every newly visible candidate has now been reviewed
        # and must not keep contributing to the accumulation trigger.
        self._mark_reviewed(registry, projection.visible_new)
        self._record_cycle(
            registry,
            cycle_key,
            batch_id=batch_id,
            outcome="change_applied",
            considered=projection.records,
            change=change,
        )
        return [change]

    def _process_serialized_operations(
        self,
        *,
        batch_id: int,
        cycle_key: str,
        registry: SkillRegistry,
        skills: list[dict[str, Any]],
        projection: _EvidenceProjection,
        visible_pending_ids: set[str],
    ) -> list[dict[str, Any]]:
        """Generate a JSON edit chain and apply it as one ordered transaction."""
        active_skills = registry.runtime_skills()
        reserved_ids = set(registry.files.known_skill_ids())
        prompt = self.prompt_manager.skill_evolution(
            insight_candidates=projection.prompt_candidates,
            skills=self._prompt_skill_documents(skills, active_skills),
            history=recent_history(registry),
            max_skill_count=registry.max_skills,
            max_skill_characters=self.max_skill_characters,
        )
        trace_id = (
            self.trace_store.start(
                stage="skill_evolution",
                prompt=prompt,
                batch_id=batch_id,
            )
            if self.trace_store is not None
            else None
        )
        successful_workspace: dict[str, SkillWorkspace] = {}

        def execute(value: dict[str, Any]) -> dict[str, Any]:
            workspace = SkillWorkspace(
                active_skills=active_skills,
                reserved_skill_ids=reserved_ids,
                insight_id_by_ref=projection.insight_id_by_ref,
                required_source_insight_ids=visible_pending_ids,
                # Capacity is checked after the primary Skill Manager has completed.
                max_skills=None,
            )
            workspace.apply_serialized_plan(value)
            successful_workspace["value"] = workspace
            return workspace.plan()

        try:
            result = run_management_call(
                think=self.think,
                prompt=prompt,
                max_attempts=self.max_attempts,
                stage_name="Skill",
                parse=self._json_object,
                execute=execute,
                retry_prompt=self._skill_retry_prompt,
            )
        except ManagementCallError as failure:
            if trace_id is not None:
                self.trace_store.fail(
                    trace_id,
                    failure.cause,
                    raw_response=failure.raw_response,
                    parsed_response=failure.parsed_response,
                    model_attempts=failure.attempts,
                )
            raise failure.cause from failure
        primary_plan = result.normalized_output
        workspace = successful_workspace["value"]

        if trace_id is not None:
            self.trace_store.complete(
                trace_id,
                raw_response=result.raw_response,
                parsed_response=result.parsed_response,
                normalized_output=primary_plan,
                model_attempts=result.attempts,
            )

        # Maintenance reorganizations operate on the already staged result but still
        # commit with the primary decision as one atomic transaction. They do not absorb
        # additional Insight evidence.
        workspace.required_source_insight_ids = set()
        staged_skills = workspace.active_skill_snapshot()
        if (
            registry.max_skills is not None
            and len(staged_skills) > int(registry.max_skills)
        ):
            capacity_prompt = self.prompt_manager.skill_capacity_reorganization(
                max_skill_count=int(registry.max_skills),
                skills=staged_skills,
            )

            def prepare_capacity(candidate: SkillWorkspace) -> None:
                candidate.max_skills = int(registry.max_skills)

            workspace = self._run_serialized_maintenance(
                stage="skill_capacity_reorganization",
                stage_name="Skill capacity reorganization",
                prompt=capacity_prompt,
                batch_id=batch_id,
                workspace=workspace,
                prepare=prepare_capacity,
                validate=lambda _: None,
            )

        workspace.max_skills = (
            int(registry.max_skills) if registry.max_skills is not None else None
        )
        staged_skills = workspace.active_skill_snapshot()
        if workspace.over_limit_skills(self.max_skill_characters):
            length_prompt = self.prompt_manager.skill_length_reorganization(
                max_skill_count=registry.max_skills,
                max_skill_characters=self.max_skill_characters,
                skills=staged_skills,
            )
            workspace = self._run_serialized_maintenance(
                stage="skill_length_reorganization",
                stage_name="Skill length reorganization",
                prompt=length_prompt,
                batch_id=batch_id,
                workspace=workspace,
                prepare=lambda candidate: setattr(
                    candidate,
                    "max_skills",
                    (
                        int(registry.max_skills)
                        if registry.max_skills is not None
                        else None
                    ),
                ),
                validate=lambda candidate: candidate.validate_skill_lengths(
                    self.max_skill_characters
                ),
            )

        plan = workspace.plan()

        if not plan["result_skills"] and not plan["retired_skills"]:
            # Empty operations are an explicit, valid manager decision. Mark
            # all newly visible candidates reviewed so the same evidence does
            # not cause a NOOP loop on every subsequent batch.
            self._mark_reviewed(registry, projection.visible_new)
            self._record_cycle(
                registry,
                cycle_key,
                batch_id=batch_id,
                outcome="noop",
                considered=projection.records,
            )
            return []

        change = registry.apply_tool_transaction(plan=plan, batch_id=batch_id)
        if change is None:
            self._record_cycle(
                registry,
                cycle_key,
                batch_id=batch_id,
                outcome="change_rejected",
                considered=projection.records,
            )
            return []

        # Skill provenance is written from result_skills by the registry.
        # Review cursors have a different purpose: all candidates seen in a
        # successful decision are acknowledged, including candidates the
        # manager intentionally chose not to absorb.
        self._mark_reviewed(registry, projection.visible_new)
        self._record_cycle(
            registry,
            cycle_key,
            batch_id=batch_id,
            outcome="change_applied",
            considered=projection.records,
            change=change,
        )
        return [change]

    def _run_serialized_maintenance(
        self,
        *,
        stage: str,
        stage_name: str,
        prompt: str,
        batch_id: int,
        workspace: SkillWorkspace,
        prepare: Callable[[SkillWorkspace], None],
        validate: Callable[[SkillWorkspace], None],
    ) -> SkillWorkspace:
        """Apply one maintenance decision to a cloned staged workspace."""
        trace_id = (
            self.trace_store.start(stage=stage, prompt=prompt, batch_id=batch_id)
            if self.trace_store is not None
            else None
        )
        successful_workspace: dict[str, SkillWorkspace] = {}

        def execute(value: dict[str, Any]) -> dict[str, Any]:
            operations = value.get("operations")
            if isinstance(operations, list) and any(
                isinstance(operation, dict)
                and bool(operation.get("source_insight_refs"))
                for operation in operations
            ):
                raise ValueError(
                    "maintenance operations must use empty source_insight_refs"
                )
            candidate = copy.deepcopy(workspace)
            candidate.required_source_insight_ids = set()
            prepare(candidate)
            candidate.apply_serialized_plan(value)
            validate(candidate)
            successful_workspace["value"] = candidate
            return candidate.plan()

        try:
            result = run_management_call(
                think=self.think,
                prompt=prompt,
                max_attempts=self.max_attempts,
                stage_name=stage_name,
                parse=self._json_object,
                execute=execute,
                retry_prompt=self._skill_maintenance_retry_prompt,
            )
        except ManagementCallError as failure:
            if trace_id is not None:
                self.trace_store.fail(
                    trace_id,
                    failure.cause,
                    raw_response=failure.raw_response,
                    parsed_response=failure.parsed_response,
                    model_attempts=failure.attempts,
                )
            raise failure.cause from failure

        if trace_id is not None:
            self.trace_store.complete(
                trace_id,
                raw_response=result.raw_response,
                parsed_response=result.parsed_response,
                normalized_output=result.normalized_output,
                model_attempts=result.attempts,
            )
        return successful_workspace["value"]

    @staticmethod
    def _skill_retry_prompt(
        original_prompt: str,
        error: Exception,
        _: int,
    ) -> str:
        """Return the original format-bearing prompt with actionable retry feedback."""
        if not isinstance(error, SkillCapacityError):
            return (
                f"{original_prompt}\n\n"
                "RETRY FEEDBACK:\n"
                f"- The previous operation plan failed validation: {error}.\n"
                "- Return a complete corrected JSON object that follows the JSON "
                "value shapes above."
            )
        return (
            f"{original_prompt}\n\n"
            "RETRY FEEDBACK:\n"
            f"- The previous operation plan failed with {type(error).__name__}: "
            f"{error}.\n"
            f"- The previous operation plan ended with {error.final_count} active "
            f"Skills, exceeding the maximum of {error.max_skills}.\n"
            "- Return a complete revised operation plan whose final active library "
            f"contains at most {error.max_skills} Skills. Balance any CREATE or SPLIT "
            "with MERGE or RETIRE operations, or omit it when it is not needed."
        )

    @staticmethod
    def _skill_maintenance_retry_prompt(
        original_prompt: str,
        error: Exception,
        _: int,
    ) -> str:
        details = [
            f"- The previous maintenance operation plan failed: "
            f"{type(error).__name__}: {error}."
        ]
        if isinstance(error, SkillCapacityError):
            details.append(
                f"- Its final library contained {error.final_count} active Skills; "
                f"the maximum is {error.max_skills}."
            )
        elif isinstance(error, SkillLengthError):
            details.append(
                f"- Every active Skill must be at most {error.max_characters} "
                "characters."
            )
        details.append(
            "- Return one complete corrected JSON object using the operation shapes above."
        )
        return f"{original_prompt}\n\nRETRY FEEDBACK:\n" + "\n".join(details)

    @classmethod
    def _project_evidence(
        cls,
        *,
        mature: list[dict[str, Any]],
        skills: list[dict[str, Any]],
        reviewed: dict[str, Any],
    ) -> _EvidenceProjection:
        """Derive the complete LLM view from review cursors and active provenance."""
        absorbed_ids = {
            str(insight_id)
            for skill in skills
            for insight_id in skill.get("source_insight_ids", [])
        }
        all_new_ids = {
            str(item["insight_id"])
            for item in mature
            if int(item.get("updated_batch", 0))
            > int(reviewed.get(str(item["insight_id"]), -1))
        }
        ordered = [
            *(item for item in mature if str(item["insight_id"]) in all_new_ids),
            *(
                item
                for item in mature
                if str(item["insight_id"]) not in all_new_ids
                and str(item["insight_id"]) not in absorbed_ids
            ),
        ]
        prompt_candidates = []
        insight_id_by_ref = {}
        for index, item in enumerate(ordered, 1):
            insight_id = str(item["insight_id"])
            insight_ref = f"i{index}"
            insight_id_by_ref[insight_ref] = insight_id
            prompt_candidates.append(
                {
                    "insight_ref": insight_ref,
                    "state": (
                        "new_or_changed"
                        if insight_id in all_new_ids
                        else "reviewed_unabsorbed"
                    ),
                    **cls._prompt_insight(item),
                }
            )
        return _EvidenceProjection(
            records=ordered,
            prompt_candidates=prompt_candidates,
            visible_new=[
                item
                for item in ordered
                if str(item["insight_id"]) in all_new_ids
            ],
            all_new_ids=all_new_ids,
            insight_id_by_ref=insight_id_by_ref,
        )

    @staticmethod
    def _prompt_insight(item: dict[str, Any]) -> dict[str, Any]:
        value = {
            "text": str(item.get("text") or ""),
            "support_count": int(item.get("support_count", 0)),
            "contradiction_count": int(item.get("contradiction_count", 0)),
            "supporting_batch_count": len(item.get("support_batches", [])),
        }
        if "mean_reward" in item:
            value["mean_reward"] = float(item["mean_reward"])
        return value

    @staticmethod
    def _prompt_skills(skills: list[dict[str, Any]]) -> list[dict[str, Any]]:
        return [
            {
                "skill_id": str(skill["skill_id"]),
                "title": str(skill.get("title") or ""),
                "description": str(skill.get("description") or ""),
                "instructions": str(skill.get("instructions") or ""),
                "source_insight_count": len(skill.get("source_insight_ids", [])),
            }
            for skill in skills
        ]

    @staticmethod
    def _prompt_skill_manifests(
        skills: list[dict[str, Any]],
    ) -> list[dict[str, Any]]:
        """Produce compact manifests for legacy compatibility callers."""
        return [
            {
                key: value
                for key, value in item.items()
                if key != "instructions"
            }
            for item in SkillEvolver._prompt_skills(skills)
        ]

    @staticmethod
    def _prompt_skill_documents(
        skills: list[dict[str, Any]],
        active_skills: list[dict[str, Any]],
    ) -> list[dict[str, Any]]:
        """Expose complete current text so one call can author exact local edits."""
        summaries = {
            str(item["skill_id"]): item
            for item in SkillEvolver._prompt_skills(skills)
        }
        projected = []
        for skill in active_skills:
            skill_id = str(skill["skill_id"])
            summary = dict(summaries.get(skill_id) or {})
            summary.pop("instructions", None)
            projected.append(
                {
                    **summary,
                    "skill_id": skill_id,
                    "content": str(skill.get("markdown") or ""),
                }
            )
        return projected

    @staticmethod
    def _mark_reviewed(
        registry: SkillRegistry,
        insights: list[dict[str, Any]],
    ) -> None:
        reviewed = registry.state.read_data("skill_reviewed_insights", {})
        for item in insights:
            reviewed[str(item["insight_id"])] = int(item.get("updated_batch", 0))
        registry.state.write_data("skill_reviewed_insights", reviewed)

    @staticmethod
    def _last_evolution_batch(registry: SkillRegistry) -> int | None:
        completed_outcomes = {"noop", "change_applied", "reorganization_noop"}
        batches = [
            int(item.get("batch_id", 0))
            for item in registry.state.records("skill_cycles")
            if item.get("outcome") in completed_outcomes
        ]
        return max(batches, default=None)

    @staticmethod
    def _record_cycle(
        registry: SkillRegistry,
        cycle_key: str,
        *,
        batch_id: int,
        outcome: str,
        considered: list[dict[str, Any]],
        change: dict[str, Any] | None = None,
        pending_mature_count: int | None = None,
        trigger_mature_insights: int | None = None,
        last_evolution_batch: int | None = None,
        min_batches_between_evolutions: int | None = None,
    ) -> None:
        value: dict[str, Any] = {
            "batch_id": int(batch_id),
            "round_id": registry.state.round_id,
            "outcome": outcome,
            "considered_insight_ids": [
                str(item["insight_id"]) for item in considered
            ],
            "created_at": utc_now(),
        }
        if change is not None:
            value["change"] = change
            value["history"] = describe_changes(registry, change, considered)
        if pending_mature_count is not None:
            value["pending_mature_count"] = int(pending_mature_count)
        if trigger_mature_insights is not None:
            value["trigger_mature_insights"] = int(trigger_mature_insights)
        if last_evolution_batch is not None:
            value["last_evolution_batch"] = int(last_evolution_batch)
        if min_batches_between_evolutions is not None:
            value["min_batches_between_evolutions"] = int(
                min_batches_between_evolutions
            )
        registry.state.create_record("skill_cycles", cycle_key, value)

    @staticmethod
    def _proposed_skill_context(proposal: dict[str, Any]) -> dict[str, Any]:
        return {
            "suggested_id": str(proposal["skill_id"]),
            "text": str(proposal["markdown"]),
        }

    @staticmethod
    def _reorganization_plan(
        value: dict[str, Any] | None,
        *,
        active_skills: list[dict[str, Any]],
        proposed_skill: dict[str, Any],
        max_skill_count: int,
        registry: SkillRegistry,
    ) -> dict[str, Any] | None:
        if value is None:
            return None
        disposition = str(value.get("proposed_skill_disposition") or "").upper()
        if disposition not in {"INTEGRATE", "DISTINCT", "DISCARD"}:
            return None

        active_by_id = {
            str(item["skill_id"]): item
            for item in active_skills
        }
        active_ids = set(active_by_id)
        raw_kept_ids = value.get("kept_skill_ids", [])
        if not isinstance(raw_kept_ids, list):
            return None
        kept_ids = [
            _skill_id(str(item)) for item in raw_kept_ids
        ]
        if any(not item for item in kept_ids) or len(kept_ids) != len(set(kept_ids)):
            return None

        retired_skills = []
        retired_ids = []
        raw_retired = value.get("retired_skills", [])
        if not isinstance(raw_retired, list):
            return None
        for item in raw_retired:
            if not isinstance(item, dict):
                return None
            skill_id = _skill_id(str(item.get("skill_id") or ""))
            replacement_id = _skill_id(
                str(item.get("replacement_skill_id") or "")
            )
            if not skill_id or not replacement_id or skill_id == replacement_id:
                return None
            retired_ids.append(skill_id)
            retired_skills.append(
                {
                    "skill_id": skill_id,
                    "replacement_skill_id": replacement_id,
                }
            )
        if len(retired_ids) != len(set(retired_ids)):
            return None

        raw_results = value.get("result_skills", [])
        if not isinstance(raw_results, list):
            return None
        results = []
        assigned_sources: list[str] = [*kept_ids, *retired_ids]
        proposed_result: dict[str, Any] | None = None
        target_ids = []
        proposed_refs = 0
        for item in raw_results:
            if not isinstance(item, dict):
                return None
            target_id = _skill_id(str(item.get("skill_id") or ""))
            if not target_id:
                return None
            raw_sources = item.get("source_skill_ids", [])
            if not isinstance(raw_sources, list):
                return None
            source_ids = []
            for raw_source in raw_sources:
                source = str(raw_source)
                normalized = (
                    PROPOSED_SKILL_TOKEN
                    if source == PROPOSED_SKILL_TOKEN
                    else _skill_id(source)
                )
                if not normalized:
                    return None
                source_ids.append(normalized)
            if not source_ids or len(source_ids) != len(set(source_ids)):
                return None
            if any(
                source != PROPOSED_SKILL_TOKEN and source not in active_ids
                for source in source_ids
            ):
                return None

            raw_skill = item.get("skill")
            if isinstance(raw_skill, dict):
                document = SkillDocument.from_structured(raw_skill)
                authored_as_text = False
            else:
                content = str(item.get("content") or "").strip()
                document = (
                    SkillDocument.from_markdown(content, target_id)
                    if content
                    else None
                )
                authored_as_text = True
            if document is None:
                return None
            document.warn_if_not_initial_only()
            markdown = document.to_markdown()
            if authored_as_text:
                warn_soft_limit(
                    f"Skill document size ({document.title})",
                    actual=len(markdown),
                    preferred_max=4_500,
                    unit="characters",
                )

            includes_proposed = PROPOSED_SKILL_TOKEN in source_ids
            if includes_proposed:
                proposed_refs += 1
            existing_sources = [
                source for source in source_ids if source != PROPOSED_SKILL_TOKEN
            ]
            assigned_sources.extend(existing_sources)
            if target_id in active_ids and target_id not in existing_sources:
                return None
            if (
                target_id not in active_ids
                and registry.files.load_manifest(target_id) is not None
            ):
                return None
            normalized_result = {
                "skill_id": target_id,
                "source_skill_ids": source_ids,
                "markdown": markdown,
            }
            results.append(normalized_result)
            target_ids.append(target_id)
            if includes_proposed:
                proposed_result = normalized_result

        if len(target_ids) != len(set(target_ids)):
            return None
        if len(assigned_sources) != len(set(assigned_sources)):
            return None
        if set(assigned_sources) != active_ids:
            return None
        if set(target_ids).intersection(set(kept_ids) | set(retired_ids)):
            return None
        resulting_ids = set(kept_ids) | set(target_ids)
        if not resulting_ids or len(resulting_ids) > int(max_skill_count):
            return None
        if any(
            item["replacement_skill_id"] not in resulting_ids
            for item in retired_skills
        ):
            return None

        if disposition == "DISCARD":
            if (
                proposed_refs
                or results
                or retired_skills
                or set(kept_ids) != active_ids
            ):
                return None
        elif proposed_refs != 1 or proposed_result is None:
            return None
        elif disposition == "INTEGRATE":
            if len(proposed_result["source_skill_ids"]) < 2:
                return None
        elif proposed_result["source_skill_ids"] != [PROPOSED_SKILL_TOKEN]:
            return None

        return {
            "operation": "REORGANIZE",
            "proposed_skill_disposition": disposition,
            "kept_skill_ids": kept_ids,
            "retired_skills": retired_skills,
            "result_skills": results,
            "expected_active_versions": {
                skill_id: int(item["version"])
                for skill_id, item in active_by_id.items()
            },
        }

    @staticmethod
    def _decision(
        value: dict[str, Any] | None,
        *,
        registry: SkillRegistry,
        insight_id_by_ref: dict[str, str] | None = None,
        allowed_insight_ids: set[str] | None = None,
        required_source_insight_ids: set[str] | None = None,
    ) -> dict[str, Any] | None:
        if value is None:
            return None
        operation = str(value.get("operation") or "").upper()
        if operation == "NOOP":
            return {"operation": "NOOP"}
        if operation not in {"CREATE", "MODIFY", "MERGE"}:
            return None
        insight_id_by_ref = dict(insight_id_by_ref or {})
        if "new_source_insight_refs" in value:
            raw_source_refs = value.get("new_source_insight_refs")
            if not isinstance(raw_source_refs, list):
                return None
            source_refs = list(dict.fromkeys(str(item) for item in raw_source_refs))
            if any(item not in insight_id_by_ref for item in source_refs):
                return None
            source_ids = [insight_id_by_ref[item] for item in source_refs]
        else:
            # Compatibility for persisted decisions and direct callers from the former
            # schema. Runtime prompts expose only batch-local references.
            raw_source_ids = value.get("source_insight_ids", [])
            if not isinstance(raw_source_ids, list):
                return None
            source_ids = list(dict.fromkeys(str(item) for item in raw_source_ids))
            allowed = set(allowed_insight_ids or insight_id_by_ref.values())
            if not set(source_ids) <= allowed:
                return None
        if operation == "CREATE" and not source_ids:
            return None
        required = set(required_source_insight_ids or [])
        if required and not required.intersection(source_ids):
            return None

        if operation == "CREATE":
            raw_skill = value.get("skill")
            suggested_id = value.get("suggested_id")
            if suggested_id is None and isinstance(raw_skill, dict):
                suggested_id = raw_skill.get("suggested_id")
            skill_id = _skill_id(str(suggested_id or ""))
            if not skill_id or registry.files.load_manifest(skill_id) is not None:
                return None
            merged_skill_ids = []
        elif operation == "MODIFY":
            skill_id = _skill_id(str(value.get("target_skill_id") or ""))
            current = registry.get(skill_id)
            if current is None or current.get("status") != "active":
                return None
            merged_skill_ids = []
        else:
            skill_id = _skill_id(str(value.get("target_skill_id") or ""))
            current = registry.get(skill_id)
            merged_skill_ids = list(
                dict.fromkeys(
                    _skill_id(str(item))
                    for item in value.get("merged_skill_ids", [])
                    if _skill_id(str(item))
                )
            )
            if (
                current is None
                or current.get("status") != "active"
                or not merged_skill_ids
                or skill_id in merged_skill_ids
                or any(registry.get(item) is None for item in merged_skill_ids)
            ):
                return None
        decision = {
            "operation": operation,
            "skill_id": skill_id,
            "source_insight_ids": source_ids,
        }
        if merged_skill_ids:
            decision["merged_skill_ids"] = merged_skill_ids
        return decision

    @staticmethod
    def _proposal_from_text(
        decision: dict[str, Any],
        *,
        skill_text: str,
        registry: SkillRegistry,
    ) -> dict[str, Any] | None:
        if decision.get("operation") not in {"CREATE", "MODIFY", "MERGE"}:
            return None
        skill_id = _skill_id(str(decision.get("skill_id") or ""))
        text = str(skill_text or "").strip()
        if not skill_id or not text:
            return None
        document = SkillDocument.from_markdown(text, skill_id)
        if not document.description or not document.instructions:
            return None
        document.warn_if_not_initial_only()
        markdown = document.to_markdown()
        warn_soft_limit(
            f"Skill document size ({document.title})",
            actual=len(markdown),
            preferred_max=4_500,
            unit="characters",
        )
        proposal = {
            **decision,
            "skill_id": skill_id,
            "markdown": markdown,
            "skill": {"text": text},
        }
        return proposal

    @staticmethod
    def _proposal(
        value: dict[str, Any] | None,
        *,
        registry: SkillRegistry,
        insight_id_by_ref: dict[str, str] | None = None,
        allowed_insight_ids: set[str] | None = None,
        required_source_insight_ids: set[str] | None = None,
    ) -> dict[str, Any] | None:
        """Compatibility parser for older persisted structured Skill decisions."""
        decision = SkillEvolver._decision(
            value,
            registry=registry,
            insight_id_by_ref=insight_id_by_ref,
            allowed_insight_ids=allowed_insight_ids,
            required_source_insight_ids=required_source_insight_ids,
        )
        if decision is None or decision["operation"] == "NOOP":
            return decision
        raw_skill = value.get("skill") if value is not None else None
        if not isinstance(raw_skill, dict):
            return None
        document = SkillDocument.from_structured(raw_skill)
        if document is None:
            return None
        document.warn_if_not_initial_only()
        proposal = {
            **decision,
            "markdown": document.to_markdown(),
            "skill": dict(raw_skill),
        }
        return proposal

    @staticmethod
    def _json_object(response: str) -> dict[str, Any] | None:
        try:
            value = json.loads(response)
        except json.JSONDecodeError:
            match = re.search(r"\{.*\}", response, re.DOTALL)
            if match is None:
                return None
            try:
                value = json.loads(match.group())
            except json.JSONDecodeError:
                return None
        return value if isinstance(value, dict) else None


__all__ = ["SkillEvolver"]

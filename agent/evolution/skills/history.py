"""Bounded learning views over persisted Skill changes and version-matched Episodes."""

from __future__ import annotations

import difflib
import json
from typing import Any

from agent.evolution.skills.registry import SkillRegistry


def describe_changes(
    registry: SkillRegistry, change: dict[str, Any], considered: list[dict[str, Any]]
) -> list[dict[str, Any]]:
    """Snapshot actual committed diffs and motivating evidence in the cycle record."""
    sources = {str(item["insight_id"]): item for item in considered}
    changes = change.get("changes", change.get("result_skills", [change]))
    history = []
    for item in changes:
        if "skill_id" not in item or "version" not in item:
            continue
        skill_id, version = str(item["skill_id"]), int(item["version"])
        directory = registry.files.version_dir(skill_id, version)
        meta = registry.state.load_json(directory / "meta.json", {})
        parent = meta.get("parent_version")
        before = (
            (registry.files.version_dir(skill_id, int(parent)) / "SKILL.md").read_text(encoding="utf-8")
            if parent is not None else ""
        )
        after = (directory / "SKILL.md").read_text(encoding="utf-8")
        history.append({
            "skill_id": skill_id,
            "version": version,
            "parent_version": parent,
            "operation": item.get("operation", change.get("operation")),
            "diff": "".join(difflib.unified_diff(
                before.splitlines(keepends=True), after.splitlines(keepends=True),
                fromfile=f"{skill_id}@{parent}", tofile=f"{skill_id}@{version}",
            )),
            # Preserve the evidence as it was at publication, not its later revision.
            # This is motivation, not a claim of experimentally verified benefit.
            "motivation_evidence": [
                {"insight_id": source_id, "text": str(sources[source_id].get("text", ""))}
                for source_id in item.get("source_insight_ids", []) if source_id in sources
            ],
            "retired_skills": change.get("retired_skills", []),
        })
    return history


def _excerpt(value: Any, limit: int) -> str:
    text = str(value)
    return text if len(text) <= limit else text[:limit] + " ... [truncated]"


def recent_history(registry: SkillRegistry) -> list[dict[str, Any]]:
    """Read only; old checkpoints without history remain valid and are not backfilled."""
    cycles = sorted(
        registry.state.records("skill_cycles"),
        key=lambda item: (int(item.get("round_id", 0)), int(item.get("batch_id", 0))),
        reverse=True,
    )
    entries = [(cycle, item) for cycle in cycles for item in cycle.get("history", [])][:6]
    if not entries:
        return []
    episodes = registry.state.records("episodes")
    result = []
    for cycle, item in entries:
        matching = [
            episode for episode in episodes
            if int((episode.get("task") or {}).get("round_id", 0)) == cycle["round_id"]
            and int(episode.get("batch_id", 0)) > cycle["batch_id"]
            and any(
                selection.get("skill_id") == item["skill_id"]
                and selection.get("version") == item["version"]
                for selection in (episode.get("memory_context") or {}).get("selected_skills", [])
            )
        ]
        matching.sort(key=lambda episode: (
            int(episode.get("batch_id", 0)), str(episode.get("episode_id", ""))
        ), reverse=True)
        # Show a recent failed and passed final internal check, when available.
        # Neither group establishes that this Skill caused the observed outcome.
        examples = []
        for failed in (True, False):
            example = next((episode for episode in matching if (
                bool((episode.get("outcome") or {}).get("failed")) if failed else
                bool((episode.get("outcome") or {}).get("passed"))
                and not (episode.get("outcome") or {}).get("failed")
            )), None)
            if example is None:
                continue
            outcome = example.get("outcome") or {}
            attempts = (example.get("evidence") or {}).get("attempts", [])
            first = min(attempts, key=lambda attempt: attempt.get("iteration", 0), default={})
            examples.append({
                "original_prompt": _excerpt((example.get("task") or {}).get("original_prompt", ""), 400),
                "first_internal_failed": [_excerpt(q, 160) for q in first.get("failed", [])[:3]],
                "final_internal_failed": [_excerpt(q, 160) for q in outcome.get("failed", [])[:3]],
                "final_internal_passed": [_excerpt(q, 160) for q in outcome.get("passed", [])[:3]],
                "final_reward": outcome.get("reward"),
            })
        projected = {
            "skill_id": item["skill_id"], "version": item["version"],
            "parent_version": item["parent_version"], "operation": item["operation"],
            "round_id": cycle["round_id"], "batch_id": cycle["batch_id"],
            "diff": _excerpt(item["diff"], 1600),
            "motivation_evidence": [
                _excerpt(source["text"], 400) for source in item["motivation_evidence"][:2]
            ],
            "retired_skills": item.get("retired_skills", []),
            "observed_episode_count": len(matching),
            "observation_examples": examples,
        }
        if len(json.dumps([*result, projected], ensure_ascii=False, indent=2)) > 12000:
            break
        result.append(projected)
    return result

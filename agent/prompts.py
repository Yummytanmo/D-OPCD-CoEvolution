"""Conditional prompt construction for task execution and harness evolution."""

from __future__ import annotations

import json
from typing import Any


class PromptManager:
    """Build prompts from semantic inputs and omit unavailable context."""

    @staticmethod
    def _render(*blocks: str | tuple[str, object | None]) -> str:
        rendered = []
        for block in blocks:
            if isinstance(block, str):
                rendered.append(block.strip())
                continue
            heading, value = block
            if value is None or not str(value).strip():
                continue
            rendered.append(f"{heading}\n{str(value).strip()}")
        return "\n\n".join(rendered)

    @staticmethod
    def _json(value: Any) -> str:
        return json.dumps(value, ensure_ascii=False, indent=2)

    def decompose(self, user_prompt: str, *, max_questions: int = 10) -> str:
        return (
            "Analyze the user's image generation prompt and break it into specific "
            "visual requirements. For each requirement, write a question answerable "
            "with yes or no. The questions must verify whether the requirement is "
            "present in an image. Produce at most "
            f"{int(max_questions)} independent questions. Prioritize the most important "
            "requirements and constraints that an image generator is likely to miss.\n"
            "Respond ONLY with a JSON array of strings.\n\n"
            f"USER PROMPT:\n{user_prompt}"
        )

    def verify(self, question: str) -> str:
        """Build the legacy single-question verifier prompt."""
        return (
            "Image: <image>\n"
            "Answer the following question with only 'yes' or 'no' based on the "
            f"provided image: {question}"
        )

    def verify_all(self, questions: list[str]) -> str:
        numbered = "\n".join(
            f"{index}. {question}" for index, question in enumerate(questions, 1)
        )
        return (
            "Image: <image>\n"
            "Evaluate every verification question below against the provided image. "
            "Return ONLY one JSON array with exactly the same number of items and in "
            "the same order as the questions. Each item must be an object with an "
            "'answer' string and a 'passed' JSON boolean. Use true only when the "
            "image clearly satisfies the question; otherwise use false. Do not use "
            "markdown or add commentary outside the JSON array.\n\n"
            f"VERIFICATION QUESTIONS:\n{numbered}"
        )

    def skill_router(self, *, manifest: str, user_prompt: str) -> str:
        return self._render(
            "You are an image-generation Skill Router. Every available Skill is an "
            "initial-prompt rewriter that may run exactly once before the first image "
            "generation. Choose the subset of candidate Skills whose transformations "
            "are applicable and useful for the original request. You may combine "
            "multiple complementary Skills or choose none.",
            ("### Candidate Initial-Prompt Skills", manifest),
            ("### Original User Request", user_prompt),
            "Return ONLY a JSON array containing the selected SKILL_ID strings in the "
            "order they should be considered. Return [] when no Skill is useful.",
        )

    def initial_prompt_enhancement(
        self,
        *,
        original_prompt: str,
        skill_rules: str | None,
    ) -> str:
        return self._render(
            "Produce the one prompt used for the first image-generation attempt. "
            "Preserve every requested fact. Apply all selected initial-prompt Skill "
            "instructions together as one coherent transformation. Combine compatible "
            "guidance according to the original request and return only the resulting "
            "prompt.",
            ("### Initial-Prompt Skill Instructions", skill_rules),
            ("### Original Prompt", original_prompt),
            "Return ONLY the final enhanced prompt.",
        )

    # Compatibility for callers outside the evolution runner. The semantic boundary is
    # still initial-only; refinement has no corresponding skill argument.
    def plan_enhancement(
        self,
        *,
        original_prompt: str,
        skill_rules: str | None,
    ) -> str:
        return self.initial_prompt_enhancement(
            original_prompt=original_prompt,
            skill_rules=skill_rules,
        )

    def summarize_experience(
        self,
        *,
        current_prompt: str,
        passed: str,
        failed: str,
        current_thought: str,
        previous_experiences: str | None,
    ) -> str:
        return self._render(
            "Task: Summarize the experience of the current image generation attempt.",
            (
                "--- CURRENT ATTEMPT ---",
                f"Prompt used: {current_prompt}\n"
                f"Passed requirements: {passed}\n"
                f"Failed requirements: {failed}\n"
                f"Reasoning/Thought before generation: {current_thought}\n"
                "Image: <image>",
            ),
            ("--- PREVIOUS EXPERIENCES ---", previous_experiences),
            (
                "--- ANALYSIS ---",
                "Based on the image, internal verification results, thought process, "
                "and prior attempts from this task, summarize what worked, what failed, "
                "and what should be changed in the next attempt. Keep it under 100 "
                "words. Do not include an introduction.",
            ),
        )

    def refine(
        self,
        *,
        original_prompt: str,
        memory: str | None,
        history_log: str,
    ) -> str:
        """Refine from task evidence and insights, never from an initial-only Skill."""
        return self._render(
            "Task: Refine the image generation prompt after failed internal checks.",
            ("ORIGINAL INTENT:", original_prompt),
            ("--- RELEVANT CROSS-TASK INSIGHTS ---", memory),
            ("--- ATTEMPT HISTORY ---", history_log),
            (
                "--- REQUIREMENTS ---",
                "1. Reinforce requirements that failed in the latest attempt.\n"
                "2. Preserve requirements that previously passed.\n"
                "3. Apply a relevant insight only when its condition matches.\n"
                "4. Keep all original user constraints and avoid conflicting "
                "language.\nReturn ONLY the new prompt.",
            ),
        )

    def episode_summary(
        self,
        *,
        original_prompt: str,
        returned_attempt: int | None,
        attempts: list[dict[str, Any]],
        attempt_deltas: list[dict[str, Any]],
        feedback_text: str | None,
        reward: float | None,
    ) -> str:
        instructions = """You summarize one completed image-generation task into one immutable episode.

Describe only what happened in this task.

Requirements:
1. Preserve the original task intent.
2. Summarize the important attempt-to-attempt changes and observed check transitions.
3. Every observation must reference valid attempts that support it.
4. Use no_clear_effect when multiple material things changed simultaneously, only
   task-level feedback is available, or the evidence does not isolate an effect.
5. Record unresolved failures or ambiguity at termination.
6. Do not generalize across tasks.
7. Do not create an insight, lesson, skill, category, or signature.
8. Do not claim causality without a controlled comparison.
9. Return at most five observations.

Return ONLY one JSON object with this schema:
{
  "task_summary": "one or two sentences",
  "trajectory_summary": "chronological task-local account",
  "observations": [
    {
      "effect": "helpful|harmful|no_clear_effect",
      "change": "task-local prompt or harness change",
      "observed_result": "task-local observed outcome",
      "evidence_attempts": [1, 2]
    }
  ],
  "unresolved": ["outstanding task-local issue"]
}"""
        feedback = {}
        if feedback_text is not None:
            feedback["text"] = feedback_text
        if reward is not None:
            feedback["normalized_reward"] = reward
        return self._render(
            instructions,
            ("ORIGINAL TASK:", original_prompt),
            ("RETURNED ATTEMPT:", returned_attempt),
            ("ATTEMPTS:", self._json(attempts)),
            ("DETERMINISTIC ATTEMPT DELTAS:", self._json(attempt_deltas)),
            ("AVAILABLE POST-TASK FEEDBACK:", self._json(feedback)),
        )

    def insight_consolidation(
        self,
        *,
        episodes: list[dict[str, Any]],
        current_insights: list[dict[str, Any]],
    ) -> str:
        instructions = """Maintain the situational Insight memory using this batch of
Episodes and the related active Insights retrieved for it.

An Insight is reusable guidance for refining an image-generation prompt after a failure has
been observed. It should capture the recognizable situation, the useful prompt adjustment
or constraint to preserve, and any important boundary or uncertainty. Keep it specific
enough for retrieval in a similar failure and general enough to transfer across tasks.
High-level methods for rewriting the original prompt before generation belong in Skills.

Build the smallest coherent memory update supported by the evidence.

Use the evidence as follows:
- Treat attempts, check transitions, and final outcomes as the factual record.
- Treat an optional summary as a secondary interpretation.
- Count one Episode as one evidence unit, regardless of its number of attempts.

Choose the memory update that best fits the evidence:
- Strengthen an Insight when its current meaning already fits.
- Revise its wording or scope when the lesson remains useful but needs correction.
- Record a contradiction when evidence weakens or limits an Insight.
- Merge overlapping Insights into one complete formulation.
- Archive knowledge that has become obsolete or redundant.
- Add an Insight for a distinct, transferable refinement lesson.
- Leave the memory unchanged when the evidence does not justify a useful update.

Write all new or revised Insight text as clear, natural situational guidance.

Return only one JSON object containing an ordered `operations` array. The available actions
and their parameters are:
- ADD: `text`, `episode_refs`
- SUPPORT: `insight_ref`, `episode_refs`
- REVISE: `insight_ref`, `text`, `episode_refs`
- CONTRADICT: `insight_ref`, `episode_refs`, `reason`
- MERGE: `target_insight_ref`, `merged_insight_refs`, `text`, `episode_refs`
- ARCHIVE: `insight_ref`, `episode_refs`, `reason`, and optional
  `replacement_insight_ref`

Each item contains `op` and the parameters for that action. Use only the local references
shown below. Operations execute from first to last as one transaction; use an empty array
when the memory should remain unchanged. For the same `insight_ref`, do not use the same
`episode_ref` in both a supporting action and a CONTRADICT action. When one Episode both
supports part of an Insight and exposes a limitation, use one REVISE action for that pair
instead of SUPPORT plus CONTRADICT."""
        return self._render(
            instructions,
            ("RETRIEVED ACTIVE INSIGHT WORKING SET:", self._json(current_insights)),
            ("CURRENT-BATCH EPISODES:", self._json(episodes)),
        )

    @staticmethod
    def _skill_operation_protocol() -> str:
        """Return the shared operation action space and exact JSON value shapes."""
        return """Return only one JSON object containing an ordered `operations` array. The available edits
and their parameters are:
- CREATE: `suggested_id`, `content`, `source_insight_refs`
- REPLACE_TEXT: `skill_id`, `old_text`, `new_text`, `source_insight_refs`
- INSERT_TEXT: `skill_id`, `anchor`, `text`, `position`, `source_insight_refs`
- DELETE_TEXT: `skill_id`, `text`, `source_insight_refs`
- RENAME: `skill_id`, `new_title`, `source_insight_refs`
- SUMMARY: `skill_id`, `content`, `source_insight_refs`
- SPLIT: `skill_id`, `new_skills`, `source_insight_refs`
- MERGE: `target_skill_id`, `merged_skill_ids`, `content`, `source_insight_refs`
- RETIRE: `skill_id`, `replacement_skill_id`

Each item contains `op` and the parameters for that edit. Use the displayed skill_ids and
copy the shortest uniquely matching exact text from the same target Skill, including
frontmatter when it is edited. Operations execute from first to last as one transaction.

Return one JSON object:

{
  "operations": [
    <operation>
  ]
}

Each operation must use one of these shapes:

CREATE:
{"op":"CREATE","suggested_id":string,"content":string,"source_insight_refs":[string]}

REPLACE_TEXT:
{"op":"REPLACE_TEXT","skill_id":string,"old_text":string,"new_text":string,
 "source_insight_refs":[string]}

INSERT_TEXT:
{"op":"INSERT_TEXT","skill_id":string,"anchor":string,"text":string,
 "position":"before"|"after",
 "source_insight_refs":[string]}

DELETE_TEXT:
{"op":"DELETE_TEXT","skill_id":string,"text":string,
 "source_insight_refs":[string]}

RENAME:
{"op":"RENAME","skill_id":string,"new_title":string,
 "source_insight_refs":[string]}

SUMMARY:
{"op":"SUMMARY","skill_id":string,"content":string,
 "source_insight_refs":[string]}

SPLIT:
{"op":"SPLIT","skill_id":string,
 "new_skills":[
   {"suggested_id":string,"content":string},
   {"suggested_id":string,"content":string}
 ],"source_insight_refs":[string]}

MERGE:
{"op":"MERGE","target_skill_id":string,"merged_skill_ids":[string],
 "content":string,"source_insight_refs":[string]}

RETIRE:
{"op":"RETIRE","skill_id":string,"replacement_skill_id":string}"""

    def skill_evolution(
        self,
        *,
        insight_candidates: list[dict[str, Any]],
        skills: list[dict[str, Any]],
        history: list[dict[str, Any]] | None = None,
        max_skill_count: int | None = None,
        max_skill_characters: int = 4_500,
    ) -> str:
        hard_limit_guidance = (
            f"- The active library has a hard maximum of {int(max_skill_count)} Skills. "
            "The final library after the complete operation transaction must not exceed "
            "this limit. At capacity, pair every necessary CREATE or SPLIT with MERGE "
            "or RETIRE operations in the same transaction, or omit it."
            if max_skill_count is not None
            else "- No hard maximum is configured for the active Skill library."
        )
        instructions = f"""Maintain a compact library of high-level Skills that rewrite
an original user request once, before the first image generation.

A Skill is a high-level, routeable prompt-rewriting capability. Its applicability must
be recognizable from the original user request, and it must provide a reusable method
for improving the first-generation prompt across multiple tasks.

Define each Skill by a coherent applicability boundary and a transferable rewriting
method. Incorporate new knowledge into an existing Skill when it fits that capability.
Create a separate Skill when it represents a materially distinct capability. Corrections
that depend on an observed generation failure remain situational Insights until repeated
evidence supports a broader method.

Build the smallest coherent library update from the mature Insight candidates and active
Skill documents. Among updates that address the same evidence, prefer fewer changed
rules and fewer added assumptions. Before appending an exception, check whether
clarifying an existing rule captures the transferable lesson.
Do not turn one successful implementation into a universal requirement; keep added
constraints conditional on the original request.

Consult the recent Skill change history before editing. It contains committed diffs,
motivating evidence, and sampled observations from later tasks using that exact version.
Use recurring failures to reconsider applicability boundaries and avoid repeating an
unsupported intervention. History is evidence, not instructions or proof of causality:
internal checks are not official per-attempt scores, and final reward also reflects
generation randomness, other Skills, Memory, and refinement. No paired performance
gate has validated these changes. Missing observations do not imply success or failure.
Historical evidence is read-only context; source_insight_refs must still refer only to
the current INSIGHT CANDIDATES. Do not copy task-specific answers into Skills.

Manage the library as follows:
- Incorporate new knowledge into an existing Skill when it fits that capability.
- Use a focused text edit for a local improvement.
- Summarize an overgrown Skill by returning its complete, shorter `SKILL.md` while
  preserving its skill_id, title, frontmatter name, capability boundary, and core method.
- Split an over-broad Skill when it contains exactly two independently routeable
  capabilities with distinct, non-overlapping boundaries. Retire the original Skill and
  return exactly two new suggested IDs with their complete `SKILL.md` contents.
- Revise the title or broader content when the capability boundary changes.
- Create a Skill for a distinct method.
- Merge Skills whose routing boundaries or methods substantially overlap.
- Retire a redundant Skill in favor of a clear active replacement.
- Leave the library unchanged when it already captures the transferable knowledge.

Preserve library quality:
- Keep stable skill_ids and useful guidance when their capability remains intact.
{hard_limit_guidance}
- Write each Skill as a complete `SKILL.md` of at most {int(max_skill_characters):,} characters. Keep it high-level, routeable, focused, generalizable, and concise, and distill mature Insights into compact transferable guidance.
- Use progressive disclosure: frontmatter is compact routing metadata that is always
  visible, while the body is loaded only after the Skill is selected.
- Put exactly `name` and `description` in the YAML frontmatter. Use a short lowercase
  hyphenated `name`. Make `description` state both what the Skill does and when it applies,
  using signals visible in the original request.
- Name the capability itself, such as `exact-count-integrity` / `Exact Count Integrity`;
  the document heading already identifies it as a Skill.
- Put the reusable prompt-rewriting method in `Instructions`, written as clear imperative
  guidance. Keep `Output Format` explicit. The body should complement rather than repeat
  the routing description.
- Use `source_insight_refs` only for candidate Insights newly absorbed by a change;
  existing evidence is inherited automatically.

Use this complete document shape for CREATE, SUMMARY, SPLIT, and MERGE content:
```markdown
---
name: <lowercase-hyphenated-name>
description: <what this Skill does and when to select it>
---

# Skill: <Human-readable title>

## Instructions
<transferable method for rewriting the original request>

## Output Format
Return ONLY the final enhanced prompt text.
```"""
        library_status = {
            "max_skill_count": (
                int(max_skill_count) if max_skill_count is not None else None
            ),
            "max_skill_characters": int(max_skill_characters),
            "active_skill_count": len(skills),
            "remaining_capacity": (
                max(0, int(max_skill_count) - len(skills))
                if max_skill_count is not None
                else None
            ),
        }
        return self._render(
            instructions,
            self._skill_operation_protocol(),
            ("LIBRARY STATUS:", self._json(library_status)),
            ("INSIGHT CANDIDATES:", self._json(insight_candidates)),
            ("EXISTING INITIAL-PROMPT SKILLS:", self._json(skills)),
            ("RECENT SKILL CHANGE HISTORY (observational, not causal):", self._json(history or [])),
        )

    def skill_capacity_reorganization(
        self,
        *,
        max_skill_count: int,
        skills: list[dict[str, Any]],
    ) -> str:
        instructions = f"""Reorganize the staged active Skill library only enough to satisfy its
hard maximum of {int(max_skill_count)} Skills. The primary Skill Manager decision has
already been applied; do not revisit its mature-Insight decision.

Preserve the original Skill contents, stable skill_ids, routing boundaries, and transferable
guidance as fully as possible. Prefer MERGE for genuinely overlapping capabilities and RETIRE
only for a redundant existing Skill with a clear active replacement. Use MERGE, not RETIRE, to
fold a newly staged Skill into an existing Skill. Do not create new capability families. Return
an empty operation list if the displayed library already satisfies the limit.

Every final Skill must remain a complete high-level, routeable, focused, generalizable, and
concise `SKILL.md`. Its applicability must be recognizable from the original user request.
Keep exactly `name` and `description` in YAML frontmatter, put the reusable rewriting method
under `Instructions`, and keep `Output Format` explicit. Use an empty
`source_insight_refs` list for every maintenance edit because this stage absorbs no new
Insight evidence.

The complete operation transaction must finish with at most {int(max_skill_count)} active
Skills."""
        status = {
            "max_skill_count": int(max_skill_count),
            "active_skill_count": len(skills),
            "excess_skill_count": max(0, len(skills) - int(max_skill_count)),
        }
        return self._render(
            instructions,
            self._skill_operation_protocol(),
            ("LIBRARY STATUS AFTER PRIMARY UPDATE:", self._json(status)),
            ("STAGED ACTIVE SKILLS:", self._json(skills)),
        )

    def skill_length_reorganization(
        self,
        *,
        max_skill_count: int | None,
        max_skill_characters: int,
        skills: list[dict[str, Any]],
    ) -> str:
        over_limit = [
            {
                "skill_id": str(item["skill_id"]),
                "characters": int(item["characters"]),
                "excess_characters": int(item["characters"])
                - int(max_skill_characters),
            }
            for item in skills
            if int(item["characters"]) > int(max_skill_characters)
        ]
        capacity = (
            f"The final library must also contain at most {int(max_skill_count)} active Skills."
            if max_skill_count is not None
            else "No active Skill-count maximum is configured."
        )
        instructions = f"""Reorganize the staged active Skill library only enough to ensure that
every complete `SKILL.md` is at most {int(max_skill_characters)} characters. The primary
Skill Manager decision and any capacity maintenance have already been applied; do not revisit
their mature-Insight decisions.

Preserve each Skill's useful transferable guidance and capability coverage. Use SUMMARY when
one coherent capability can be distilled. Use SPLIT only when an over-broad Skill contains
exactly two independently routeable capabilities with distinct, non-overlapping boundaries.
Avoid changing Skills that already satisfy the threshold unless needed to keep the resulting
library coherent. {capacity}

Every final Skill must remain high-level, routeable, focused, generalizable, and concise. Its
applicability must be recognizable from the original user request. Keep exactly `name` and
`description` in YAML frontmatter, put the reusable rewriting method under `Instructions`,
and keep `Output Format` explicit. Use an empty `source_insight_refs` list for every
maintenance edit because this stage absorbs no new Insight evidence.

The complete operation transaction must leave every active Skill at or below
{int(max_skill_characters)} characters."""
        status = {
            "max_skill_count": (
                int(max_skill_count) if max_skill_count is not None else None
            ),
            "active_skill_count": len(skills),
            "max_skill_characters": int(max_skill_characters),
            "over_limit_skills": over_limit,
        }
        return self._render(
            instructions,
            self._skill_operation_protocol(),
            ("LIBRARY STATUS AFTER PRIOR STAGES:", self._json(status)),
            ("STAGED ACTIVE SKILLS:", self._json(skills)),
        )

    def skill_reorganization(
        self,
        *,
        max_skill_count: int,
        active_skills: list[dict[str, Any]],
        proposed_skill: dict[str, Any],
        proposed_skill_insights: list[dict[str, Any]],
    ) -> str:
        instructions = """Reorganize the proposed Skill together with the active Skill
library into a compact set of high-level prompt-rewriting capability families.

Use routing boundaries and rewriting methods to decide the clearest organization. Active
Skill documents contain the knowledge already absorbed; the supporting mature Insights
provide evidence for stable capability boundaries.

Organize the capabilities as follows:
- Integrate the proposal when it forms one method with an existing capability.
- Keep it distinct when it contributes a different high-level method.
- Consolidate existing overlap when a merged capability is clearer.
- Preserve an existing skill_id when it remains the main capability in a result.
- Retire a weaker or redundant Skill only in favor of a named active replacement.

Represent the resulting library as follows:
- Put unchanged Skills in `kept_skill_ids`.
- Put every new, modified, or merged Skill in `result_skills`.
- List the contributing active IDs in `source_skill_ids` and use
  `__proposed_skill__` wherever the proposal is incorporated.
- Write each result as a complete `SKILL.md`, preferably within about 4,500 characters.
- Use exactly `name` and `description` in YAML frontmatter. Treat them as the compact
  routing layer: `name` is lowercase and hyphenated; `description` states what the Skill
  does and when it applies from the original request.
- Name the capability itself, such as `spatial-composition` / `Spatial Composition`;
  the document heading already identifies it as a Skill.
- Put the transferable rewriting method under `Instructions` and the required response
  under `Output Format`. This body is disclosed only after routing selects the Skill.
- Respect the active Skill limit in `LIBRARY STATUS`.

Use this complete document shape for every `result_skills[].content`:
```markdown
---
name: <lowercase-hyphenated-name>
description: <what this Skill does and when to select it>
---

# Skill: <Human-readable title>

## Instructions
<transferable method for rewriting the original request>

## Output Format
Return ONLY the final enhanced prompt text.
```

Set `proposed_skill_disposition` to:
- `INTEGRATE` when the proposal joins an existing capability family.
- `DISTINCT` when it remains a separate capability.
- `DISCARD` when the active library already contains its useful knowledge.

Return ONLY one JSON object:
{
  "proposed_skill_disposition": "INTEGRATE|DISTINCT|DISCARD",
  "kept_skill_ids": ["unchanged_active_skill_id"],
  "retired_skills": [
    {
      "skill_id": "retired_active_skill_id",
      "replacement_skill_id": "kept_or_result_skill_id"
    }
  ],
  "result_skills": [
    {
      "skill_id": "stable_existing_or_new_skill_id",
      "source_skill_ids": ["active_skill_id_or___proposed_skill__"],
      "content": "complete SKILL.md text for this resulting Skill"
    }
  ]
}

Keep the result consistent:
- Place every active Skill in exactly one of `kept_skill_ids`, a result's
  `source_skill_ids`, or `retired_skills`.
- When the disposition is `DISCARD`, return the active library unchanged."""
        status = {
            "max_skill_count": int(max_skill_count),
            "active_skill_count": len(active_skills),
            "remaining_capacity": (
                max(0, int(max_skill_count) - len(active_skills))
                if max_skill_count is not None
                else None
            ),
        }
        return self._render(
            instructions,
            ("LIBRARY STATUS:", self._json(status)),
            ("ACTIVE SKILLS:", self._json(active_skills)),
            ("PROPOSED SKILL:", self._json(proposed_skill)),
            (
                "MATURE INSIGHTS SUPPORTING THE PROPOSED SKILL:",
                self._json(proposed_skill_insights),
            ),
        )


__all__ = ["PromptManager"]

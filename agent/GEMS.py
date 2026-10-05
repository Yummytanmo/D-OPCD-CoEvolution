import json
import re
import uuid

from agent.base_agent import BaseAgent
from agent.evolution.models import (
    AttemptRecord,
    FirstAttemptReplay,
    PlanResult,
    RunResult,
)
from agent.evolution.prompting import PromptCompiler
from agent.evolution.soft_constraints import warn_soft_limit
from agent.prompts import PromptManager
from agent.pq_strategy_sampler import build_prompt, strategy_for


_REFINEMENT_QUERY_SECTION_CHARS = 480


class GEMS(BaseAgent):
    def __init__(
        self,
        gen_url,
        mllm_url,
        max_iterations,
        *,
        skill_manager=None,
        insight_store=None,
        generator_model="unknown",
        round_id=0,
        verification_max_questions=10,
        generator_config=None,
        mllm_config=None,
        enable_skills=True,
        enable_memory=True,
        prompt_adapter=False,
        prompt_adapter_seed=20260920,
        verbose=True,
    ):
        super().__init__(
            gen_url,
            mllm_url,
            skill_manager=skill_manager,
            generator_config=generator_config,
            mllm_config=mllm_config,
        )
        self.max_iterations = max_iterations
        self.insight_store = insight_store
        self.generator_model = str(generator_model)
        self.round_id = int(round_id)
        self.verification_max_questions = int(verification_max_questions)
        self.enable_skills = bool(enable_skills)
        self.enable_memory = bool(enable_memory)
        self.prompt_adapter = bool(prompt_adapter)
        self.prompt_adapter_seed = int(prompt_adapter_seed)
        self.verbose = bool(verbose)
        if self.verification_max_questions <= 0:
            raise ValueError("verification.max_questions must be positive")
        self.prompt_compiler = PromptCompiler()
        self.prompt_manager = PromptManager()

    def _log(self, *values, **kwargs) -> None:
        """Emit task-level diagnostics only for interactive single-agent runs."""
        if self.verbose:
            print(*values, **kwargs)

    def decompose(self, prompt: str, *, warning_context: str | None = None) -> list:
        """Create task-local checks without consulting cross-task memory or skills."""
        task = self.prompt_manager.decompose(
            prompt,
            max_questions=self.verification_max_questions,
        )
        response = self.think(task).strip()
        try:
            match = re.search(r"\[.*\]", response, re.DOTALL)
            questions = json.loads(match.group() if match else response)
        except json.JSONDecodeError as error:
            self._log(
                f"[Error] JSON parsing failed; using question-like lines: {error}"
            )
            valid = [
                line.strip() for line in response.splitlines() if "?" in line
            ]
            warn_soft_limit(
                "fallback verification questions"
                + (f" ({warning_context})" if warning_context else ""),
                actual=len(valid),
                preferred_max=self.verification_max_questions,
                unit="questions",
            )
            return valid
        if not isinstance(questions, list):
            return []
        valid = [question for question in questions if isinstance(question, str)]
        warn_soft_limit(
            "verification questions"
            + (f" ({warning_context})" if warning_context else ""),
            actual=len(valid),
            preferred_max=self.verification_max_questions,
            unit="questions",
        )
        return valid

    def verify_image(self, image_bytes: bytes, questions: list) -> list:
        """Evaluate every question in one multimodal verifier call."""
        if not questions:
            return []

        full_query = self.prompt_manager.verify_all(questions)
        try:
            response = self.think(full_query, images=[image_bytes])
            return self._parse_verifications(response, questions)
        except Exception as error:
            # A malformed batch must not abort the task or trigger extra verifier
            # calls. Conservatively fail every check so refinement can continue.
            answer = f"Error: batch verification failed: {error}"
            return [
                {"question": question, "answer": answer, "passed": False}
                for question in questions
            ]

    @staticmethod
    def _parse_verifications(response: str, questions: list) -> list:
        match = re.search(r"\[.*\]", str(response), re.DOTALL)
        try:
            payload = json.loads(match.group() if match else response)
        except (json.JSONDecodeError, TypeError) as error:
            raise ValueError("verifier returned invalid JSON") from error

        if not isinstance(payload, list) or len(payload) != len(questions):
            raise ValueError("verifier returned the wrong number of results")

        verifications = []
        for question, item in zip(questions, payload):
            if not isinstance(item, dict) or not isinstance(item.get("passed"), bool):
                raise ValueError(
                    "each verifier result must contain a boolean 'passed' field"
                )
            verifications.append(
                {
                    # Input order and wording are authoritative. The verifier's
                    # echoed question is intentionally ignored.
                    "question": question,
                    "answer": str(item.get("answer", "")).strip(),
                    "passed": item["passed"],
                }
            )
        return verifications

    def _retrieve_insights(self, query: str) -> list[dict]:
        """Retrieve situational knowledge after a failed generation attempt."""
        if not self.enable_memory or self.insight_store is None:
            return []
        return self.insight_store.retrieve(query)

    @staticmethod
    def _parse_skill_selection(
        response: str,
        candidate_by_id: dict[str, dict],
    ) -> list[dict]:
        """Resolve an ordered, unique subset while ignoring unknown Skill IDs."""
        raw = str(response).strip()
        parsed = None
        try:
            parsed = json.loads(raw)
        except json.JSONDecodeError:
            match = re.search(r"\[.*\]", raw, re.DOTALL)
            if match:
                try:
                    parsed = json.loads(match.group())
                except json.JSONDecodeError:
                    parsed = None

        if isinstance(parsed, list):
            requested = [str(item).strip() for item in parsed if isinstance(item, str)]
        elif isinstance(parsed, str):
            requested = [parsed.strip()]
        elif raw in candidate_by_id:
            # Accept the former single-ID response during rolling upgrades.
            requested = [raw]
        else:
            requested = []

        selected = []
        seen = set()
        for skill_id in requested:
            if skill_id in seen or skill_id not in candidate_by_id:
                continue
            selected.append(candidate_by_id[skill_id])
            seen.add(skill_id)
        return selected

    @staticmethod
    def _build_refinement_query(
        *,
        original_prompt: str,
        current_attempt: AttemptRecord,
        previous_attempt: AttemptRecord | None,
    ) -> str:
        """Describe the current failure situation for semantic Insight retrieval."""

        def clipped(value: str, limit: int = _REFINEMENT_QUERY_SECTION_CHARS) -> str:
            normalized = " ".join(str(value).split())
            if len(normalized) <= limit:
                return normalized
            head = normalized[: limit - 1].rsplit(" ", 1)[0]
            return f"{head or normalized[: limit - 1]}…"

        def items(values: list[str], *, per_item: int | None = None) -> str:
            if not values:
                return "- None"
            limit = per_item or max(
                40,
                _REFINEMENT_QUERY_SECTION_CHARS // len(values) - 2,
            )
            return "\n".join(f"- {clipped(value, limit)}" for value in values)

        sections = [
            "Task:",
            clipped(original_prompt),
        ]
        if previous_attempt is None:
            sections.extend(
                [
                    "Current failures:",
                    items(current_attempt.failed),
                ]
            )
        else:
            previous_passed = set(previous_attempt.passed)
            previous_failed = set(previous_attempt.failed)
            current_failed = set(current_attempt.failed)
            failure_groups = [
                sorted(previous_failed & current_failed),
                sorted(previous_passed & current_failed),
            ]
            failure_count = sum(len(group) for group in failure_groups)
            per_failure_item = max(
                40,
                _REFINEMENT_QUERY_SECTION_CHARS // max(1, failure_count) - 2,
            )
            sections.extend(
                [
                    "Persistent failures:",
                    items(failure_groups[0], per_item=per_failure_item),
                    "New regressions:",
                    items(failure_groups[1], per_item=per_failure_item),
                ]
            )
        sections.extend(
            [
                "Observed result:",
                clipped(current_attempt.experience or "None"),
            ]
        )
        return "\n".join(sections)

    def plan_with_context(self, original_prompt: str, *, sample_key: str | None = None) -> PlanResult:
        candidates = (
            self.skill_manager.active_skills() if self.enable_skills else []
        )
        selected_skills = []
        if candidates:
            manifest = self.skill_manager.get_skill_manifest()
            decision = self.think(
                self.prompt_manager.skill_router(
                    manifest=manifest,
                    user_prompt=original_prompt,
                )
            ).strip()
            candidate_by_id = {
                str(item["skill_id"]): item for item in candidates
            }
            selected_skills = self._parse_skill_selection(
                decision,
                candidate_by_id,
            )
            if selected_skills:
                self._log(
                    "🎯 Skills selected: "
                    + ", ".join(
                        f"{skill['skill_id']}@v{skill.get('version', 0)}"
                        for skill in selected_skills
                    )
                )

        if not selected_skills:
            self._log("⏭️ No initial-prompt Skill selected.")
            return PlanResult(prompt=original_prompt)

        skill_rules = self.prompt_compiler.initial_skills_rules(selected_skills)
        if self.prompt_adapter:
            strategy = strategy_for(sample_key or original_prompt, seed=self.prompt_adapter_seed)
            self._log("PQ adapter strategy: " + strategy)
            enhancement_task = build_prompt(original_prompt, skill_rules, strategy)
        else:
            enhancement_task = self.prompt_manager.initial_prompt_enhancement(
                skill_rules=skill_rules,
                original_prompt=original_prompt,
            )
        enhanced = self.think(enhancement_task).strip()
        if not enhanced:
            self._log("⚠️ Initial prompt rewrite was empty; retrying the MLLM once.")
            enhanced = self.think(enhancement_task).strip()
        return PlanResult(
            prompt=enhanced or original_prompt,
            selected_skills=selected_skills,
        )

    def plan(self, original_prompt: str) -> str:
        """Backward-compatible prompt-only planning API."""
        return self.plan_with_context(original_prompt).prompt

    def _refine_after_failure(
        self,
        *,
        original_prompt: str,
        attempts: list[AttemptRecord],
        refinement_insight_ids: set[str],
    ) -> tuple[str, str]:
        """Build the next prompt from one shared failed-attempt history."""
        current_attempt = attempts[-1]
        refine_query = self._build_refinement_query(
            original_prompt=original_prompt,
            current_attempt=current_attempt,
            previous_attempt=attempts[-2] if len(attempts) > 1 else None,
        )
        refine_insights = self._retrieve_insights(refine_query)
        refinement_insight_ids.update(
            str(item["insight_id"]) for item in refine_insights
        )

        history_log = ""
        history_images = []
        for record in attempts:
            history_log += (
                f"Attempt {record.iteration}:\n"
                f"- Prompt: {record.prompt}\n"
                f"- Experience: {record.experience}\n"
                f"- Passed: {', '.join(record.passed) if record.passed else 'None'}\n"
                f"- Failed: {', '.join(record.failed) if record.failed else 'None'}\n"
                "- Image: <image>\n\n"
            )
            history_images.append(record.image_bytes)

        refine_task = self.prompt_manager.refine(
            original_prompt=original_prompt,
            memory=self.prompt_compiler.insights(refine_insights),
            history_log=history_log,
        )
        next_prompt, next_thought = self.think_with_thought(
            refine_task,
            images=history_images,
        )
        next_prompt = next_prompt.strip()
        if not next_prompt:
            self._log("⚠️ Prompt refinement was empty; retrying the MLLM once.")
            next_prompt, next_thought = self.think_with_thought(
                refine_task,
                images=history_images,
            )
            next_prompt = next_prompt.strip()
        return next_prompt or current_attempt.prompt.strip() or original_prompt, next_thought

    def run_with_result(
        self,
        item: dict,
        *,
        first_attempt_replay: FirstAttemptReplay | None = None,
    ) -> RunResult:
        self.reset_call_counters()
        original_prompt = str(item.get("prompt", ""))
        task_id = str(item.get("sample_id") or item.get("task_id") or uuid.uuid4())
        base_seed = item.get("inference_seed")
        base_seed = int(base_seed) if base_seed is not None else None
        # Retained as an empty compatibility field in RunResult. Situational Insight
        # memory is intentionally unavailable before the first attempt fails.
        initial_insight_ids: set[str] = set()
        refinement_insight_ids: set[str] = set()
        replayed_generator_calls = 0
        replayed_mllm_calls = 0

        if first_attempt_replay is None:
            self._log("\n[Step 0] Initial-prompt Skill planning...")
            if self.prompt_adapter:
                # Preserve the namespaces used by the recorded adapter experiments.
                benchmark = str(item.get("benchmark") or "")
                namespace = str(item.get("adapter_dataset") or {
                    "geneval": "GenEval", "geneval2": "GenEval2",
                    "wise": "WISE", "r2ibench": "R2I", "ocr": "OCR",
                }.get(benchmark.lower(), benchmark))
                identity = str(item.get("sample_id") or item.get("task_id") or original_prompt)
                sample_key = f"{namespace}:{identity}" if namespace else identity
                plan = self.plan_with_context(original_prompt, sample_key=sample_key)
            else:
                plan = self.plan_with_context(original_prompt)
            current_prompt = plan.prompt
            selected_skill_documents = list(plan.selected_skills)
            selected_skills = []
            used_initial_skill = bool(selected_skill_documents)
            for selected_skill in selected_skill_documents:
                selected_skills.append(
                    {
                        "skill_id": selected_skill["skill_id"],
                        "version": int(selected_skill.get("version", 0)),
                        "execution_stage": "before_first_generation",
                    }
                )
            current_thought = (
                "Initial prompt compiled from the selected initial-prompt Skills."
                if used_initial_skill
                else "Initial prompt uses the original request."
            )
            # Full Skill bodies are intentionally discarded after the one initial rewrite.
            plan.selected_skills.clear()
            plan.selected_skill = None
            selected_skill_documents.clear()

            # Keep verification task-local so learned guidance can improve generation
            # without quietly redefining the success criteria used to check the image.
            self._log("\n[Step 1] Decomposing requirements...")
            questions = self.decompose(
                original_prompt,
                warning_context=f"task={task_id}",
            )
            attempts: list[AttemptRecord] = []

            if not questions:
                image = self.generate(current_prompt, seed=base_seed)
                attempts.append(
                    AttemptRecord(
                        iteration=1,
                        prompt=current_prompt,
                        passed=[],
                        failed=[],
                        seed=base_seed,
                        experience="No internal verification questions were produced.",
                        image_bytes=image,
                    )
                )
                return self._result(
                    task_id,
                    original_prompt,
                    current_prompt,
                    image,
                    attempts,
                    selected_skills,
                    initial_insight_ids,
                    refinement_insight_ids,
                    returned_attempt=1,
                )

            best_image = None
            best_prompt = current_prompt
            best_iteration = None
            max_passed = -1
            first_iteration = 1
        else:
            replay = first_attempt_replay
            if replay.task_id != task_id:
                raise ValueError(
                    "first-attempt replay task_id mismatch: "
                    f"expected {task_id!r}, got {replay.task_id!r}"
                )
            if replay.original_prompt != original_prompt:
                raise ValueError(
                    f"first-attempt replay prompt mismatch for task {task_id}"
                )
            if replay.attempt.iteration != 1:
                raise ValueError("first-attempt replay must contain iteration 1")
            if replay.attempt.image_bytes is None:
                raise ValueError("first-attempt replay does not contain image bytes")
            if (
                base_seed is not None
                and replay.attempt.seed is not None
                and int(replay.attempt.seed) != base_seed
            ):
                raise ValueError(
                    f"first-attempt replay seed mismatch for task {task_id}: "
                    f"expected {base_seed}, got {replay.attempt.seed}"
                )
            overlap = set(replay.attempt.passed) & set(replay.attempt.failed)
            if overlap:
                raise ValueError(
                    f"first-attempt replay has contradictory checks: {sorted(overlap)}"
                )

            self._log(
                "\n[Step 0-1] Replaying the frozen Skill-only first trajectory "
                f"from {replay.source}"
            )
            current_prompt = replay.attempt.prompt
            selected_skills = [dict(item) for item in replay.selected_skills]
            current_thought = (
                "Initial prompt compiled from the selected initial-prompt Skills."
                if selected_skills
                else "Initial prompt uses the original request."
            )
            replayed_attempt = AttemptRecord(
                iteration=1,
                prompt=replay.attempt.prompt,
                passed=list(replay.attempt.passed),
                failed=list(replay.attempt.failed),
                seed=replay.attempt.seed,
                experience=replay.attempt.experience,
                image_path=replay.attempt.image_path,
                image_bytes=replay.attempt.image_bytes,
            )
            attempts = [replayed_attempt]
            questions = list(
                dict.fromkeys([*replayed_attempt.passed, *replayed_attempt.failed])
            )
            best_image = replayed_attempt.image_bytes
            best_prompt = replayed_attempt.prompt
            best_iteration = 1
            max_passed = len(replayed_attempt.passed)
            replayed_generator_calls = int(replay.prefix_generator_calls)
            replayed_mllm_calls = int(replay.prefix_mllm_calls)

            if not questions or not replayed_attempt.failed or self.max_iterations <= 1:
                return self._result(
                    task_id,
                    original_prompt,
                    best_prompt,
                    best_image,
                    attempts,
                    selected_skills,
                    initial_insight_ids,
                    refinement_insight_ids,
                    returned_attempt=1,
                    first_attempt_replay=replay,
                    replayed_generator_calls=replayed_generator_calls,
                    replayed_mllm_calls=replayed_mllm_calls,
                )
            if not replayed_attempt.experience.strip():
                raise ValueError(
                    f"failed first-attempt replay lacks experience for task {task_id}"
                )
            current_prompt, current_thought = self._refine_after_failure(
                original_prompt=original_prompt,
                attempts=attempts,
                refinement_insight_ids=refinement_insight_ids,
            )
            first_iteration = 2

        for iteration in range(first_iteration, self.max_iterations + 1):
            self._log(f"\n--- Round {iteration}/{self.max_iterations} ---")
            attempt_seed = base_seed + iteration - 1 if base_seed is not None else None
            image = self.generate(current_prompt, seed=attempt_seed)
            verifications = self.verify_image(image, questions)
            failed = [item["question"] for item in verifications if not item["passed"]]
            passed = [item["question"] for item in verifications if item["passed"]]

            if len(passed) > max_passed:
                max_passed = len(passed)
                best_image = image
                best_prompt = current_prompt
                best_iteration = iteration
                self._log(
                    f" [Updating best solution] {max_passed}/{len(questions)} passed"
                )

            for verification in verifications:
                marker = "✅" if verification["passed"] else "❌"
                self._log(f"  {marker} {verification['question']}")

            attempt = AttemptRecord(
                iteration=iteration,
                prompt=current_prompt,
                passed=passed,
                failed=failed,
                seed=attempt_seed,
                image_bytes=image,
            )

            if not failed:
                attempt.experience = "All internal requirements passed."
                attempts.append(attempt)
                best_image = image
                best_prompt = current_prompt
                self._log("\nSuccess: all internal requirements passed.")
                break

            if iteration >= self.max_iterations:
                attempt.experience = "Maximum refinement iterations reached."
                attempts.append(attempt)
                break

            previous_experiences = "\n".join(
                f"Round {record.iteration}: {record.experience}"
                for record in attempts
            )
            attempt.experience = self.think(
                self.prompt_manager.summarize_experience(
                    current_prompt=current_prompt,
                    passed=", ".join(passed) if passed else "None",
                    failed=", ".join(failed) if failed else "None",
                    current_thought=current_thought,
                    previous_experiences=previous_experiences,
                ),
                images=[image],
            ).strip()
            warn_soft_limit(
                f"attempt experience summary (task={task_id}, iteration={iteration})",
                actual=len(attempt.experience.split()),
                preferred_max=100,
                unit="words",
            )
            attempts.append(attempt)
            current_prompt, current_thought = self._refine_after_failure(
                original_prompt=original_prompt,
                attempts=attempts,
                refinement_insight_ids=refinement_insight_ids,
            )

        if best_image is None:
            best_image = self.generate(original_prompt, seed=base_seed)
            best_prompt = original_prompt
        self._log(f"Returning best image ({max_passed}/{len(questions)} passed).")
        return self._result(
            task_id,
            original_prompt,
            best_prompt,
            best_image,
            attempts,
            selected_skills,
            initial_insight_ids,
            refinement_insight_ids,
            returned_attempt=best_iteration,
            first_attempt_replay=first_attempt_replay,
            replayed_generator_calls=replayed_generator_calls,
            replayed_mllm_calls=replayed_mllm_calls,
        )

    def _result(
        self,
        task_id,
        original_prompt,
        final_prompt,
        final_image,
        attempts,
        selected_skills,
        initial_insight_ids,
        refinement_insight_ids,
        returned_attempt,
        *,
        first_attempt_replay: FirstAttemptReplay | None = None,
        replayed_generator_calls: int = 0,
        replayed_mllm_calls: int = 0,
    ) -> RunResult:
        return RunResult(
            task_id=task_id,
            original_prompt=original_prompt,
            final_prompt=final_prompt,
            final_image_bytes=final_image,
            attempts=attempts,
            selected_skills=selected_skills,
            initial_insight_ids=sorted(initial_insight_ids),
            refinement_insight_ids=sorted(refinement_insight_ids),
            returned_attempt=returned_attempt,
            generator_calls=self.generator_calls + int(replayed_generator_calls),
            mllm_calls=self.mllm_calls + int(replayed_mllm_calls),
            generator_model=self.generator_model,
            round_id=self.round_id,
            first_trajectory_source=(
                first_attempt_replay.source
                if first_attempt_replay is not None
                else None
            ),
            replayed_attempts=[1] if first_attempt_replay is not None else [],
        )

    def run(self, item: dict) -> bytes:
        """Run one task and return the selected image bytes."""
        result = self.run_with_result(item)
        if result.final_image_bytes is None:
            raise RuntimeError("image generation returned no bytes")
        return result.final_image_bytes

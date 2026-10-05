"""Evidence-backed insight storage, retrieval, and deterministic batch updates."""

from __future__ import annotations

import hashlib
import math
import threading
from collections import defaultdict
from typing import Any, Iterable

from agent.evolution.embedding import TextEmbedder
from agent.evolution.storage import StateStore, utc_now


RETRIEVAL_STRATEGIES = frozenset({"embedding", "embedding_evidence"})


def _normalized_text(value: str) -> str:
    return " ".join(str(value).casefold().split())


class InsightStore:
    """Keep insight identity/evidence stable and derive availability at read time."""

    def __init__(
        self,
        state: StateStore,
        *,
        embedder: TextEmbedder | None = None,
        top_k: int = 4,
        consolidation_top_k: int = 4,
        semantic_threshold: float = 0.35,
        duplicate_threshold: float = 0.92,
        retrieval_strategy: str = "embedding",
    ) -> None:
        self.state = state
        self.embedder = embedder
        self.top_k = int(top_k)
        self.consolidation_top_k = int(consolidation_top_k)
        self.semantic_threshold = float(semantic_threshold)
        self.duplicate_threshold = float(duplicate_threshold)
        self.retrieval_strategy = str(retrieval_strategy).strip().casefold()
        self._embedding_cache: dict[tuple[str, int, str], list[float]] = {}
        self._embedding_lock = threading.Lock()
        if self.top_k <= 0:
            raise ValueError("top_k must be positive")
        if self.consolidation_top_k <= 0:
            raise ValueError("consolidation_top_k must be positive")
        if not -1.0 <= self.semantic_threshold <= 1.0:
            raise ValueError("memory.semantic_threshold must be between -1 and 1")
        if not -1.0 <= self.duplicate_threshold <= 1.0:
            raise ValueError("memory.duplicate_threshold must be between -1 and 1")
        if self.retrieval_strategy not in RETRIEVAL_STRATEGIES:
            choices = ", ".join(sorted(RETRIEVAL_STRATEGIES))
            raise ValueError(
                f"memory.retrieval_strategy must be one of: {choices}"
            )
        self._ensure_state_snapshot()

    def _ensure_state_snapshot(self) -> None:
        """Make one atomically replaced file the source of visible Insight state."""
        with self.state.locked():
            existing = self.state.read_data("insight_state")
            if isinstance(existing, dict) and isinstance(
                existing.get("insights"), list
            ):
                return
            legacy = []
            for stored in self.state.records("insights"):
                core = self._core(stored)
                if core is not None:
                    legacy.append(core)
            self.state.write_data(
                "insight_state",
                {
                    "schema_version": 2,
                    "revision": 0,
                    "insights": sorted(
                        legacy, key=lambda item: str(item["insight_id"])
                    ),
                    "batch_audits": {},
                    "updated_at": utc_now(),
                },
            )

    def _state_snapshot(self) -> dict[str, Any]:
        value = self.state.read_data("insight_state", {})
        if not isinstance(value, dict) or not isinstance(value.get("insights"), list):
            raise RuntimeError("Insight state snapshot is missing or invalid")
        return value

    def _visible_cores(self, envelope: dict[str, Any]) -> list[dict[str, Any]]:
        """Read the atomic snapshot plus explicit legacy/manual unversioned imports."""
        current = {
            str(item["insight_id"]): dict(item)
            for item in envelope["insights"]
            if isinstance(item, dict) and item.get("insight_id")
        }
        projection_hashes = dict(
            self.state.read_data("insight_projection_hashes", {}) or {}
        )
        for stored in self.state.records("insights"):
            core = self._core(stored)
            if core is not None:
                digest = hashlib.sha1(
                    repr(sorted(core.items())).encode("utf-8")
                ).hexdigest()
                # Matching records are internal readable projections. A different hash
                # is an explicit legacy/manual import and remains backwards compatible.
                if projection_hashes.get(str(core["insight_id"])) == digest:
                    continue
                current[str(core["insight_id"])] = core
        return sorted(current.values(), key=lambda item: str(item["insight_id"]))

    @staticmethod
    def _vectors(
        values: list[list[float]],
        *,
        expected: int,
        label: str,
    ) -> list[list[float]]:
        if len(values) != expected:
            raise ValueError(f"{label} embedder returned an unexpected vector count")
        parsed = [[float(component) for component in vector] for vector in values]
        dimensions = {len(vector) for vector in parsed}
        if not parsed or dimensions == {0} or len(dimensions) != 1:
            raise ValueError(f"{label} embedder returned invalid dimensions")
        if any(
            not math.isfinite(component)
            for vector in parsed
            for component in vector
        ):
            raise ValueError(f"{label} embedder returned non-finite values")
        return parsed

    @staticmethod
    def _cosine(left: list[float], right: list[float]) -> float:
        if len(left) != len(right):
            raise ValueError("query and Insight embedding dimensions differ")
        left_norm = math.sqrt(sum(value * value for value in left))
        right_norm = math.sqrt(sum(value * value for value in right))
        if left_norm == 0.0 or right_norm == 0.0:
            return 0.0
        return sum(a * b for a, b in zip(left, right)) / (left_norm * right_norm)

    def _semantic_vectors(
        self,
        query: str,
        rows: list[dict[str, Any]],
    ) -> tuple[list[float], dict[str, list[float]]]:
        if self.embedder is None:
            raise RuntimeError("InsightStore requires an embedder for semantic retrieval")
        with self._embedding_lock:
            query_vectors = self._vectors(
                self.embedder.embed_queries([query]),
                expected=1,
                label="query",
            )
            row_by_id = {str(row["insight_id"]): row for row in rows}
            keys = {
                str(row["insight_id"]): (
                    str(row["insight_id"]),
                    int(row.get("updated_batch", 0)),
                    str(row["text"]),
                )
                for row in rows
            }
            missing_ids = [
                insight_id
                for insight_id, key in keys.items()
                if key not in self._embedding_cache
            ]
            if missing_ids:
                documents = [
                    str(row_by_id[insight_id]["text"])
                    for insight_id in missing_ids
                ]
                vectors = self._vectors(
                    self.embedder.embed_documents(documents),
                    expected=len(documents),
                    label="document",
                )
                for insight_id, vector in zip(missing_ids, vectors):
                    self._embedding_cache[keys[insight_id]] = vector
            return query_vectors[0], {
                insight_id: self._embedding_cache[key]
                for insight_id, key in keys.items()
            }

    def _document_comparison_vectors(
        self,
        text: str,
        rows: list[dict[str, Any]],
    ) -> tuple[list[float], dict[str, list[float]]]:
        """Embed both sides as documents for symmetric near-duplicate checks."""
        if self.embedder is None:
            raise RuntimeError("InsightStore requires an embedder for semantic dedup")
        with self._embedding_lock:
            candidate = self._vectors(
                self.embedder.embed_documents([text]),
                expected=1,
                label="duplicate candidate",
            )[0]
            row_by_id = {str(row["insight_id"]): row for row in rows}
            keys = {
                str(row["insight_id"]): (
                    str(row["insight_id"]),
                    int(row.get("updated_batch", 0)),
                    str(row["text"]),
                )
                for row in rows
            }
            missing_ids = [
                insight_id
                for insight_id, key in keys.items()
                if key not in self._embedding_cache
            ]
            if missing_ids:
                vectors = self._vectors(
                    self.embedder.embed_documents(
                        [str(row_by_id[item]["text"]) for item in missing_ids]
                    ),
                    expected=len(missing_ids),
                    label="duplicate documents",
                )
                for insight_id, vector in zip(missing_ids, vectors):
                    self._embedding_cache[keys[insight_id]] = vector
            return candidate, {
                insight_id: self._embedding_cache[key]
                for insight_id, key in keys.items()
            }

    def _utility(self) -> dict[str, dict[str, Any]]:
        utility: dict[str, dict[str, Any]] = defaultdict(
            lambda: {"reward_count": 0, "reward_sum": 0.0}
        )
        for usage in self.state.records("insight_usage"):
            reward = usage.get("reward")
            if isinstance(reward, (int, float)) and not isinstance(reward, bool):
                value = utility[str(usage.get("insight_id") or "")]
                value["reward_count"] += 1
                value["reward_sum"] += float(reward)
        return utility

    @staticmethod
    def _core(stored: dict[str, Any]) -> dict[str, Any] | None:
        """Read schema-v1 records and compact legacy lesson files when encountered."""
        insight_id = str(stored.get("insight_id") or "").strip()
        text = str(stored.get("text") or "").strip()
        if not insight_id or not text:
            return None
        support = stored.get("support_episode_ids")
        if support is None:
            # A legacy record may contain task IDs under evidence. Keep them auditable
            # but unavailable until real Episode IDs support the revised memory.
            support = [
                item
                for item in stored.get("evidence", [])
                if str(item).startswith("ep_")
            ]
        return {
            "schema_version": 2,
            "insight_id": insight_id,
            "text": text,
            "support_episode_ids": list(
                dict.fromkeys(str(item) for item in support or [])
            ),
            "contradict_episode_ids": list(
                dict.fromkeys(
                    str(item) for item in stored.get("contradict_episode_ids", [])
                )
            ),
            "created_batch": int(stored.get("created_batch", 0)),
            "updated_batch": int(stored.get("updated_batch", 0)),
            "status": (
                "archived"
                if str(stored.get("status") or "active").casefold() == "archived"
                else "active"
            ),
            **(
                {"archived_batch": int(stored["archived_batch"])}
                if stored.get("archived_batch") is not None
                else {}
            ),
            **(
                {"archive_reason": str(stored["archive_reason"])}
                if str(stored.get("archive_reason") or "").strip()
                else {}
            ),
            **(
                {
                    "archive_episode_ids": list(
                        dict.fromkeys(
                            str(item)
                            for item in stored.get("archive_episode_ids", [])
                        )
                    )
                }
                if stored.get("archive_episode_ids")
                else {}
            ),
            **(
                {"replacement_insight_id": str(stored["replacement_insight_id"])}
                if str(stored.get("replacement_insight_id") or "").strip()
                else {}
            ),
        }

    def all(self, *, include_archived: bool = False) -> list[dict[str, Any]]:
        utility = self._utility()
        rows = []
        envelope = self._state_snapshot()
        for stored in self._visible_cores(envelope):
            core = self._core(stored)
            if core is None:
                continue
            if not include_archived and core["status"] == "archived":
                continue
            support_count = len(core["support_episode_ids"])
            contradiction_count = len(core["contradict_episode_ids"])
            item = {
                **core,
                "support_count": support_count,
                "contradiction_count": contradiction_count,
                "planner_visible": (
                    support_count >= 2 and support_count > contradiction_count
                ),
            }
            counts = utility[core["insight_id"]]
            item["reward_count"] = counts["reward_count"]
            if counts["reward_count"]:
                item["mean_reward"] = round(
                    counts["reward_sum"] / counts["reward_count"], 6
                )
            rows.append(item)
        return sorted(rows, key=lambda item: str(item["insight_id"]))

    def snapshot(self) -> list[dict[str, Any]]:
        """Return active Insights; callers should retrieve a local working set."""
        return [
            self._snapshot_row(item)
            for item in self.all()
        ]

    @staticmethod
    def _snapshot_row(item: dict[str, Any]) -> dict[str, Any]:
        return {
            key: item[key]
            for key in (
                "schema_version",
                "insight_id",
                "text",
                "support_episode_ids",
                "contradict_episode_ids",
                "created_batch",
                "updated_batch",
                "status",
            )
        }

    def retrieve(self, query: str, *, top_k: int | None = None) -> list[dict[str, Any]]:
        """Retrieve evidence-eligible Insights with semantic relevance as a hard gate."""
        limit = int(top_k or self.top_k)
        eligible = [row for row in self.all() if row["planner_visible"]]
        if not eligible or not str(query).strip():
            return []
        query_vector, insight_vectors = self._semantic_vectors(query, eligible)
        ranked: list[tuple[float, dict[str, Any]]] = []
        for row in eligible:
            semantic = self._cosine(
                query_vector,
                insight_vectors[str(row["insight_id"])],
            )
            if semantic < self.semantic_threshold:
                continue
            score = self._retrieval_score(semantic, row)
            item = dict(row)
            item["semantic_similarity"] = round(semantic, 6)
            item["retrieval_score"] = round(score, 6)
            ranked.append((score, item))
        ranked.sort(key=lambda pair: (-pair[0], str(pair[1]["insight_id"])))
        return [item for _, item in ranked[:limit]]

    def _retrieval_score(self, semantic: float, row: dict[str, Any]) -> float:
        """Score one semantically eligible Insight under the configured strategy."""
        if self.retrieval_strategy == "embedding":
            return semantic
        support = min(max(int(row.get("support_count", 0)), 0), 5) / 5.0
        contradiction = (
            min(max(int(row.get("contradiction_count", 0)), 0), 5) / 5.0
        )
        return 0.80 * semantic + 0.20 * support - 0.20 * contradiction

    def retrieve_for_consolidation(
        self,
        query: str,
        *,
        top_k: int | None = None,
    ) -> list[dict[str, Any]]:
        """Retrieve related active Insights, including immature hypotheses.

        Consolidation must be able to strengthen or contradict a one-Episode Insight,
        while inference deliberately sees only evidence-eligible Insights.  This is a
        separate retrieval path so those two visibility policies cannot be confused.
        """
        limit = int(top_k or self.consolidation_top_k)
        eligible = self.all()
        if not eligible or not str(query).strip():
            return []
        query_vector, insight_vectors = self._semantic_vectors(query, eligible)
        ranked: list[tuple[float, dict[str, Any]]] = []
        for row in eligible:
            semantic = self._cosine(
                query_vector,
                insight_vectors[str(row["insight_id"])],
            )
            if semantic < self.semantic_threshold:
                continue
            score = self._retrieval_score(semantic, row)
            item = dict(row)
            item["semantic_similarity"] = round(semantic, 6)
            item["retrieval_score"] = round(score, 6)
            ranked.append((score, item))
        ranked.sort(key=lambda pair: (-pair[0], str(pair[1]["insight_id"])))
        return [item for _, item in ranked[:limit]]

    def consolidation_working_set(
        self,
        queries: Iterable[str],
        *,
        top_k: int | None = None,
    ) -> list[dict[str, Any]]:
        """Union per-Episode retrieval results into one deterministic local set."""
        selected: dict[str, dict[str, Any]] = {}
        for query in queries:
            for item in self.retrieve_for_consolidation(query, top_k=top_k):
                insight_id = str(item["insight_id"])
                previous = selected.get(insight_id)
                if previous is None or float(item["retrieval_score"]) > float(
                    previous["retrieval_score"]
                ):
                    selected[insight_id] = item
        ranked = sorted(
            selected.values(),
            key=lambda item: (-float(item["retrieval_score"]), str(item["insight_id"])),
        )
        return [self._snapshot_row(item) for item in ranked]

    @staticmethod
    def render(insights: Iterable[dict[str, Any]]) -> str | None:
        lines = [f"- {item['text']}" for item in insights]
        return "\n".join(lines) or None

    def record_episode_usage(self, episode: dict[str, Any]) -> None:
        """Record one task-level utility event per retrieved insight."""
        outcome = dict(episode.get("outcome") or {})
        reward = outcome.get("reward")
        if not isinstance(reward, (int, float)) or isinstance(reward, bool):
            return
        if not 0.0 <= float(reward) <= 1.0:
            raise ValueError("reward must be between 0 and 1")
        memory = dict(episode.get("memory_context") or {})
        initial = {str(item) for item in memory.get("initial_insight_ids", [])}
        refinement = {
            str(item) for item in memory.get("refinement_insight_ids", [])
        }
        task_id = str(episode["task_id"])
        with self.state.locked():
            for insight_id in sorted(initial | refinement):
                if self.state.read_record("insights", insight_id) is None:
                    continue
                stages = []
                if insight_id in initial:
                    stages.append("initial_prompt")
                if insight_id in refinement:
                    stages.append("refinement")
                self.state.create_record(
                    "insight_usage",
                    f"{self.state.episode_key(task_id)}:{insight_id}",
                    {
                        "task_id": task_id,
                        "round_id": self.state.round_id,
                        "insight_id": insight_id,
                        "stages": stages,
                        "reward": float(reward),
                        "created_at": utc_now(),
                    },
                )

    def mature(
        self,
        *,
        min_support: int,
        min_batches: int,
        min_mean_reward: float,
    ) -> list[dict[str, Any]]:
        """Return insights eligible for initial-prompt Skill consolidation."""
        episode_batches = {
            str(item.get("episode_id")): int(item.get("batch_id", -1))
            for item in self.state.records("episodes")
        }
        mature = []
        for item in self.all():
            support_batches = {
                episode_batches[episode_id]
                for episode_id in item["support_episode_ids"]
                if episode_id in episode_batches
            }
            if int(item["support_count"]) < int(min_support):
                continue
            if int(item["support_count"]) <= int(item["contradiction_count"]):
                continue
            if len(support_batches) < int(min_batches):
                continue
            if int(item.get("reward_count", 0)) and float(
                item.get("mean_reward", 0.0)
            ) < float(min_mean_reward):
                continue
            mature.append(
                {
                    "insight_id": item["insight_id"],
                    "text": item["text"],
                    "support_count": item["support_count"],
                    "contradiction_count": item["contradiction_count"],
                    "support_batches": sorted(support_batches),
                    "updated_batch": item["updated_batch"],
                    **(
                        {"mean_reward": item["mean_reward"]}
                        if int(item.get("reward_count", 0))
                        else {}
                    ),
                }
            )
        return sorted(
            mature,
            key=lambda item: (
                int(item["support_count"]) - int(item["contradiction_count"]),
                int(item["updated_batch"]),
            ),
            reverse=True,
        )

    def apply_operations(
        self,
        *,
        batch_id: int,
        episode_ids: Iterable[str],
        operations: Iterable[dict[str, Any]],
    ) -> dict[str, Any]:
        """Validate and atomically apply model-proposed operations for one batch."""
        batch_key = f"{self.state.round_id}:{int(batch_id)}"
        allowed_episodes = {str(item) for item in episode_ids}
        with self.state.locked():
            envelope = self._state_snapshot()
            existing_audit = dict(envelope.get("batch_audits") or {}).get(batch_key)
            if isinstance(existing_audit, dict):
                return dict(existing_audit)
            current = {
                str(item["insight_id"]): dict(item)
                for item in self._visible_cores(envelope)
                if isinstance(item, dict) and item.get("insight_id")
            }
            original_ids = {
                insight_id
                for insight_id, item in current.items()
                if item.get("status", "active") == "active"
            }
            accepted: list[dict[str, Any]] = []
            rejected: list[dict[str, Any]] = []
            directions: dict[tuple[str, str], str] = {}
            revised: set[str] = set()

            for index, raw in enumerate(operations):
                operation = dict(raw) if isinstance(raw, dict) else {}
                error = self._validate_operation(
                    operation,
                    allowed_episodes=allowed_episodes,
                    original_ids=original_ids,
                )
                op = str(operation.get("op") or "").upper()
                insight_id = str(operation.get("insight_id") or "")
                target_insight_id = str(
                    operation.get("target_insight_id") or ""
                )
                cited = list(
                    dict.fromkeys(
                        str(item) for item in operation.get("episode_ids", [])
                    )
                )
                direction = "contradict" if op == "CONTRADICT" else "support"
                if error is None and insight_id:
                    if current[insight_id].get("status", "active") != "active":
                        error = "referenced insight is no longer active"
                    if op == "REVISE" and insight_id in revised:
                        error = "an insight may be revised at most once per batch"
                    for episode_id in cited:
                        previous = directions.get((insight_id, episode_id))
                        if previous is not None and previous != direction:
                            error = "one episode cannot support and contradict one insight"
                            break
                if error is None and op == "MERGE":
                    merged_ids = [
                        str(item) for item in operation.get("merged_insight_ids", [])
                    ]
                    involved = [target_insight_id, *merged_ids]
                    if target_insight_id in revised:
                        error = (
                            "an insight may be revised or merged at most once per batch"
                        )
                    if any(
                        current[item].get("status", "active") != "active"
                        for item in involved
                    ):
                        error = "MERGE references an insight that is no longer active"
                if error is None and op == "ARCHIVE":
                    replacement = str(
                        operation.get("replacement_insight_id") or ""
                    )
                    if replacement and current[replacement].get(
                        "status", "active"
                    ) != "active":
                        error = "ARCHIVE replacement is no longer active"
                if error is not None:
                    rejected.append(
                        {"index": index, "operation": operation, "error": error}
                    )
                    continue

                if op == "ADD":
                    duplicate, match_kind, similarity = self._duplicate_match(
                        str(operation["text"]),
                        current,
                    )
                    if duplicate is None:
                        insight_id = self._new_id(str(operation["text"]), current)
                        now_record = {
                            "schema_version": 2,
                            "insight_id": insight_id,
                            "text": str(operation["text"]).strip(),
                            "support_episode_ids": cited,
                            "contradict_episode_ids": [],
                            "created_batch": int(batch_id),
                            "updated_batch": int(batch_id),
                            "status": "active",
                        }
                        current[insight_id] = now_record
                        applied = {**operation, "insight_id": insight_id}
                    else:
                        insight_id = duplicate
                        current[insight_id]["support_episode_ids"] = list(
                            dict.fromkeys(
                                [*current[insight_id]["support_episode_ids"], *cited]
                            )
                        )
                        current[insight_id]["updated_batch"] = int(batch_id)
                        applied = {
                            **operation,
                            "insight_id": insight_id,
                            "resolved_as": "SUPPORT",
                            "duplicate_match": match_kind,
                            **(
                                {"duplicate_similarity": round(similarity, 6)}
                                if similarity is not None
                                else {}
                            ),
                        }
                elif op == "MERGE":
                    merged_ids = list(
                        dict.fromkeys(
                            str(item)
                            for item in operation.get("merged_insight_ids", [])
                        )
                    )
                    target = current[target_insight_id]
                    support = [*target["support_episode_ids"], *cited]
                    contradictions = list(target["contradict_episode_ids"])
                    for merged_id in merged_ids:
                        source = current[merged_id]
                        support.extend(source["support_episode_ids"])
                        contradictions.extend(source["contradict_episode_ids"])
                        source.update(
                            {
                                "schema_version": 2,
                                "status": "archived",
                                "archived_batch": int(batch_id),
                                "archive_reason": (
                                    f"Merged into {target_insight_id}."
                                ),
                                "archive_episode_ids": cited,
                                "replacement_insight_id": target_insight_id,
                                "updated_batch": int(batch_id),
                            }
                        )
                    contradiction_set = set(contradictions)
                    target.update(
                        {
                            "schema_version": 2,
                            "text": str(operation["text"]).strip(),
                            "support_episode_ids": list(
                                dict.fromkeys(
                                    item for item in support if item not in contradiction_set
                                )
                            ),
                            "contradict_episode_ids": list(
                                dict.fromkeys(contradictions)
                            ),
                            "updated_batch": int(batch_id),
                            "status": "active",
                        }
                    )
                    revised.add(target_insight_id)
                    insight_id = target_insight_id
                    applied = dict(operation)
                elif op == "ARCHIVE":
                    target = current[insight_id]
                    target.update(
                        {
                            "schema_version": 2,
                            "status": "archived",
                            "archived_batch": int(batch_id),
                            "archive_reason": str(operation["reason"]).strip(),
                            "archive_episode_ids": cited,
                            "updated_batch": int(batch_id),
                        }
                    )
                    replacement = str(
                        operation.get("replacement_insight_id") or ""
                    )
                    if replacement:
                        target["replacement_insight_id"] = replacement
                    applied = dict(operation)
                else:
                    target = current[insight_id]
                    evidence_field = (
                        "contradict_episode_ids"
                        if op == "CONTRADICT"
                        else "support_episode_ids"
                    )
                    target[evidence_field] = list(
                        dict.fromkeys([*target[evidence_field], *cited])
                    )
                    if op == "REVISE":
                        target["text"] = str(operation["text"]).strip()
                        revised.add(insight_id)
                    target["updated_batch"] = int(batch_id)
                    applied = dict(operation)

                direction_target = target_insight_id if op == "MERGE" else insight_id
                if op not in {"ARCHIVE"}:
                    for episode_id in cited:
                        directions[(direction_target, episode_id)] = direction
                accepted.append(applied)

            # Record files remain readable projections. The collection snapshot below is
            # the visibility boundary, so a crash during these writes leaves readers on
            # the previous complete revision.
            next_revision = int(envelope.get("revision", 0)) + 1
            projection_hashes = {
                insight_id: hashlib.sha1(
                    repr(sorted(value.items())).encode("utf-8")
                ).hexdigest()
                for insight_id, value in current.items()
            }
            # Publish expected hashes before projection files. If a crash occurs before
            # the final collection snapshot, old projections merely restate old state
            # while completed future projections remain hidden.
            self.state.write_data("insight_projection_hashes", projection_hashes)
            for insight_id, value in current.items():
                before = self.state.read_record("insights", insight_id)
                if before != value:
                    self.state.write_record("insights", insight_id, value)
            audit = {
                "schema_version": 2,
                "batch_id": int(batch_id),
                "episode_ids": sorted(allowed_episodes),
                "accepted_operations": accepted,
                "rejected_operations": rejected,
                "touched_insight_ids": list(
                    dict.fromkeys(
                        str(insight_id)
                        for item in accepted
                        for insight_id in self._operation_insight_ids(item)
                    )
                ),
                "created_at": utc_now(),
            }
            self.state.write_record("insight_batches", batch_key, audit)
            batch_audits = dict(envelope.get("batch_audits") or {})
            batch_audits[batch_key] = audit
            self.state.write_data(
                "insight_state",
                {
                    "schema_version": 2,
                    "revision": next_revision,
                    "insights": sorted(
                        current.values(), key=lambda item: str(item["insight_id"])
                    ),
                    "batch_audits": batch_audits,
                    "updated_at": utc_now(),
                },
            )
            return audit

    @staticmethod
    def _validate_operation(
        operation: dict[str, Any],
        *,
        allowed_episodes: set[str],
        original_ids: set[str],
    ) -> str | None:
        op = str(operation.get("op") or "").upper()
        if op not in {
            "ADD",
            "SUPPORT",
            "REVISE",
            "CONTRADICT",
            "MERGE",
            "ARCHIVE",
        }:
            return "unsupported operation"
        episode_ids = operation.get("episode_ids")
        if not isinstance(episode_ids, list) or not episode_ids:
            return "episode_ids must be a non-empty list"
        cited = {str(item) for item in episode_ids}
        if not cited <= allowed_episodes:
            return "operation cites an episode outside the current batch"
        text = str(operation.get("text") or "").strip()
        if op in {"ADD", "REVISE", "MERGE"} and not text:
            return f"{op} requires non-empty text"
        if op in {"SUPPORT", "CONTRADICT", "ARCHIVE"} and text:
            return f"{op} cannot modify insight text"
        if op == "MERGE":
            target = str(operation.get("target_insight_id") or "")
            merged = [
                str(item) for item in operation.get("merged_insight_ids", [])
            ]
            if target not in original_ids:
                return "MERGE target insight does not exist in the local collection"
            if not merged:
                return "MERGE requires at least one merged_insight_id"
            if target in merged or len(set(merged)) != len(merged):
                return "MERGE insight references must be distinct"
            if not set(merged) <= original_ids:
                return "MERGE source insight does not exist in the local collection"
            return None
        if op != "ADD" and str(operation.get("insight_id") or "") not in original_ids:
            return "referenced insight does not exist in the frozen collection"
        if op in {"CONTRADICT", "ARCHIVE"} and not str(
            operation.get("reason") or ""
        ).strip():
            return f"{op} requires a reason"
        if op == "ARCHIVE":
            replacement = str(operation.get("replacement_insight_id") or "")
            if replacement and replacement not in original_ids:
                return "ARCHIVE replacement does not exist in the local collection"
            if replacement == str(operation.get("insight_id") or ""):
                return "ARCHIVE replacement must be a different Insight"
        return None

    def _duplicate_match(
        self,
        text: str,
        current: dict[str, dict[str, Any]],
    ) -> tuple[str | None, str | None, float | None]:
        active = {
            insight_id: item
            for insight_id, item in current.items()
            if item.get("status", "active") == "active"
        }
        exact = next(
            (
                insight_id
                for insight_id, item in active.items()
                if _normalized_text(item["text"]) == _normalized_text(text)
            ),
            None,
        )
        if exact is not None:
            return exact, "exact", None
        if not active or self.embedder is None:
            return None, None, None
        rows = list(active.values())
        query_vector, insight_vectors = self._document_comparison_vectors(text, rows)
        ranked = sorted(
            (
                (
                    self._cosine(query_vector, insight_vectors[insight_id]),
                    insight_id,
                )
                for insight_id in active
            ),
            reverse=True,
        )
        similarity, insight_id = ranked[0]
        if similarity >= self.duplicate_threshold:
            return insight_id, "semantic", similarity
        return None, None, similarity

    @staticmethod
    def _operation_insight_ids(operation: dict[str, Any]) -> list[str]:
        return list(
            dict.fromkeys(
                str(item)
                for item in (
                    operation.get("insight_id"),
                    operation.get("target_insight_id"),
                    *list(operation.get("merged_insight_ids") or []),
                )
                if item
            )
        )

    def _new_id(
        self,
        text: str,
        current: dict[str, dict[str, Any]],
    ) -> str:
        digest = hashlib.sha1(
            f"{self.state.run_id}|{_normalized_text(text)}".encode("utf-8")
        ).hexdigest()[:12]
        base = f"ins_{digest}"
        candidate = base
        suffix = 2
        while candidate in current:
            candidate = f"{base}_{suffix}"
            suffix += 1
        return candidate


__all__ = ["InsightStore"]

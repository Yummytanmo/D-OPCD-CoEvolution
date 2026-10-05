"""Small blocking pool for independent evaluator instances."""

from __future__ import annotations

import logging
from queue import Queue
import time
from typing import Callable, Generic, Iterable, TypeVar


LOGGER = logging.getLogger("evaluation.feedback_service")
EvaluatorT = TypeVar("EvaluatorT")
ResultT = TypeVar("ResultT")


class EvaluatorPool(Generic[EvaluatorT]):
    """Allow safe concurrent inference without sharing one model instance."""

    def __init__(self, benchmark: str, evaluators: Iterable[EvaluatorT]) -> None:
        resources = tuple(evaluators)
        if not resources:
            raise ValueError("EvaluatorPool requires at least one evaluator")
        self.benchmark = benchmark
        self._resources: Queue[tuple[int, EvaluatorT]] = Queue(len(resources))
        for worker_id, evaluator in enumerate(resources):
            self._resources.put((worker_id, evaluator))

    @property
    def concurrency(self) -> int:
        return self._resources.maxsize

    def run(self, operation: Callable[[EvaluatorT], ResultT]) -> ResultT:
        waiting_started = time.monotonic()
        worker_id, evaluator = self._resources.get()
        LOGGER.info(
            "evaluator worker acquired benchmark=%s worker=%d wait_seconds=%.3f",
            self.benchmark,
            worker_id,
            time.monotonic() - waiting_started,
        )
        try:
            return operation(evaluator)
        finally:
            self._resources.put((worker_id, evaluator))
            LOGGER.info(
                "evaluator worker released benchmark=%s worker=%d",
                self.benchmark,
                worker_id,
            )

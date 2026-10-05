"""Construct a feedback agent for one paper benchmark."""

from typing import Any

from evaluation.feedback_agents.base import FeedbackAgent
from evaluation.feedback_agents.geneval import GenEvalFeedbackAgent
from evaluation.feedback_agents.geneval2 import GenEval2FeedbackAgent
from evaluation.feedback_agents.r2ibench import R2IBenchFeedbackAgent
from evaluation.feedback_agents.wise import WiseFeedbackAgent


def create_feedback_agent(benchmark: str, **kwargs: Any) -> FeedbackAgent:
    classes = {
        "geneval": GenEvalFeedbackAgent,
        "geneval2": GenEval2FeedbackAgent,
        "wise": WiseFeedbackAgent,
        "r2ibench": R2IBenchFeedbackAgent,
    }
    try:
        agent_class = classes[benchmark.strip().casefold()]
    except KeyError as error:
        raise ValueError(f"Unsupported feedback benchmark: {benchmark}") from error
    return agent_class(**kwargs)

"""Central prompt registry for every feedback agent.

Evaluator outputs are private grounding evidence.  These prompts require the
verbalizer to turn that evidence into a direct description of the submitted
image without exposing the evaluation machinery or suggesting future changes.
"""

from __future__ import annotations

import json
from typing import Any


SYSTEM_PROMPT = """You are a post-task image feedback agent.

Evaluate only the single submitted image against the current task. Write one
concise paragraph in English as a direct human visual assessment.

The supplied evaluator evidence is private grounding. Never reveal, quote, or
name any evaluator, model, detector, metric, score, probability,
confidence, threshold, recognized-output log, or other evaluation procedure.

State only what the image visibly presents, fails to present, renders clearly,
renders ambiguously, or renders inconsistently with the current request. Do not
give advice, fixes, prompt changes, reusable lessons, or comments about later
attempts or future tasks. Do not use phrases such as “should”, “try”,
“next time”, or “could be improved”. Do not add a heading, label, bullet list,
or preamble.
"""


BENCHMARK_INSTRUCTIONS = {
    "geneval": """Use the private compositional evidence to assess only the
current image's objects, counts, colors, exclusions, and spatial relations.
Express failures as direct visual facts, for example that only two dogs are
clearly visible or that one object appears on the wrong side of another.""",
    "geneval2": """Use the private question-level compositional evidence to
assess only the current image's objects, counts, attributes, actions, and spatial
relations. Express failures only as direct visual facts about the submitted
image.""",
    "r2ibench": """Use the private checklist evidence to assess whether the
current image visibly realizes the requested concepts and relationships.
Describe missing or unclear requirements as direct visual facts. Do not reveal
the checklist, its criteria, weights, or the judge's output.""",
    "wise": """Use the private knowledge-based result to describe whether the
current image visibly realizes the intended meaning. Do not quote the reference
explanation or reveal the judge's output.""",
}


def build_feedback_prompt(
    benchmark: str,
    task_prompt: str,
    private_evidence: dict[str, Any],
) -> str:
    """Build the only user prompt used by feedback verbalizers."""
    try:
        benchmark_instruction = BENCHMARK_INSTRUCTIONS[benchmark]
    except KeyError as exc:
        raise ValueError(f"Unsupported feedback benchmark: {benchmark}") from exc

    evidence_json = json.dumps(
        private_evidence,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )
    return (
        f"Current task request:\n{task_prompt}\n\n"
        "Submitted image:\n<image>\n\n"
        f"Benchmark-specific instruction:\n{benchmark_instruction}\n\n"
        "Private evaluator evidence (grounding only; never expose its source, "
        f"fields, or numeric values):\n{evidence_json}\n\n"
        "Return only the task-local visual assessment paragraph."
    )

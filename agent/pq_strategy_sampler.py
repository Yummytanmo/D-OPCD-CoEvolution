"""Content-independent, deterministic strategy selection. No API calls on import."""
import hashlib

BASE = "Produce the prompt for the first image-generation attempt. Use the supplied Skills as guidance for fulfilling the original request. Apply guidance only where it is relevant to that request, and preserve the request's meaning and requirements. If a Skill conflicts with the original request, the request takes priority. Return only the resulting prompt. Returning the original request unchanged is allowed."
STRATEGIES = {
    'local_edit': "Keep the original request's wording and order where possible. Incorporate applicable Skill guidance through local edits or additions.",
    'restate': "Write the resulting prompt afresh in your own words rather than retaining the original sentence structure. Express the same request together with applicable Skill guidance.",
    'integrate': "Weave applicable Skill guidance directly into the description of the request, as one coherent image-generation prompt.",
    'separate': "State the original request first, then express applicable Skill guidance in separate sentences within the same final image-generation prompt. Do not output explanations of your work.",
}
def strategy_for(sample_key: str, seed: int = 20260920) -> str:
    digest = hashlib.sha256(f'pq-strategy-v1:{seed}:{sample_key}'.encode()).digest()
    return tuple(STRATEGIES)[int.from_bytes(digest[:8], 'big') % len(STRATEGIES)]

def build_prompt(original_prompt: str, skill_rules: str | None, strategy: str | None = None) -> str:
    if strategy is not None and strategy not in STRATEGIES:
        raise ValueError('Unknown strategy')
    blocks = [BASE]
    if skill_rules and skill_rules.strip():
        blocks.append('### Initial-Prompt Skill Instructions\n' + skill_rules.strip())
    blocks.append('### Original Prompt\n' + original_prompt.strip())
    if strategy is not None:
        blocks.append('### Expression strategy\n' + STRATEGIES[strategy] +
                      '\nThis strategy changes expression, not the requested content. It does not authorize dropping applicable guidance or adding new requirements merely to create variety.')
    blocks.append('Return ONLY the final enhanced prompt.')
    return '\n\n'.join(blocks)

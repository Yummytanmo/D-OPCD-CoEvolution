from __future__ import annotations

from typing import Iterable

import torch


def render_chat_prompts(tokenizer, prompts: Iterable[str]) -> list[str]:
    rendered: list[str] = []
    for prompt in prompts:
        messages = [{"role": "user", "content": str(prompt)}]
        try:
            text = tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
                enable_thinking=True,
            )
        except TypeError:
            text = tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
            )
        rendered.append(text)
    return rendered


@torch.no_grad()
def encode_prompts(text_encoder, tokenizer, prompts, device, max_sequence_length: int, dtype=None):
    rendered = render_chat_prompts(tokenizer, prompts)
    unpadded = tokenizer(rendered, padding=False, truncation=False, return_attention_mask=False)
    lengths = [len(ids) for ids in unpadded["input_ids"]]
    too_long = [length for length in lengths if length > max_sequence_length]
    if too_long:
        raise ValueError(
            f"Prompt exceeds max_sequence_length={max_sequence_length}: "
            f"max={max(too_long)}, lengths={lengths}"
        )
    inputs = tokenizer(rendered, padding="longest", truncation=False, return_tensors="pt")
    input_ids = inputs.input_ids.to(device)
    attention_mask = inputs.attention_mask.to(device).bool()
    hidden_states = text_encoder(
        input_ids=input_ids,
        attention_mask=attention_mask,
        output_hidden_states=True,
        use_cache=False,
    ).hidden_states[-2]
    if dtype is not None:
        hidden_states = hidden_states.to(dtype=dtype)
    embeddings = [hidden_states[index][attention_mask[index]] for index in range(len(rendered))]
    return embeddings, lengths

from __future__ import annotations

from collections.abc import Iterable
import json
from pathlib import Path

import torch
from tokenizers import AddedToken
from transformers import Qwen2Tokenizer


PROMPT_TEMPLATE = (
    "<|im_start|>system\n"
    "Describe the image by detailing the color, shape, size, texture, quantity, text, "
    "spatial relationships of the objects and background:<|im_end|>\n"
    "<|im_start|>user\n{}<|im_end|>\n"
    "<|im_start|>assistant\n"
)
PROMPT_TEMPLATE_ENCODE_START_IDX = 34
QWEN_IMAGE_TOKENIZER_MAX_LENGTH = 1024


def load_qwen_image_tokenizer(model_path: str | Path) -> Qwen2Tokenizer:
    """Load Qwen's BPE exactly, including a Transformers 5 RC no-tokenizer.json fallback."""
    root = Path(model_path).expanduser().resolve()
    tokenizer_dir = root / "tokenizer" if (root / "tokenizer").is_dir() else root
    tokenizer_json = tokenizer_dir / "tokenizer.json"
    if tokenizer_json.is_file():
        return Qwen2Tokenizer.from_pretrained(tokenizer_dir, local_files_only=True)

    vocab_path = tokenizer_dir / "vocab.json"
    merges_path = tokenizer_dir / "merges.txt"
    config_path = tokenizer_dir / "tokenizer_config.json"
    for required in (vocab_path, merges_path, config_path):
        if not required.is_file():
            raise FileNotFoundError(required)
    with vocab_path.open("r", encoding="utf-8") as handle:
        vocab = json.load(handle)
    with merges_path.open("r", encoding="utf-8") as handle:
        merges = [
            tuple(line.strip().split())
            for line in handle
            if line.strip() and not line.startswith("#")
        ]
    if any(len(pair) != 2 for pair in merges):
        raise ValueError(f"Invalid BPE merge in {merges_path}")
    with config_path.open("r", encoding="utf-8") as handle:
        config = json.load(handle)

    added_decoder = config.pop("added_tokens_decoder", {})
    special_values = {
        key: config.pop(key, None)
        for key in (
            "unk_token",
            "bos_token",
            "eos_token",
            "pad_token",
            "additional_special_tokens",
        )
    }
    tokenizer = Qwen2Tokenizer(
        vocab=vocab,
        merges=merges,
        unk_token=None,
        bos_token=None,
        eos_token=None,
        pad_token=None,
        additional_special_tokens=[],
        **config,
    )
    added_items = sorted((int(token_id), spec) for token_id, spec in added_decoder.items())
    expected_id = len(vocab)
    for token_id, _ in added_items:
        if token_id != expected_id:
            raise ValueError(
                f"Added token IDs must be contiguous from {len(vocab)}; got {token_id}"
            )
        expected_id += 1
    added_tokens = [
        AddedToken(
            spec["content"],
            special=bool(spec.get("special", False)),
            lstrip=bool(spec.get("lstrip", False)),
            rstrip=bool(spec.get("rstrip", False)),
            single_word=bool(spec.get("single_word", False)),
            normalized=bool(spec.get("normalized", True)),
        )
        for _, spec in added_items
    ]
    if tokenizer.add_tokens(added_tokens) != len(added_tokens):
        raise ValueError("Failed to restore Qwen added tokens with stable IDs")
    for key in ("unk_token", "bos_token", "eos_token", "pad_token"):
        if special_values[key] is not None:
            setattr(tokenizer, key, special_values[key])
    if special_values["additional_special_tokens"]:
        tokenizer.additional_special_tokens = special_values["additional_special_tokens"]

    for token, expected in (
        ("<|endoftext|>", 151643),
        ("<|im_start|>", 151644),
        ("<|im_end|>", 151645),
    ):
        actual = tokenizer.convert_tokens_to_ids(token)
        if actual != expected:
            raise ValueError(f"Qwen special token {token} has ID {actual}, expected {expected}")
    return tokenizer


def render_qwen_image_prompts(prompts: Iterable[str]) -> list[str]:
    return [PROMPT_TEMPLATE.format(str(prompt)) for prompt in prompts]


def qwen_image_prompt_token_lengths(tokenizer, prompts: Iterable[str]) -> list[int]:
    rendered = render_qwen_image_prompts(prompts)
    tokenized = tokenizer(
        rendered,
        padding=False,
        truncation=False,
        return_attention_mask=False,
    )["input_ids"]
    return [max(0, len(input_ids) - PROMPT_TEMPLATE_ENCODE_START_IDX) for input_ids in tokenized]


@torch.no_grad()
def encode_qwen_image_prompts(
    text_encoder,
    tokenizer,
    prompts,
    device,
    max_sequence_length: int,
    dtype=None,
):
    if not 0 < max_sequence_length <= QWEN_IMAGE_TOKENIZER_MAX_LENGTH:
        raise ValueError(
            f"max_sequence_length must be in [1, {QWEN_IMAGE_TOKENIZER_MAX_LENGTH}]"
        )
    rendered = render_qwen_image_prompts(prompts)
    lengths = qwen_image_prompt_token_lengths(tokenizer, prompts)
    too_long = [length for length in lengths if length > max_sequence_length]
    if too_long:
        raise ValueError(
            f"Prompt exceeds max_sequence_length={max_sequence_length}: "
            f"max={max(too_long)}, lengths={lengths}"
        )
    inputs = tokenizer(
        rendered,
        max_length=max_sequence_length + PROMPT_TEMPLATE_ENCODE_START_IDX,
        padding="longest",
        truncation=False,
        return_tensors="pt",
    )
    input_ids = inputs.input_ids.to(device)
    attention_mask = inputs.attention_mask.to(device)
    hidden_states = text_encoder(
        input_ids=input_ids,
        attention_mask=attention_mask,
        output_hidden_states=True,
        use_cache=False,
    ).hidden_states[-1]

    valid_lengths = attention_mask.bool().sum(dim=1)
    selected = hidden_states[attention_mask.bool()]
    split_hidden = torch.split(selected, valid_lengths.tolist(), dim=0)
    split_hidden = [value[PROMPT_TEMPLATE_ENCODE_START_IDX:] for value in split_hidden]
    max_length = max(value.shape[0] for value in split_hidden)
    prompt_embeds = torch.stack(
        [
            torch.cat(
                [value, value.new_zeros(max_length - value.shape[0], value.shape[1])],
                dim=0,
            )
            for value in split_hidden
        ],
        dim=0,
    )
    prompt_mask = torch.stack(
        [
            torch.cat(
                [
                    torch.ones(value.shape[0], device=value.device, dtype=torch.long),
                    torch.zeros(max_length - value.shape[0], device=value.device, dtype=torch.long),
                ],
                dim=0,
            )
            for value in split_hidden
        ],
        dim=0,
    )
    if dtype is not None:
        prompt_embeds = prompt_embeds.to(dtype=dtype)
    return prompt_embeds, prompt_mask, lengths

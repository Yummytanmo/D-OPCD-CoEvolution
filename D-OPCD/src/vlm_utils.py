"""Qwen3-VL teacher context embeds for the D-OPSD image-referenced teacher.

Ported from the pinned upstream D-OPSD release
(third_party/D-OPSD/z-image-turbo_self-distill-vlm/vlm_utils.py,
git@github.com:vvvvvjdy/D-OPSD.git @ b9dfa2d).  The teacher prompt embeds are
produced by a Qwen3-VL model whose language model is initialized from the
Z-Image text encoder weights, so the embeds match the Z-Image text axis.
"""

from __future__ import annotations

from typing import List, Optional, Union

import torch


def load_matching_state_dict(target_module, source_state_dict, verbose=True):
    target_state = target_module.state_dict()
    matched_state = {}

    for k, v in source_state_dict.items():
        if k in target_state and target_state[k].shape == v.shape:
            matched_state[k] = v

    missing_keys, unexpected_keys = target_module.load_state_dict(matched_state, strict=False)

    if verbose:
        print(f"Matched keys: {len(matched_state)} / {len(target_state)}")
        print(f"Missing keys: {len(missing_keys)}")
        print(f"Unexpected keys: {len(unexpected_keys)}")
        if len(missing_keys) > 0:
            print("Missing sample:", missing_keys[:20])
        if len(unexpected_keys) > 0:
            print("Unexpected sample:", unexpected_keys[:20])

    return missing_keys, unexpected_keys


def _extract_masked_hidden(hidden_states: torch.Tensor, attention_mask: torch.Tensor):
    """
    hidden_states: [B, L, D]
    attention_mask: [B, L]
    return: List[[Li, D]]
    """
    results = []
    for hs, mask in zip(hidden_states, attention_mask):
        valid = mask.bool()
        results.append(hs[valid])
    return results


@torch.no_grad()
def get_qwen3vl_zimage_prompt_embeds(
    vl_model,
    processor,
    prompts: Union[str, List[str]],
    images=None,
    device=None,
    dtype: Optional[torch.dtype] = None,
    max_sequence_length: int = 512,
    num_images_per_prompt: int = 1,
    hidden_state_layer: int = -2,
    add_generation_prompt: bool = True,
    system_prompt: Optional[str] = None,
    use_system_prompt: bool = False,
):

    if device is None:
        device = vl_model.device
    dtype = dtype or vl_model.dtype

    if isinstance(prompts, str):
        prompts = [prompts]

    has_images = images is not None
    if has_images:
        if not isinstance(images, list):
            images = [images]
        if len(images) != len(prompts):
            raise ValueError(f"`images` and `prompts` must have the same length: {len(images)} vs {len(prompts)}")
    else:
        images = [None] * len(prompts)

    def _is_tensor_image(x):
        return isinstance(x, torch.Tensor)

    split_hidden_states = []

    for text, image in zip(prompts, images):
        user_content = []

        if image is not None:
            user_content.append({"type": "image", "image": image})

        user_content.append({"type": "text", "text": text})

        if use_system_prompt and system_prompt is not None:
            conv = [
                {
                    "role": "system",
                    "content": [{"type": "text", "text": system_prompt}],
                },
                {
                    "role": "user",
                    "content": user_content,
                },
            ]
        else:
            conv = [
                {
                    "role": "user",
                    "content": user_content,
                }
            ]

        text_input = processor.apply_chat_template(
            conv,
            tokenize=False,
            add_generation_prompt=add_generation_prompt,
        )

        if image is not None:
            if _is_tensor_image(image):
                if image.numel() > 0 and image.max() <= 1.0:
                    do_rescale = False
            else:
                do_rescale = True
            model_inputs = processor(
                text=[text_input],
                images=[image],
                padding=True,
                truncation=True,
                return_tensors="pt",
                do_rescale=do_rescale,
            )
        else:
            model_inputs = processor(
                text=[text_input],
                padding=True,
                truncation=True,
                return_tensors="pt",
            )

        model_inputs = {
            k: v.to(device) if hasattr(v, "to") else v
            for k, v in model_inputs.items()
        }

        outputs = vl_model(
            **model_inputs,
            output_hidden_states=True,
            return_dict=True,
            use_cache=False,
        )

        prompt_embeds_full = outputs.hidden_states[hidden_state_layer].to(
            device=device, dtype=dtype
        )  # [1, L, D]

        attention_mask = model_inputs["attention_mask"]  # [1, L]

        # Remove padding and keep variable-length outputs
        sample_hidden = _extract_masked_hidden(prompt_embeds_full, attention_mask)
        e = sample_hidden[0][:max_sequence_length]

        split_hidden_states.append(e)

    # Repeat according to num_images_per_prompt
    if num_images_per_prompt > 1:
        split_hidden_states = [e for e in split_hidden_states for _ in range(num_images_per_prompt)]

    return split_hidden_states

"""Reusable GenEval2 Soft-TIFA runtime backed by a local Qwen3-VL model."""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any


NUMBER_WORDS = {
    "one": "1",
    "two": "2",
    "three": "3",
    "four": "4",
    "five": "5",
    "six": "6",
    "seven": "7",
    "eight": "8",
    "nine": "9",
    "ten": "10",
}


def answer_variants(question: str, answer: str) -> list[str]:
    """Match the answer spellings used by the official GenEval2 evaluator."""
    if question.startswith("How many"):
        numeric = NUMBER_WORDS.get(answer.casefold(), "other")
        return list(
            dict.fromkeys(
                (
                    answer,
                    answer.capitalize(),
                    f" {answer}",
                    f" {answer.capitalize()}",
                    numeric,
                    f" {numeric}",
                )
            )
        )
    return ["Yes", "yes", " yes", " Yes"]


def geometric_mean(values: list[float]) -> float:
    if not values:
        raise ValueError("cannot compute a geometric mean without values")
    if any(value < 0 for value in values):
        raise ValueError("geometric mean inputs must be non-negative")
    if any(value == 0 for value in values):
        return 0.0
    return math.exp(sum(math.log(value) for value in values) / len(values))


class GenEval2Runtime:
    """Load Qwen3-VL once and score one generated image at a time."""

    def __init__(
        self,
        *,
        model_path: str | Path,
        device_map: str = "auto",
        dtype: str = "bfloat16",
        local_files_only: bool = True,
    ) -> None:
        self.model_path = str(model_path)
        self.device_map = str(device_map)
        self.dtype_name = str(dtype)
        self.local_files_only = bool(local_files_only)
        self._torch = None
        self._processor = None
        self._model = None

    def load(self) -> None:
        if self._model is not None:
            return
        import torch
        from transformers import AutoProcessor, Qwen3VLForConditionalGeneration

        try:
            dtype = getattr(torch, self.dtype_name)
        except AttributeError as error:
            raise ValueError(f"unsupported GenEval2 dtype: {self.dtype_name}") from error
        processor = AutoProcessor.from_pretrained(
            self.model_path,
            local_files_only=self.local_files_only,
        )
        model = Qwen3VLForConditionalGeneration.from_pretrained(
            self.model_path,
            dtype=dtype,
            device_map=self.device_map,
            local_files_only=self.local_files_only,
        )
        model.eval()
        self._torch = torch
        self._processor = processor
        self._model = model

    @property
    def loaded(self) -> bool:
        return self._model is not None

    def _score_question(
        self,
        *,
        question: str,
        answer: str,
        image_path: Path,
    ) -> dict[str, Any]:
        self.load()
        torch = self._torch
        processor = self._processor
        model = self._model
        if torch is None or processor is None or model is None:
            raise RuntimeError("GenEval2 evaluator failed to initialize")
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": str(image_path)},
                    {
                        "type": "text",
                        "text": f"{question} Answer in one word.",
                    },
                ],
            }
        ]
        inputs = processor.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True,
            return_dict=True,
            return_tensors="pt",
        )
        inputs = inputs.to(model.device)
        with torch.inference_mode():
            outputs = model.generate(
                **inputs,
                max_new_tokens=1,
                do_sample=False,
                output_scores=True,
                return_dict_in_generate=True,
            )
        probabilities = torch.nn.functional.softmax(outputs.scores[0], dim=-1)
        token_ids: list[int] = []
        for variant in answer_variants(question, answer):
            encoded = processor.tokenizer.encode(variant)
            if encoded:
                token_ids.append(int(encoded[0]))
        if not token_ids:
            raise RuntimeError(f"GenEval2 answer did not tokenize: {answer!r}")
        probability = min(
            1.0,
            sum(float(probabilities[0, token_id].item()) for token_id in token_ids),
        )
        predicted_id = int(torch.argmax(probabilities, dim=-1).item())
        predicted = processor.batch_decode([[predicted_id]])[0]
        return {
            "question": question,
            "answer": answer,
            "prediction": predicted,
            "answer_probability": probability,
        }

    def evaluate_image(
        self,
        image_path: str | Path,
        metadata: dict[str, Any],
    ) -> dict[str, Any]:
        resolved_image = Path(image_path).expanduser().resolve()
        if not resolved_image.is_file():
            raise FileNotFoundError(resolved_image)
        raw_vqa = metadata.get("vqa_list")
        raw_skills = metadata.get("skills")
        if not isinstance(raw_vqa, list) or not raw_vqa:
            raise ValueError("GenEval2 metadata requires a non-empty vqa_list")
        if not isinstance(raw_skills, list) or len(raw_skills) != len(raw_vqa):
            raise ValueError("GenEval2 metadata has misaligned skills and vqa_list")

        questions: list[dict[str, Any]] = []
        for index, item in enumerate(raw_vqa):
            if (
                not isinstance(item, list)
                or len(item) != 2
                or not all(isinstance(value, str) and value for value in item)
            ):
                raise ValueError(f"invalid GenEval2 VQA item at index {index}")
            question = self._score_question(
                question=item[0],
                answer=item[1],
                image_path=resolved_image,
            )
            question["skill"] = str(raw_skills[index])
            questions.append(question)

        scores = [float(item["answer_probability"]) for item in questions]
        return {
            "atom_count": int(metadata.get("atom_count") or len(questions)),
            "question_count": len(questions),
            "soft_tifa_am": sum(scores) / len(scores),
            "soft_tifa_gm": geometric_mean(scores),
            "questions": questions,
        }


__all__ = [
    "GenEval2Runtime",
    "NUMBER_WORDS",
    "answer_variants",
    "geometric_mean",
]

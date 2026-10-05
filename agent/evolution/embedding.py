"""Text embedding providers used by semantic Insight retrieval."""

from __future__ import annotations

import threading
from collections.abc import Sequence
from typing import Any, Protocol


class TextEmbedder(Protocol):
    """Keep query and document encoding distinct for retrieval-tuned models."""

    def embed_queries(self, texts: Sequence[str]) -> list[list[float]]: ...

    def embed_documents(self, texts: Sequence[str]) -> list[list[float]]: ...


class TransformerTextEmbedder:
    """Lazy, thread-safe Hugging Face encoder with normalized output vectors."""

    def __init__(
        self,
        model: str,
        *,
        device: str = "cpu",
        pooling: str = "cls",
        batch_size: int = 32,
        max_length: int = 512,
        query_prefix: str = "",
        document_prefix: str = "",
        trust_remote_code: bool = False,
        local_files_only: bool = False,
    ) -> None:
        self.model_name = str(model).strip()
        self.device_name = str(device).strip() or "cpu"
        self.pooling = str(pooling).casefold()
        self.batch_size = int(batch_size)
        self.max_length = int(max_length)
        self.query_prefix = str(query_prefix)
        self.document_prefix = str(document_prefix)
        self.trust_remote_code = bool(trust_remote_code)
        self.local_files_only = bool(local_files_only)
        if not self.model_name:
            raise ValueError("memory.embedding.model must be non-empty")
        if self.pooling not in {"cls", "mean"}:
            raise ValueError("memory.embedding.pooling must be 'cls' or 'mean'")
        if self.batch_size <= 0:
            raise ValueError("memory.embedding.batch_size must be positive")
        if self.max_length <= 0:
            raise ValueError("memory.embedding.max_length must be positive")
        self._load_lock = threading.Lock()
        self._inference_lock = threading.Lock()
        self._torch = None
        self._tokenizer = None
        self._model = None
        self._device = None

    def _load(self) -> None:
        if self._model is not None:
            return
        with self._load_lock:
            if self._model is not None:
                return
            try:
                import torch
                from transformers import AutoModel, AutoTokenizer
            except ModuleNotFoundError as error:
                raise RuntimeError(
                    "Semantic Insight retrieval requires torch and transformers"
                ) from error
            device = self.device_name
            if device == "auto":
                device = "cuda" if torch.cuda.is_available() else "cpu"
            options = {
                "trust_remote_code": self.trust_remote_code,
                "local_files_only": self.local_files_only,
            }
            tokenizer = AutoTokenizer.from_pretrained(self.model_name, **options)
            model = AutoModel.from_pretrained(self.model_name, **options)
            model.eval()
            model.to(device)
            self._torch = torch
            self._tokenizer = tokenizer
            self._model = model
            self._device = device

    def _embed(self, texts: Sequence[str], *, prefix: str) -> list[list[float]]:
        values = [f"{prefix}{str(text)}" for text in texts]
        if not values:
            return []
        self._load()
        torch = self._torch
        tokenizer = self._tokenizer
        model = self._model
        if torch is None or tokenizer is None or model is None:
            raise RuntimeError("embedding model did not initialize")
        vectors: list[list[float]] = []
        # A single shared encoder is used by task workers. Serialize forward passes to
        # avoid oversubscribing CPU/GPU kernels while retaining a process-wide cache.
        with self._inference_lock, torch.inference_mode():
            for start in range(0, len(values), self.batch_size):
                batch = values[start : start + self.batch_size]
                encoded = tokenizer(
                    batch,
                    padding=True,
                    truncation=True,
                    max_length=self.max_length,
                    return_tensors="pt",
                )
                encoded = {
                    key: value.to(self._device) for key, value in encoded.items()
                }
                hidden = model(**encoded).last_hidden_state
                if self.pooling == "cls":
                    pooled = hidden[:, 0]
                else:
                    mask = encoded["attention_mask"].unsqueeze(-1).to(hidden.dtype)
                    pooled = (hidden * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1)
                pooled = torch.nn.functional.normalize(pooled, p=2, dim=1)
                vectors.extend(pooled.detach().cpu().float().tolist())
        return vectors

    def embed_queries(self, texts: Sequence[str]) -> list[list[float]]:
        return self._embed(texts, prefix=self.query_prefix)

    def embed_documents(self, texts: Sequence[str]) -> list[list[float]]:
        return self._embed(texts, prefix=self.document_prefix)


def build_text_embedder(memory_config: dict[str, Any]) -> TextEmbedder:
    """Build the configured semantic encoder without loading model weights yet."""
    value = memory_config.get("embedding") or {}
    if not isinstance(value, dict):
        raise ValueError("memory.embedding must be a JSON object")
    provider = str(value.get("provider") or "transformers").casefold()
    if provider != "transformers":
        raise ValueError("memory.embedding.provider must be 'transformers'")
    return TransformerTextEmbedder(
        str(value.get("model") or "BAAI/bge-m3"),
        device=str(value.get("device") or "cpu"),
        pooling=str(value.get("pooling") or "cls"),
        batch_size=int(value.get("batch_size", 32)),
        max_length=int(value.get("max_length", 512)),
        query_prefix=str(value.get("query_prefix") or ""),
        document_prefix=str(value.get("document_prefix") or ""),
        trust_remote_code=bool(value.get("trust_remote_code", False)),
        local_files_only=bool(value.get("local_files_only", False)),
    )


__all__ = ["TextEmbedder", "TransformerTextEmbedder", "build_text_embedder"]

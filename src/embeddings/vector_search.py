"""Batch-oriented multilingual text embedding generation.

Author: Miguel Medina Cantos
"""

from __future__ import annotations

import math
import threading
from typing import Any, Protocol, Sequence


class EmbeddingError(RuntimeError):
    """Raised when an embedding model returns invalid vectors."""


class EmbeddingService(Protocol):
    """Port for text-to-vector implementations."""

    @property
    def model_name(self) -> str:
        """Return the exact model identifier used for generated vectors."""

    @property
    def dimension(self) -> int:
        """Return the expected vector dimension."""

    def embed_texts(self, texts: Sequence[str]) -> list[list[float]]:
        """Embed a batch of non-empty texts."""

    def embed_query(self, query: str) -> list[float]:
        """Embed one search query."""


class SentenceTransformerEmbeddingService:
    """Load MiniLM once per process and generate normalized embeddings in batches."""

    _models: dict[tuple[str, str], Any] = {}
    _models_lock = threading.Lock()

    def __init__(
        self,
        *,
        model_name: str,
        dimension: int,
        batch_size: int,
        device: str,
        normalize: bool = True,
    ) -> None:
        self._model_name = model_name
        self._dimension = dimension
        self._batch_size = batch_size
        self._device = device
        self._normalize = normalize

    @property
    def model_name(self) -> str:
        """Return the configured sentence-transformer model."""

        return self._model_name

    @property
    def dimension(self) -> int:
        """Return the configured output dimension."""

        return self._dimension

    def _get_model(self) -> Any:
        key = (self._model_name, self._device)
        model = self._models.get(key)
        if model is not None:
            return model
        with self._models_lock:
            model = self._models.get(key)
            if model is None:
                try:
                    from sentence_transformers import SentenceTransformer
                except ImportError as error:
                    raise RuntimeError(
                        "sentence-transformers is required to generate invoice embeddings."
                    ) from error
                model = SentenceTransformer(self._model_name, device=self._device)
                detected_dimension = model.get_sentence_embedding_dimension()
                if detected_dimension != self._dimension:
                    raise EmbeddingError(
                        f"Model dimension {detected_dimension} does not match configured "
                        f"dimension {self._dimension}."
                    )
                self._models[key] = model
        return model

    def embed_texts(self, texts: Sequence[str]) -> list[list[float]]:
        """Encode a bounded text batch and validate every vector."""

        if not texts:
            return []
        if any(not isinstance(text, str) or not text.strip() for text in texts):
            raise EmbeddingError("Embedding inputs must be non-empty strings.")
        encoded = self._get_model().encode(
            list(texts),
            batch_size=self._batch_size,
            normalize_embeddings=self._normalize,
            convert_to_numpy=True,
            show_progress_bar=False,
        )
        vectors = [[float(value) for value in row] for row in encoded]
        if len(vectors) != len(texts):
            raise EmbeddingError("Embedding model returned an unexpected vector count.")
        for vector in vectors:
            self._validate_vector(vector)
        return vectors

    def embed_query(self, query: str) -> list[float]:
        """Encode one query with the same model and normalization as indexed data."""

        if not query.strip():
            raise EmbeddingError("Search query cannot be empty.")
        return self.embed_texts([query])[0]

    def _validate_vector(self, vector: Sequence[float]) -> None:
        if len(vector) != self._dimension:
            raise EmbeddingError(
                f"Embedding has dimension {len(vector)}; expected {self._dimension}."
            )
        if any(not math.isfinite(value) for value in vector):
            raise EmbeddingError("Embedding contains a non-finite value.")
        if self._normalize:
            magnitude = math.sqrt(sum(value * value for value in vector))
            if not math.isclose(magnitude, 1.0, rel_tol=1e-4, abs_tol=1e-4):
                raise EmbeddingError("Normalized embedding does not have unit magnitude.")

# Text embedding via sentence-transformers.

# Wraps a single sentence-transformers model behind a small interface. The model
# is loaded lazily on first use and reused for the lifetime of the process, since
# loading is expensive relative to encoding.

from __future__ import annotations
from typing import TYPE_CHECKING

from app.config import settings

if TYPE_CHECKING:
    from sentence_transformers import SentenceTransformer


class Embedder:
    """Lazily-loaded sentence-transformers embedding model."""

    def __init__(self, model_name: str = settings.embedding_model) -> None:
        self.model_name = model_name
        self._model: SentenceTransformer | None = None

    @property
    def model(self) -> SentenceTransformer:
        """Return the underlying model, loading it on first access."""
        if self._model is None:
            # Imported here so the (heavy) dependency is only required at runtime.
            from sentence_transformers import SentenceTransformer

            self._model = SentenceTransformer(self.model_name)
        return self._model

    def embed(self, texts: list[str]) -> list[list[float]]:
        """Embed a batch of texts, returning plain Python float lists."""
        embeddings = self.model.encode(texts, convert_to_numpy=True)
        return embeddings.tolist()


embedder = Embedder()

# Persistent vector store backed by ChromaDB.

# Stores chunk embeddings on disk and exposes cosine-similarity search. The
# collection is configured for cosine distance so that scores reflect semantic
# similarity regardless of embedding magnitude.


from __future__ import annotations
import logging
from typing import TYPE_CHECKING

import chromadb

from app.config import settings

if TYPE_CHECKING:
    from app.ingestion import Chunk

# This Chroma version ships a posthog-incompatible telemetry client: it calls
# posthog.capture() with a signature that raises TypeError, then logs it as an
# error on every event. Disabling telemetry doesn't help (the failing call
# happens regardless), so silence that logger directly.
logging.getLogger("chromadb.telemetry.product.posthog").setLevel(logging.CRITICAL)


class VectorStore:
    """Thin wrapper around a persistent ChromaDB collection."""

    def __init__(
        self,
        persist_dir: str = settings.chroma_persist_dir,
        collection_name: str = settings.collection_name,
    ) -> None:
        self._client = chromadb.PersistentClient(
            path=persist_dir,
            settings=chromadb.Settings(anonymized_telemetry=False),
        )
        self._collection = self._client.get_or_create_collection(
            name=collection_name,
            metadata={"hnsw:space": "cosine"},
        )

    def add_chunks(self, chunks: list[Chunk], embeddings: list[list[float]]) -> None:
        """Upsert chunks (keyed by ``chunk.id``) with their embeddings."""
        if not chunks:
            return
        self._collection.upsert(
            ids=[c.id for c in chunks],
            embeddings=embeddings,
            documents=[c.text for c in chunks],
            metadatas=[{"source": c.source, "section": c.section} for c in chunks],
        )

    def query(
        self, query_embedding: list[float], top_k: int = settings.top_k
    ) -> list[dict]:
        """Return the ``top_k`` closest chunks, best match first."""
        result = self._collection.query(
            query_embeddings=[query_embedding],
            n_results=top_k,
        )
        documents = result["documents"][0]
        metadatas = result["metadatas"][0]
        distances = result["distances"][0]
        return [
            {
                "text": document,
                "source": metadata.get("source"),
                "section": metadata.get("section"),
                "distance": distance,
            }
            for document, metadata, distance in zip(documents, metadatas, distances)
        ]

    def count(self) -> int:
        """Return the number of stored chunks."""
        return self._collection.count()


vector_store = VectorStore()

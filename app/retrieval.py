# Semantic retrieval: turn a question into the most relevant chunks.

# Thin orchestration layer that embeds the query and searches the vector store,
# returning typed results. Embedding and storage logic live in their own modules.


import sys
from dataclasses import dataclass

from app.config import settings
from app.embeddings import embedder
from app.vectorstore import vector_store


@dataclass
class RetrievalResult:
    """A single retrieved chunk and how closely it matched the query."""

    text: str
    source: str
    section: str
    distance: float


def retrieve(query: str, top_k: int = settings.top_k) -> list[RetrievalResult]:
    """Return the ``top_k`` chunks most relevant to ``query``, best match first."""
    query_embedding = embedder.embed([query])[0]
    hits = vector_store.query(query_embedding, top_k=top_k)
    return [
        RetrievalResult(
            text=hit["text"],
            source=hit["source"],
            section=hit["section"],
            distance=hit["distance"],
        )
        for hit in hits
    ]


if __name__ == "__main__":
    question = " ".join(sys.argv[1:]) or "How do I create an account?"
    print(f"Q: {question}\n")
    for result in retrieve(question):
        preview = result.text[:100].replace("\n", " ")
        print(f"[{result.distance:.3f}] {result.source} | {result.section}")
        print(f"    {preview!r}\n")

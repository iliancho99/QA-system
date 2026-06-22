# CLI: build chunks, embed them, and index them into the vector store.

# Run with::

#     python -m scripts.ingest


from app.config import settings
from app.embeddings import embedder
from app.ingestion import build_chunks
from app.vectorstore import vector_store


def main() -> None:
    chunks = build_chunks(settings.docs_dir)
    if not chunks:
        print(f"No chunks found under {settings.docs_dir!r}; nothing to index.")
        return

    embeddings = embedder.embed([chunk.text for chunk in chunks])
    vector_store.add_chunks(chunks, embeddings)

    print(f"Indexed {len(chunks)} chunks.")
    print(f"Collection now contains {vector_store.count()} chunks.")


if __name__ == "__main__":
    main()

"""Retrieval-augmented generation: retrieve context, then synthesize an answer.

Ties the retrieval layer to the LLM. The model is instructed to answer strictly
from the retrieved context so answers stay grounded in the document corpus.
"""

from __future__ import annotations

from app.llm import llm
from app.retrieval import RetrievalResult, retrieve

SYSTEM_PROMPT = (
    "You are a helpful assistant that answers questions using ONLY the provided "
    "context. Follow these rules strictly:\n"
    "- Base your answer solely on the numbered context passages below. Do not "
    "use outside knowledge.\n"
    "- Cite the source document name(s) you used, e.g. (source: billing.md).\n"
    "- If the context does not contain enough information to answer, reply "
    'exactly: "I don\'t have enough information to answer that."'
)


def build_context(results: list[RetrievalResult]) -> str:
    """Format retrieved chunks into a numbered, source-attributed context block."""
    blocks = []
    for i, result in enumerate(results, start=1):
        blocks.append(
            f"[{i}] (source: {result.source} | section: {result.section})\n"
            f"{result.text}"
        )
    return "\n\n".join(blocks)


def _unique_sources(results: list[RetrievalResult]) -> list[dict]:
    """Return distinct (source, section) pairs, preserving retrieval order."""
    seen: set[tuple[str, str]] = set()
    sources: list[dict] = []
    for result in results:
        key = (result.source, result.section)
        if key not in seen:
            seen.add(key)
            sources.append({"source": result.source, "section": result.section})
    return sources


def ask(question: str) -> dict:
    """Answer ``question`` from the document corpus.

    Returns ``{"answer": str, "sources": [{"source", "section"}, ...]}``.
    """
    results = retrieve(question)
    context = build_context(results)
    user_prompt = (
        f"Context:\n{context}\n\n"
        f"Question: {question}\n\n"
        "Answer using only the context above."
    )
    answer = llm.generate(user_prompt, system_prompt=SYSTEM_PROMPT)
    return {"answer": answer, "sources": _unique_sources(results)}

"""Tests for semantic retrieval (app.retrieval).

These are integration tests: they exercise the real embedding model and the
persisted Chroma collection, so the corpus must have been ingested first
(``python -m scripts.ingest``).
"""

from __future__ import annotations

from app.retrieval import retrieve


def test_retrieve_returns_results() -> None:
    results = retrieve("How do I create an account?")

    assert results, "expected at least one retrieved chunk"
    for result in results:
        assert result.text
        assert result.source
        assert result.section
        assert isinstance(result.distance, float)


def test_retrieve_best_match_is_relevant() -> None:
    # This question is squarely about the user-accounts document.
    results = retrieve("How do I reset my password?")

    assert results
    assert results[0].source == "user-accounts.md"

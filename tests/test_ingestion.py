"""Tests for markdown loading and chunking (app.ingestion)."""

from __future__ import annotations

from app.config import settings
from app.ingestion import chunk_markdown

SOURCE = "test.md"


def test_chunk_markdown_splits_by_headers() -> None:
    raw = "# Title\nIntro body.\n\n## Section A\nBody A.\n\n## Section B\nBody B.\n"
    chunks = chunk_markdown(raw, SOURCE)

    assert len(chunks) == 3
    assert [c.section for c in chunks] == [
        "test.md > Title",
        "test.md > Title > Section A",
        "test.md > Title > Section B",
    ]
    assert all(c.source == SOURCE for c in chunks)


def test_chunk_markdown_respects_max_size() -> None:
    # One section whose body comfortably exceeds CHUNK_SIZE.
    body = "word " * (settings.chunk_size)  # ~5x chunk_size chars
    raw = f"# Big\n{body}\n"

    chunks = chunk_markdown(raw, SOURCE)

    assert len(chunks) > 1
    # Each window is at most CHUNK_SIZE chars (overlap only affects start step).
    assert all(len(c.text) <= settings.chunk_size for c in chunks)


def test_chunk_markdown_skips_empty() -> None:
    # The middle header has no body of its own.
    raw = (
        "# Title\n"
        "Real content.\n\n"
        "## Empty Section\n\n"
        "## Filled Section\n"
        "More content.\n"
    )
    chunks = chunk_markdown(raw, SOURCE)

    assert all(c.text.strip() for c in chunks)
    sections = [c.section for c in chunks]
    assert "test.md > Title > Empty Section" not in sections
    assert "test.md > Title > Filled Section" in sections


def test_chunk_ids_are_deterministic() -> None:
    raw = "# A\nAlpha.\n\n## B\nBeta.\n"

    first = chunk_markdown(raw, SOURCE)
    second = chunk_markdown(raw, SOURCE)

    assert [c.id for c in first] == [c.id for c in second]
    # ids follow the documented "{source}::{index}" form.
    assert [c.id for c in first] == ["test.md::0", "test.md::1"]

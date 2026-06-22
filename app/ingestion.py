# Document loading and chunking.

# Reads markdown documents from the docs directory and splits them into
# section-aware chunks. No embeddings are produced here — this stage only
# prepares the text units that will later be embedded and stored.

import re
from dataclasses import dataclass
from pathlib import Path

from app.config import settings

_HEADER_RE = re.compile(r"^(#{1,6})\s+(.*)$")


@dataclass
class Chunk:
    """A single retrievable unit of text with its provenance."""

    id: str
    text: str
    source: str
    section: str


def load_documents(docs_dir: str) -> list[tuple[str, str]]:
    """Read every ``*.md`` file in ``docs_dir``.

    Returns a list of ``(filename, raw_text)`` tuples, sorted by filename so
    that chunk ids are stable across runs.
    """
    base = Path(docs_dir)
    documents: list[tuple[str, str]] = []
    for path in sorted(base.glob("*.md")):
        documents.append((path.name, path.read_text(encoding="utf-8")))
    return documents


def _section_label(source: str, header_path: list[str]) -> str:
    """Render the section breadcrumb, e.g. ``user-accounts.md > Creating``."""
    return " > ".join([source, *header_path]) if header_path else source


def _split_sections(raw_text: str) -> list[tuple[list[str], str]]:
    """Split raw markdown into ``(header_path, body)`` sections.

    The header path is the trail of enclosing headers, so nested sections carry
    their parents' titles.
    """
    sections: list[tuple[list[str], str]] = []
    header_stack: list[tuple[int, str]] = []
    body_lines: list[str] = []

    def flush() -> None:
        body = "\n".join(body_lines).strip()
        if body:
            sections.append(([title for _, title in header_stack], body))
        body_lines.clear()

    for line in raw_text.splitlines():
        match = _HEADER_RE.match(line)
        if match:
            flush()
            level = len(match.group(1))
            title = match.group(2).strip()
            while header_stack and header_stack[-1][0] >= level:
                header_stack.pop()
            header_stack.append((level, title))
        else:
            body_lines.append(line)
    flush()
    return sections


def _split_with_overlap(text: str, size: int, overlap: int) -> list[str]:
    """Window ``text`` into ~``size`` char pieces overlapping by ``overlap``."""
    if len(text) <= size:
        return [text]
    step = max(1, size - overlap)
    pieces: list[str] = []
    for start in range(0, len(text), step):
        piece = text[start : start + size].strip()
        if piece:
            pieces.append(piece)
        if start + size >= len(text):
            break
    return pieces


def chunk_markdown(raw_text: str, source: str) -> list[Chunk]:
    """Split one document into section-aware, size-bounded chunks."""
    chunks: list[Chunk] = []
    index = 0
    for header_path, body in _split_sections(raw_text):
        section = _section_label(source, header_path)
        for piece in _split_with_overlap(
            body, settings.chunk_size, settings.chunk_overlap
        ):
            if not piece.strip():
                continue
            chunks.append(
                Chunk(
                    id=f"{source}::{index}",
                    text=piece,
                    source=source,
                    section=section,
                )
            )
            index += 1
    return chunks


def build_chunks(docs_dir: str) -> list[Chunk]:
    """Load and chunk every document under ``docs_dir``."""
    chunks: list[Chunk] = []
    for filename, raw_text in load_documents(docs_dir):
        chunks.extend(chunk_markdown(raw_text, filename))
    return chunks


if __name__ == "__main__":
    documents = load_documents(settings.docs_dir)
    chunks = build_chunks(settings.docs_dir)
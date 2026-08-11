"""Materialize the chunk corpus as files an agent can grep and read.

Direct Corpus Interaction needs the corpus on disk, not in a vector index. Each
page becomes one Markdown file whose chunks are separated by a marker carrying
the chunk id, and a JSON table of contents records the line range of every
chunk. That marker is what keeps the method scoreable: a grep hit at a line maps
back to exactly one `page_chunks.id`, the unit the benchmark scores.

The workspace is derived state and is built to survive a corpus far larger than
the current one:

- pages stream out of SQLite one at a time, so the build never holds the corpus
  in memory;
- each page carries its own content hash, so a rebuild rewrites only the pages
  that changed and deletes only the ones that disappeared;
- path, page and chunk lookups are materialized once per load, never per query.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Iterator
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path

from src.db.pages import connect_pages as connect
from src.shared.env import ROOT

CHUNK_MARKER = "<!-- chunk:"
DEFAULT_WORKSPACE = Path(".local/dci_workspace")
INDEX_VERSION = 1


@dataclass(frozen=True)
class ChunkSpan:
    """Where one chunk lives inside its page file."""

    chunk_id: str
    heading_path: str
    start_line: int
    end_line: int


@dataclass(frozen=True)
class PageDocument:
    page_id: str
    path: str
    title: str
    source: str
    url: str
    line_count: int
    spans: tuple[ChunkSpan, ...]
    content_hash: str = ""

    def toc(self) -> str:
        """Line-numbered table of contents, precomputed offline in RISE."""
        lines = [f"{self.path} - {self.title or self.url} ({self.line_count} lines)"]
        lines.extend(
            f"  lines {span.start_line}-{span.end_line}  {span.chunk_id}  "
            f"{span.heading_path or '(no heading)'}"
            for span in self.spans
        )
        return "\n".join(lines)

    def chunk_at_line(self, line: int) -> str | None:
        for span in self.spans:
            if span.start_line <= line <= span.end_line:
                return span.chunk_id
        return None


@dataclass(frozen=True)
class CorpusWorkspace:
    root: Path
    documents: tuple[PageDocument, ...]
    by_path: dict[str, PageDocument] = field(default_factory=dict, repr=False)
    by_page_id: dict[str, PageDocument] = field(default_factory=dict, repr=False)
    by_chunk_id: dict[str, PageDocument] = field(default_factory=dict, repr=False)

    @classmethod
    def of(cls, root: Path, documents: tuple[PageDocument, ...]) -> CorpusWorkspace:
        # Built once per load. Rebuilding these per query is the difference
        # between a constant-time lookup and an O(corpus) scan on every step.
        return cls(
            root=root,
            documents=documents,
            by_path={document.path: document for document in documents},
            by_page_id={document.page_id: document for document in documents},
            by_chunk_id={
                span.chunk_id: document
                for document in documents
                for span in document.spans
            },
        )

    def read_lines(self, path: str, start: int, end: int) -> list[tuple[int, str]]:
        document = self.by_path.get(path)
        if document is None:
            return []
        text = (self.root / path).read_text(encoding="utf-8").splitlines()
        start = max(1, start)
        end = min(len(text), end)
        return [(number, text[number - 1]) for number in range(start, end + 1)]

    def chunk_for(self, path: str, line: int) -> str | None:
        document = self.by_path.get(path)
        return document.chunk_at_line(line) if document else None

    def document_for_chunk(self, chunk_id: str) -> PageDocument | None:
        return self.by_chunk_id.get(chunk_id)


def build_workspace(root: Path | None = None, *, rebuild: bool = False):
    """Write the corpus to disk, rewriting only what changed."""
    root = (root or ROOT / DEFAULT_WORKSPACE).resolve()
    index_path = root / "index.json"
    stored: dict[str, dict] = {}
    if index_path.exists() and not rebuild:
        payload = json.loads(index_path.read_text(encoding="utf-8"))
        if payload.get("version") == INDEX_VERSION:
            stored = {entry["path"]: entry for entry in payload.get("documents", [])}

    documents: list[PageDocument] = []
    written = 0
    for page in _pages():
        path = _page_path(page)
        previous = stored.get(path)
        content_hash = _page_hash(page["chunks"])
        if previous is not None and previous.get("content_hash") == content_hash:
            documents.append(_document_from_entry(previous))
            continue
        documents.append(_write_page(root, path, page, content_hash))
        written += 1

    live = {document.path for document in documents}
    for stale in sorted(set(stored) - live):
        (root / stale).unlink(missing_ok=True)

    index_path.parent.mkdir(parents=True, exist_ok=True)
    index_path.write_text(
        json.dumps(
            {
                "version": INDEX_VERSION,
                "documents": [_entry_from_document(doc) for doc in documents],
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    if written:
        print(f"dci workspace: wrote {written} of {len(documents)} pages", flush=True)
    return CorpusWorkspace.of(root, tuple(documents))


@lru_cache(maxsize=4)
def load_workspace(root: str | None = None) -> CorpusWorkspace:
    """Process-wide cached workspace; a benchmark loads it once, not per query."""
    return build_workspace(Path(root) if root else None)


def _pages() -> Iterator[dict]:
    """Stream pages in page order so the corpus never lands in memory at once."""
    with connect() as conn:
        cursor = conn.execute(
            """
            select c.id, c.page_id, c.chunk_index, c.heading_path, c.text,
                   coalesce(m.title, '') as title, m.source, m.url
            from page_chunks c
            join page_metadata m on m.id = c.page_id
            order by c.page_id, c.chunk_index
            """
        )
        current: list[dict] = []
        for row in cursor:
            row = dict(row)
            if current and row["page_id"] != current[0]["page_id"]:
                yield {"page_id": current[0]["page_id"], "chunks": current}
                current = []
            current.append(row)
        if current:
            yield {"page_id": current[0]["page_id"], "chunks": current}


def _page_hash(chunks: list[dict]) -> str:
    digest = hashlib.sha256()
    for chunk in chunks:
        digest.update(chunk["id"].encode("utf-8"))
        digest.update(b"\0")
        digest.update(str(chunk["text"]).encode("utf-8"))
        digest.update(b"\0")
    return digest.hexdigest()


def _page_path(page: dict) -> str:
    source = _safe_name(page["chunks"][0]["source"] or "unknown")
    return f"{source}/{page['page_id']}.md"


def _write_page(root: Path, path: str, page: dict, content_hash: str) -> PageDocument:
    chunks = page["chunks"]
    target = root / path
    target.parent.mkdir(parents=True, exist_ok=True)

    lines: list[str] = []
    spans: list[ChunkSpan] = []
    for chunk in chunks:
        heading = chunk["heading_path"] or ""
        lines.append(f"{CHUNK_MARKER} {chunk['id']} | {heading} -->")
        start = len(lines) + 1
        lines.extend(str(chunk["text"]).splitlines() or [""])
        spans.append(
            ChunkSpan(
                chunk_id=chunk["id"],
                heading_path=heading,
                start_line=start,
                end_line=len(lines),
            )
        )
        lines.append("")
    target.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return PageDocument(
        page_id=page["page_id"],
        path=path,
        title=chunks[0]["title"],
        source=chunks[0]["source"] or "",
        url=chunks[0]["url"] or "",
        line_count=len(lines),
        spans=tuple(spans),
        content_hash=content_hash,
    )


def _entry_from_document(document: PageDocument) -> dict:
    return {
        "page_id": document.page_id,
        "path": document.path,
        "title": document.title,
        "source": document.source,
        "url": document.url,
        "line_count": document.line_count,
        "content_hash": document.content_hash,
        "spans": [
            {
                "chunk_id": span.chunk_id,
                "heading_path": span.heading_path,
                "start_line": span.start_line,
                "end_line": span.end_line,
            }
            for span in document.spans
        ],
    }


def _document_from_entry(entry: dict) -> PageDocument:
    return PageDocument(
        page_id=entry["page_id"],
        path=entry["path"],
        title=entry["title"],
        source=entry["source"],
        url=entry["url"],
        line_count=entry["line_count"],
        spans=tuple(ChunkSpan(**span) for span in entry["spans"]),
        content_hash=entry.get("content_hash", ""),
    )


def _safe_name(value: str) -> str:
    return "".join(char if char.isalnum() or char in "-_" else "_" for char in value)

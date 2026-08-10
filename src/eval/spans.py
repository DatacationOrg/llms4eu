from __future__ import annotations

from dataclasses import dataclass

from src.db.pages import connect_pages as connect

DEFAULT_MIN_OVERLAP = 0.5


@dataclass(frozen=True)
class ChunkSpan:
    id: str
    page_id: str
    start: int
    end: int

    def overlap(self, other: ChunkSpan) -> int:
        if self.page_id != other.page_id:
            return 0
        return max(0, min(self.end, other.end) - max(self.start, other.start))


def load_chunk_spans(variant: str) -> list[ChunkSpan]:
    """Character spans of one variant's chunks, ordered by page and index."""
    with connect() as conn:
        rows = conn.execute(
            """
            select id, page_id, start_char, end_char
            from page_chunks
            where variant = ?
              and start_char is not null
              and end_char is not null
            order by page_id, chunk_index
            """,
            (variant,),
        ).fetchall()
    return [
        ChunkSpan(row["id"], row["page_id"], row["start_char"], row["end_char"])
        for row in rows
    ]


def project_relevance(
    rows: list[dict],
    gold_variant: str,
    target_variant: str,
    min_overlap: float = DEFAULT_MIN_OVERLAP,
) -> list[dict]:
    """Re-express one variant's qrels as qrels over another variant's chunks.

    Every chunking variant cuts the same pages differently, so a gold chunk id
    from `gold_variant` does not exist in `target_variant`. This maps the gold
    chunk's character span onto whichever target chunks cover it, which is what
    makes two variants comparable even though each has its own generated
    questions.

    A target chunk qualifies when the overlap is at least `min_overlap` of the
    shorter of the two spans. Measuring against the shorter span is what keeps
    the comparison fair in both directions: a small target chunk lying inside a
    large gold span qualifies, and so does a large target chunk that swallows a
    small gold span, while a chunk that merely clips an edge does not.

    Spans are paragraph-aligned, so this is a proxy for "returned the
    answer-bearing text": report it beside the strict per-variant metrics, never
    merged into them.
    """
    if gold_variant == target_variant:
        return list(rows)

    gold_by_id = {span.id: span for span in load_chunk_spans(gold_variant)}
    targets_by_page: dict[str, list[ChunkSpan]] = {}
    for span in load_chunk_spans(target_variant):
        targets_by_page.setdefault(span.page_id, []).append(span)

    projected: list[dict] = []
    for row in rows:
        gold = gold_by_id.get(row["chunk_id"])
        if gold is None:
            continue
        gold_length = gold.end - gold.start
        for target in targets_by_page.get(gold.page_id, []):
            overlap = target.overlap(gold)
            if not overlap:
                continue
            shorter = min(gold_length, target.end - target.start) or 1
            if overlap >= max(int(shorter * min_overlap), 1):
                projected.append(
                    {"question_id": row["question_id"], "chunk_id": target.id}
                )
    return projected


def neighbour_ids(variant: str) -> dict[str, list[str]]:
    """Adjacent chunk ids per chunk, for neighbour expansion at retrieval time."""
    with connect() as conn:
        rows = conn.execute(
            """
            select id, page_id
            from page_chunks
            where variant = ?
            order by page_id, chunk_index
            """,
            (variant,),
        ).fetchall()

    by_page: dict[str, list[str]] = {}
    for row in rows:
        by_page.setdefault(row["page_id"], []).append(row["id"])

    neighbours: dict[str, list[str]] = {}
    for ids in by_page.values():
        for index, chunk_id in enumerate(ids):
            around = []
            if index > 0:
                around.append(ids[index - 1])
            if index + 1 < len(ids):
                around.append(ids[index + 1])
            neighbours[chunk_id] = around
    return neighbours

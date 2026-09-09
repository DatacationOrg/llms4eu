from __future__ import annotations

import argparse
from pathlib import Path

from src.db.pages import (
    connect_pages as connect,
)
from src.db.pages import (
    initialize_page_artifacts_db,
    raw_pages_db_path,
)
from src.preprocess.chunkers import SIZE_UNITS, STRATEGIES, Chunker, build_chunker
from src.preprocess.legacy_chunker import Chunk, chunk_markdown
from src.shared.env import ROOT, load_yaml

CONFIG = load_yaml(Path(__file__).with_name("config.yaml"))

# A chunk *variant* is a way of cutting pages into chunks (sizes, overlap, unit).
# A chunk *version* is how one chunk is turned into text for embedding. They are
# independent: any variant can be indexed with any version.
BASE_CHUNK_VARIANT = "base"

__all__ = [
    "BASE_CHUNK_VARIANT",
    "Chunk",
    "backfill_chunk_spans",
    "chunk_id",
    "chunk_markdown",
    "default_chunker",
    "rebuild_page_chunks",
]


def chunk_id(page_id: str, variant: str, index: int) -> str:
    """Stable chunk id. The base variant keeps its historical bare form.

    Existing eval labels, reports, checkpoints and Chroma collections all name
    base chunks as `{page_id}:{index}`, so only new variants take a suffix.
    """
    if variant == BASE_CHUNK_VARIANT:
        return f"{page_id}:{index}"
    return f"{page_id}:{variant}:{index}"


def default_chunker() -> Chunker:
    """The frozen chunker that produced every currently labelled chunk."""
    return build_chunker(
        "legacy",
        size=CONFIG["chunk_target_chars"],
        overlap=0,
        unit="chars",
    )


def rebuild_page_chunks(
    variant: str = BASE_CHUNK_VARIANT,
    chunker: Chunker | None = None,
    clean: bool = False,
) -> None:
    """Chunk pages that this variant has not covered yet.

    Append-only per variant: a page already chunked for `variant` is left alone,
    so eval labels keep pointing at valid chunk ids. `clean` re-cuts the variant
    from scratch and is refused for the base variant in the durable database,
    where it would cascade-delete the approved labels.
    """
    chunker = chunker or default_chunker()
    initialize_page_artifacts_db()
    if clean:
        _clean_variant(variant)

    with connect() as conn:
        pages = conn.execute(
            """
            select m.id, m.title, m.source, m.page_kind, c.markdown,
                   coalesce(m.language, s.language) as language
            from page_metadata m
            join page_markdown_content c on c.page_id = m.id
            left join page_sources s on s.source = m.source
            where m.page_kind != 'empty'
              and not exists (
                select 1
                from page_chunks chunks
                where chunks.page_id = m.id
                  and chunks.variant = ?
              )
            order by m.id
            """,
            (variant,),
        ).fetchall()

        rows = []
        for page in pages:
            context = dict(page)
            for index, chunk in enumerate(chunker.split(page["markdown"], context)):
                rows.append(
                    (
                        chunk_id(page["id"], variant, index),
                        page["id"],
                        index,
                        variant,
                        chunk.heading_path or None,
                        chunk.text,
                        len(chunk.text),
                        chunk.start_char,
                        chunk.end_char,
                    )
                )

        conn.executemany(
            """
            insert into page_chunks (
              id, page_id, chunk_index, variant, heading_path, text, char_count,
              start_char, end_char
            ) values (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            rows,
        )

    print(f"chunked {len(pages)} new pages into {len(rows)} {variant} chunks")


def backfill_chunk_spans(
    variant: str = BASE_CHUNK_VARIANT,
    chunker: Chunker | None = None,
) -> int:
    """Fill in start_char/end_char for chunks stored before spans existed.

    Re-runs the variant's chunker and takes spans from the result, writing a
    span only when the regenerated chunk's text matches the stored text exactly.
    Text is never modified, and a chunk the chunker no longer reproduces is left
    alone rather than given a span that points at the wrong place.
    """
    chunker = chunker or default_chunker()
    initialize_page_artifacts_db()
    updates: list[tuple[int, int, str]] = []
    with connect() as conn:
        pages = conn.execute(
            """
            select m.id, m.title, m.source, m.page_kind, c.markdown,
                   coalesce(m.language, s.language) as language
            from page_metadata m
            join page_markdown_content c on c.page_id = m.id
            left join page_sources s on s.source = m.source
            where exists (
              select 1 from page_chunks k
              where k.page_id = m.id and k.variant = ? and k.start_char is null
            )
            order by m.id
            """,
            (variant,),
        ).fetchall()

        for page in pages:
            stored = {
                row["chunk_index"]: row
                for row in conn.execute(
                    "select chunk_index, text from page_chunks "
                    "where page_id = ? and variant = ?",
                    (page["id"], variant),
                )
            }
            for index, chunk in enumerate(chunker.split(page["markdown"], dict(page))):
                row = stored.get(index)
                if row is not None and row["text"] == chunk.text:
                    updates.append(
                        (
                            chunk.start_char,
                            chunk.end_char,
                            chunk_id(page["id"], variant, index),
                        )
                    )

        conn.executemany(
            "update page_chunks set start_char = ?, end_char = ? where id = ?",
            updates,
        )
    print(f"backfilled spans for {len(updates)} {variant} chunks")
    return len(updates)


def _clean_variant(variant: str) -> None:
    if variant == BASE_CHUNK_VARIANT and raw_pages_db_path() == _durable_db_path():
        raise RuntimeError(
            "Refusing to clean the base variant in the durable database: "
            "eval_relevant_chunks cascades on delete from page_chunks, so this "
            "would destroy the approved labels. Snapshot first with "
            "`just chunk-sweep-db` and set PAGES_DB_PATH."
        )
    with connect() as conn:
        labels = conn.execute(
            """
            select count(*)
            from eval_relevant_chunks r
            join page_chunks c on c.id = r.chunk_id
            where c.variant = ?
            """,
            (variant,),
        ).fetchone()[0]
        deleted = conn.execute(
            "delete from page_chunks where variant = ?", (variant,)
        ).rowcount
    print(f"deleted {deleted} existing {variant} chunks")
    if labels:
        # The delete cascaded. Say so: a silent label wipe looks like a clean run
        # until a later eval reports every question as a miss.
        print(f"WARNING: cascade also deleted {labels} {variant} eval labels")


def _durable_db_path() -> Path:
    """The configured database, ignoring any PAGES_DB_PATH override."""
    from src.db.pages import CONFIG as DB_CONFIG

    return ROOT / DB_CONFIG["raw_pages_db"]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--variant", default=BASE_CHUNK_VARIANT)
    parser.add_argument("--strategy", choices=STRATEGIES, default="legacy")
    parser.add_argument("--size", type=int, default=CONFIG["chunk_target_chars"])
    parser.add_argument("--overlap", type=int, default=0)
    parser.add_argument("--unit", choices=SIZE_UNITS, default="chars")
    parser.add_argument(
        "--provider",
        default=None,
        help="Embedding provider whose tokenizer sizes token-unit chunks.",
    )
    parser.add_argument("--chunk-version", default="v1")
    parser.add_argument(
        "--min-size",
        type=int,
        default=0,
        help=(
            "Fold a chunk smaller than this into a neighbour, in the same unit "
            "as --size. Defaults to half of --size; 1 disables merging."
        ),
    )
    parser.add_argument(
        "--clean",
        action="store_true",
        help="Re-cut this variant from scratch instead of appending.",
    )
    parser.add_argument(
        "--backfill-spans",
        action="store_true",
        help="Only fill missing start_char/end_char; do not chunk anything.",
    )
    args = parser.parse_args()
    chunker = build_chunker(
        args.strategy,
        size=args.size,
        overlap=args.overlap,
        unit=args.unit,
        provider=args.provider,
        chunk_version=args.chunk_version,
        min_size=args.min_size,
    )
    if args.backfill_spans:
        backfill_chunk_spans(args.variant, chunker)
        return
    rebuild_page_chunks(
        variant=args.variant,
        chunker=chunker,
        clean=args.clean,
    )


if __name__ == "__main__":
    main()

"""Store page chunks for one chunking variant.

This module owns storage only: which strategy to run, where the rows go, and how
a variant is kept separate from the others. The splitting itself lives in
`src.preprocess.chunkers`, which delegates to `langchain-text-splitters`.
"""

from __future__ import annotations

import argparse

from src.db.pages import (
    connect_pages as connect,
)
from src.db.pages import (
    initialize_page_artifacts_db,
)
from src.indexing.chunk_text import BASE_CHUNK_VARIANT as BASE_VARIANT
from src.preprocess.chunkers import SIZE_UNITS, STRATEGIES, build_chunker
from src.preprocess.legacy_chunker import base_bounds

PAGE_COLUMNS = """
    m.id, m.title, m.source, m.page_kind, s.language, c.markdown
"""


def base_variant_chunker():
    """The chunking the approved eval labels were built on."""
    target, _max_size, _min_size = base_bounds()
    return build_chunker("legacy", size=target, overlap=0, unit="chars")


def rebuild_page_chunks(
    variant: str = BASE_VARIANT,
    chunker=None,
    clean: bool = False,
) -> None:
    """Chunk every page that has no chunks yet for this variant.

    Append-only per variant: a page already chunked for `variant` is left alone
    so approved labels keep pointing at the same text. `clean` drops the
    variant's rows first, which is how a variant gets re-cut.
    """
    chunker = chunker or base_variant_chunker()
    initialize_page_artifacts_db()

    with connect() as conn:
        if clean:
            deleted = conn.execute(
                "delete from page_chunks where variant = ?", (variant,)
            ).rowcount
            if deleted:
                print(f"removed {deleted} existing {variant} chunks")

        pages = conn.execute(
            f"""
            select {PAGE_COLUMNS}
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
            for index, chunk in enumerate(chunker.split(page["markdown"], dict(page))):
                rows.append(
                    (
                        chunk_id(page["id"], variant, index),
                        page["id"],
                        index,
                        variant,
                        chunk.heading_path,
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

    print(
        f"chunked {len(pages)} new pages into {len(rows)} "
        f"{variant} chunks ({chunker.name})"
    )


def backfill_chunk_spans(variant: str = BASE_VARIANT, chunker=None) -> None:
    """Fill start_char/end_char for chunks written before spans existed.

    Re-cuts each page and copies the spans across by chunk index. A page whose
    text no longer matches is reported and skipped rather than labelled from a
    different cut, so a stale strategy cannot corrupt the spans.
    """
    chunker = chunker or base_variant_chunker()
    initialize_page_artifacts_db()
    updates: list[tuple[int, int, str]] = []
    skipped: list[str] = []

    with connect() as conn:
        pages = conn.execute(
            f"""
            select {PAGE_COLUMNS}
            from page_metadata m
            join page_markdown_content c on c.page_id = m.id
            left join page_sources s on s.source = m.source
            where exists (
              select 1 from page_chunks chunks
              where chunks.page_id = m.id
                and chunks.variant = ?
                and chunks.start_char is null
            )
            order by m.id
            """,
            (variant,),
        ).fetchall()

        for page in pages:
            stored = conn.execute(
                """
                select id, text from page_chunks
                where page_id = ? and variant = ?
                order by chunk_index
                """,
                (page["id"], variant),
            ).fetchall()
            chunks = chunker.split(page["markdown"], dict(page))
            if len(chunks) != len(stored) or any(
                chunk.text != row["text"] for chunk, row in zip(chunks, stored)
            ):
                skipped.append(page["id"])
                continue
            updates.extend(
                (chunk.start_char, chunk.end_char, row["id"])
                for chunk, row in zip(chunks, stored)
            )

        conn.executemany(
            "update page_chunks set start_char = ?, end_char = ? where id = ?",
            updates,
        )

    print(f"backfilled spans for {len(updates)} {variant} chunks")
    if skipped:
        print(f"skipped {len(skipped)} pages whose text no longer matches")


def chunk_id(page_id: str, variant: str, index: int) -> str:
    """Stable chunk id. The base variant keeps its historical two-part form."""
    if variant == BASE_VARIANT:
        return f"{page_id}:{index}"
    return f"{page_id}:{variant}:{index}"


def variant_names() -> list[str]:
    with connect() as conn:
        return [
            row["variant"]
            for row in conn.execute(
                "select distinct variant from page_chunks order by variant"
            )
        ]


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Rebuild page chunks for one chunking variant."
    )
    parser.add_argument("--variant", default=BASE_VARIANT)
    parser.add_argument(
        "--strategy",
        choices=STRATEGIES,
        default="recursive",
        help="recursive: library splitter. markdown: per heading section. "
        "legacy: the frozen original, needed to reproduce `base`.",
    )
    parser.add_argument("--size", type=int, help="Chunk size, in --unit")
    parser.add_argument("--overlap", type=int, default=0)
    parser.add_argument("--unit", choices=SIZE_UNITS, default="chars")
    parser.add_argument(
        "--provider",
        help="Embedding provider whose tokenizer sizes chunks when --unit tokens",
    )
    parser.add_argument("--chunk-version", default="v1")
    parser.add_argument("--clean", action="store_true")
    parser.add_argument("--backfill-spans", action="store_true")
    args = parser.parse_args()

    chunker = build_chunker(
        strategy=args.strategy,
        size=args.size or _default_size(args),
        overlap=args.overlap,
        unit=args.unit,
        provider=args.provider,
        chunk_version=args.chunk_version,
    )
    if args.backfill_spans:
        backfill_chunk_spans(args.variant, chunker)
        return
    rebuild_page_chunks(args.variant, chunker, clean=args.clean)


def _default_size(args) -> int:
    """Fall back to the provider's window in tokens, or the configured size."""
    if args.unit == "tokens":
        from src.shared.tokenizers import provider_token_limit

        return provider_token_limit(args.provider or "")
    return base_bounds()[0]


if __name__ == "__main__":
    main()

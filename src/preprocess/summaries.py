from __future__ import annotations

import re
from concurrent.futures import ThreadPoolExecutor, as_completed

from pydantic import BaseModel, Field

from src.db.pages import (
    connect_pages as connect,
)
from src.db.pages import (
    initialize_page_artifacts_db,
)
from src.shared.env import load_local_env
from src.shared.llm import AzureFoundryStructuredLlm

SUMMARY_TARGET_WORDS = 40
SUMMARY_MAX_TOKENS = 128
SUMMARY_MAX_WORDS = 96


class ChunkSummary(BaseModel):
    summary: str = Field(
        max_length=800,
        description="Search-optimized summary in the same language as the source chunk.",
    )


def generate_missing_summaries(limit: int | None, workers: int = 10) -> None:
    initialize_page_artifacts_db()
    load_local_env()
    client = AzureFoundryStructuredLlm.from_env(max_tokens=SUMMARY_MAX_TOKENS)
    chunks = _chunks_missing_summary(limit)
    print(f"found {len(chunks)} chunks missing summaries", flush=True)

    completed = 0
    updated = 0
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = [pool.submit(_foundry_summary, chunk, client) for chunk in chunks]
        for future in as_completed(futures):
            completed += 1
            chunk, summary = future.result()
            if summary:
                with connect() as conn:
                    conn.execute(
                        "update page_chunks set summary = ? where id = ?",
                        (summary, chunk["id"]),
                    )
                updated += 1
            if completed % 25 == 0 or completed == len(chunks):
                print(
                    f"processed {completed}/{len(chunks)}, updated {updated}",
                    flush=True,
                )
    print(f"done, updated {updated} summaries", flush=True)


def _chunks_missing_summary(limit: int | None) -> list[dict]:
    sql_limit = "limit ?" if limit else ""
    params = (limit,) if limit else ()
    with connect() as conn:
        return [
            dict(row)
            for row in conn.execute(
                f"""
                select c.id, c.text, c.heading_path, m.title, s.language
                from page_chunks c
                join page_metadata m on m.id = c.page_id
                join page_sources s on s.source = m.source
                where c.summary is null or trim(c.summary) = ''
                order by c.id
                {sql_limit}
                """,
                params,
            )
        ]


def _foundry_summary(
    chunk: dict,
    client: AzureFoundryStructuredLlm,
) -> tuple[dict, str | None]:
    try:
        result = client.structured_output(
            _summary_prompt(chunk),
            ChunkSummary,
            retries=3,
        )
        return chunk, _clean_summary(result.summary)
    except (RuntimeError, ValueError) as exc:
        print(f"failed summary {chunk['id']}: {exc}", flush=True)
        return chunk, None


def _summary_prompt(chunk: dict) -> list[tuple[str, str]]:
    return [
        (
            "system",
            f"""
Write a search-optimized retrieval summary for embedding search.

Use the same language as the source chunk. Preserve exact proper nouns, place
names, dates, event names, amenities, activities, address/location clues, and
alternate names users might search for. Include compact synonyms only when they
are directly supported by the chunk. Do not add facts, marketing language, or
phrases like "this chunk". Target about {SUMMARY_TARGET_WORDS} words.
""",
        ),
        (
            "human",
            f"""
Title: {chunk["title"] or ""}
Heading: {chunk["heading_path"] or ""}
Language: {chunk["language"]}

Chunk:
{chunk["text"]}
""",
        ),
    ]


def _clean_summary(text: str) -> str:
    summary = re.sub(r"\s+", " ", text.strip().strip('"')).strip()
    if not summary:
        raise ValueError("empty summary")
    if len(summary.split()) > SUMMARY_MAX_WORDS:
        raise ValueError(f"summary exceeded {SUMMARY_MAX_WORDS} words")
    return summary

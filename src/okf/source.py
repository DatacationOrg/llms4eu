from __future__ import annotations

import sqlite3
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class SourcePage:
    id: str
    source: str
    url: str
    title: str
    language: str
    markdown: str


def load_source_pages(
    db_path: Path,
    *,
    source: str | None = None,
    limit: int | None = None,
) -> list[SourcePage]:
    uri = f"file:{db_path.resolve()}?mode=ro"
    with sqlite3.connect(uri, uri=True) as conn:
        conn.row_factory = sqlite3.Row
        sql = """
            select m.id, m.source, m.url, coalesce(m.title, '') as title,
                   coalesce(m.language, s.language, '') as language, c.markdown
            from page_metadata m
            join page_markdown_content c on c.page_id = m.id
            left join page_sources s on s.source = m.source
            where m.error is null
              and m.page_kind != 'empty'
              and length(trim(c.markdown)) > 0
        """
        params: list[str | int] = []
        if source:
            sql += " and m.source = ?"
            params.append(source)
        sql += " order by m.source, m.url"
        if limit is not None:
            sql += " limit ?"
            params.append(limit)
        return [SourcePage(**dict(row)) for row in conn.execute(sql, params)]

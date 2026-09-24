from datetime import datetime

from pydantic import BaseModel, Field


__all__ = ["PageMarkdownContent", "PageMetadata"]


class PageMetadata(BaseModel):
    id: str
    source: str
    url: str
    final_url: str | None = None
    fetched_at: datetime
    status_code: int | None = None
    content_type: str | None = None
    title: str | None = None
    content_hash: str | None = None
    raw_bytes: int = 0
    fetch_method: str = "httpx"
    extractor: str = "trafilatura"
    language: str | None = None
    page_kind: str = "prose"
    markdown_chars: int = 0
    error: str | None = None

    @classmethod
    def db_columns(cls) -> list[str]:
        return list(cls.model_fields)

    def db_values(self) -> tuple:
        values = []
        for column in self.db_columns():
            value = getattr(self, column)
            if isinstance(value, datetime):
                value = value.isoformat()
            values.append(value)
        return tuple(values)


class PageMarkdownContent(BaseModel):
    page_id: str
    markdown: str = Field(default="")

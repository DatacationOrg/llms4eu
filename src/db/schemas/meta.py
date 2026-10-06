from __future__ import annotations

from typing import Any, Literal

from pydantic import Json

from src.db.schemas.base import Row, Split


class Meta(Row):
    """A list or geo question whose answer is a set of pages."""

    file = "meta"
    key: str
    kind: Literal["list", "geo"]
    lang: str
    spec: str  # the task in English
    question: str
    question_en: str
    gold_pages: list[str]
    gold: list[str]  # Wikidata ids
    n_gold: int
    anchor: str | None  # geo: the centre page
    radius_km: int | None
    category: str | None
    check: Json[dict[str, Any]]
    ok: bool
    split: Split

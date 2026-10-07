from __future__ import annotations

from typing import Any, Literal

from pydantic import Json

from src.db.schemas.base import Category, Lang, Row, Split


class Meta(Row):
    """A question whose answer is a set of pages: all places of a kind in a country
    (`list`) or near a place (`geo`). Test on `ok` rows."""

    file = "qa/wiki_qa_meta.parquet"
    key: str
    """Question id: `list|<category>|<country>` or `geo|<anchor page id>`."""
    kind: Literal["list", "geo"]
    """`list`: by kind and country; `geo`: by kind within `radius_km` of `anchor`."""
    lang: Lang
    """Language of the question."""
    spec: str
    """The task in English, e.g. `all national parks within 5 km of Lac d'Arratille`."""
    question: str
    """The question."""
    question_en: str
    """The question in English, for reading."""
    gold_pages: list[str]
    """The answer: every page (`Page.id`) that fits."""
    gold: list[str]
    """The same places as Wikidata ids."""
    n_gold: int
    """Size of the answer, 1 to 20."""
    anchor: str | None
    """Geo: the page the distance is measured from; null for `list`."""
    radius_km: Literal[5, 10, 25] | None
    """Geo: the distance; null for `list`."""
    category: Category | None
    """Geo: the kind of place asked for; null for `list` (it is in `key`)."""
    check: Json[dict[str, Any]]
    """Checks by a second call: matches_spec, fluent (bool each)."""
    ok: bool
    """Whether every check passed."""
    split: Split
    """Dev or test."""

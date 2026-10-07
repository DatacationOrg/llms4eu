"""What every dataset file shares: the row base and the value sets used across files.

Each field is documented by the docstring under it (pydantic keeps it as the field's
description). A field is optional when its type ends in `| None`; null then means
what its docstring says.
"""

from __future__ import annotations

from typing import ClassVar, Literal

from pydantic import BaseModel, ConfigDict

Split = Literal["dev", "test"]
"""`dev` to tune on, `test` to report on; a page's questions all share one split."""

Lang = Literal[
    "bg", "cs", "da", "de", "el", "en", "es", "et", "fi", "fr", "ga", "hr", "hu",
    "it", "lb", "lt", "lv", "mt", "nl", "pl", "pt", "ro", "sk", "sl", "sv", "tr",
]  # fmt: skip
"""ISO 639-1 code of a language: the 24 EU languages, plus lb and tr."""

Country = Literal[
    "AT", "BE", "BG", "CY", "CZ", "DE", "DK", "EE", "ES", "FI", "FR", "GR", "HR", "HU",
    "IE", "IT", "LT", "LU", "LV", "MT", "NL", "PL", "PT", "RO", "SE", "SI", "SK",
]  # fmt: skip
"""ISO 3166-1 alpha-2 code of an EU27 country."""

Category = Literal[
    "castle", "castle_ruin", "fortification", "national_park", "nature_reserve",
    "natural_monument", "forest", "cave", "waterfall", "lake", "mountain", "garden",
]  # fmt: skip
"""Kind of place, from its Wikidata class (`datagen/corpus/config.yaml`)."""


class Row(BaseModel):
    model_config = ConfigDict(use_attribute_docstrings=True)
    file: ClassVar[str]
    """The file, relative to the dataset folder."""


class Qrels(BaseModel):
    """Which other pages fit the question, judged by an LLM over its BM25 top 20.
    Every field but `qrels_judged` is null when the question was not judged."""

    model_config = ConfigDict(use_attribute_docstrings=True)

    qrels_judged: bool
    """Whether the other pages were judged at all."""
    gold_match: Literal["yes", "partly", "no"] | None = None
    """The judge's verdict on the gold page itself, a control."""
    gold_answers: bool | None = None
    """Whether the gold page's text answers the question."""
    relevant: list[str] | None = None
    """Other page ids the question fits fully; they count as correct."""
    partial: list[str] | None = None
    """Other page ids the question fits partly."""
    answering: list[str] | None = None
    """Page ids whose text answers the question."""
    hard_negatives: list[str] | None = None
    """Similar page ids that do not fit: training negatives."""
    n_relevant: int | None = None
    """Pages the question fits: the gold page plus `relevant`, 1 to 21."""
    hits: Literal["unique", "few", "many"] | None = None
    """`n_relevant` in words: 1, 2 to 3, or 4 and more."""
    pool_saturated: bool | None = None
    """Whether more pages likely fit than the 20 judged."""

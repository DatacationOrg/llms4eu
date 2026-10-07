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
"""Kind of place, from its Wikidata class (instance of)."""


class Row(BaseModel):
    model_config = ConfigDict(use_attribute_docstrings=True)
    file: ClassVar[str]
    """The file, relative to the dataset folder."""


class Qrels(BaseModel):
    """Which other pages fit the question, judged over its BM25 top 20 (null = not judged)."""

    qrels_judged: bool
    gold_match: str | None = None  # the gold page's own judgement: yes / partly / no
    gold_answers: bool | None = None
    relevant: list[str] | None = None  # other pages it fits fully
    partial: list[str] | None = None  # other pages it fits partly
    answering: list[str] | None = None  # pages whose text answers it
    hard_negatives: list[str] | None = None  # similar pages that do not fit
    n_relevant: int | None = None  # gold + relevant
    hits: Literal["unique", "few", "many"] | None = None
    pool_saturated: bool | None = None  # likely more matches than judged

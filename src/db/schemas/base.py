"""What every wiki QA file shares."""

from __future__ import annotations

from typing import ClassVar, Literal

from pydantic import BaseModel

Split = Literal["dev", "test"]


class Row(BaseModel):
    file: ClassVar[str]  # relative to the dataset folder


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

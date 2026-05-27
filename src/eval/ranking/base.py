from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol


@dataclass(frozen=True)
class RankedChunk:
    id: str
    score: float
    text: str


class RankingMethod(Protocol):
    name: str

    def retrieve(self, query: str, limit: int) -> list[RankedChunk]: ...

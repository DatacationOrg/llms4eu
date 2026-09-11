from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

from src.retrieval.base import RankedChunk, Retriever
from src.shared.geo_boost import distance_multiplier, haversine_km
from src.shared.geocode import Coordinates, GeocodeProvider, locate_text
from src.shared.llm import StructuredLlm

__all__ = ["GeoBoostRetriever"]


@dataclass(frozen=True)
class GeoBoostRetriever:
    name: str
    base_retriever: Retriever
    llm: StructuredLlm
    geocoder: GeocodeProvider
    chunk_coordinates: Callable[[list[str]], dict[str, Coordinates]]
    weight: float
    decay_km: float

    def retrieve(self, query: str, limit: int) -> list[RankedChunk]:
        candidates = self.base_retriever.retrieve(query, limit)
        return self._boost(query, candidates)

    def retrieve_batch(
        self,
        queries: list[str],
        limit: int,
    ) -> dict[int, list[RankedChunk]]:
        rankings = self.base_retriever.retrieve_batch(queries, limit)
        return {
            index: self._boost(queries[index], chunks)
            for index, chunks in rankings.items()
        }

    def _boost(self, query: str, chunks: list[RankedChunk]) -> list[RankedChunk]:
        if not chunks:
            return chunks
        question_coords = locate_text(self.llm, self.geocoder, query)
        if question_coords is None:
            return chunks

        coords_by_chunk = self.chunk_coordinates([chunk.id for chunk in chunks])
        boosted = [
            RankedChunk(
                id=chunk.id,
                score=chunk.score
                * self._multiplier(chunk.id, coords_by_chunk, question_coords),
                text=chunk.text,
            )
            for chunk in chunks
        ]
        return sorted(boosted, key=lambda chunk: chunk.score, reverse=True)

    def _multiplier(
        self,
        chunk_id: str,
        coords_by_chunk: dict[str, Coordinates],
        question_coords: Coordinates,
    ) -> float:
        coords = coords_by_chunk.get(chunk_id)
        if coords is None:
            return 1.0
        distance_km = haversine_km(question_coords, coords)
        return distance_multiplier(distance_km, weight=self.weight, decay_km=self.decay_km)

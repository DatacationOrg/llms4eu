"""Geo-aware retrieval: geography re-ranks (soft) or pre-filters (strict).

Soft (`*_hybrid_geo`, `*_hybrid_rerank_geo`, the geo agents' first stage): the
existing stage retrieves `limit * overfetch` chunks unfiltered and every score
is recombined as `(1 - w) * text + w * s_geo`, both in [0, 1]. `s_geo` is
`exp(-km / decay)` for a point scope, 1.0 for a chunk whose page lies in a
region scope, 0.0 for a located page outside it, and 1.0 (neutral) for a page
with no location. A region level that keeps more than `max_scope_share` of the
located pages carries no information and is skipped (the selectivity gate).
This is the convex score fusion Bruch et al. (TOIS 2023) found beats rank
fusion, with the unknown-footprint rule from the 2026-09-08 geo run: a hard
filter lost 15 of 72 scoped questions, all to gold pages with no footprint.

Strict (`*_hybrid_rerank_geo_strict`): the shape Spatial-RAG (Yu et al. 2025)
calls sparse spatial filtering plus dense semantic matching. The scope filters
the stage (Chroma `where`, a BM25 page set), widening radius -> NUTS-3 ->
NUTS-2 -> country -> none while fewer than `min_candidates` located in-scope
chunks come back, then boosts by distance. Kept for measurement and for the
agent's `pages_in_region` tool, where the user asked for containment.

The scope is resolved once per query and threaded into the stage through
constructor fields (`VectorChunkRetriever.where`, `SparseRetriever.allowed_page_ids`),
so the public `retrieve(query, limit)` contract, and with it the eval harness,
is untouched.
"""

from __future__ import annotations

import math
from collections.abc import Callable
from dataclasses import dataclass, field, replace
from typing import Literal

from src.db.pages import PageLocation
from src.retrieval.base import RankedChunk, Retriever
from src.shared.geo_resolver import ScopeResolver
from src.shared.geo_scope import GeoScope
from src.shared.geocode import distance_multiplier, haversine_km

__all__ = ["GeoScopedRetriever", "GeoStats"]

GeoMode = Literal["soft", "strict"]
_CODE_LEVELS = ("nuts3", "nuts2", "country")


@dataclass
class GeoStats:
    """Per-query trace, for the action log and the report."""

    resolved: int = 0
    unresolved: int = 0
    # Strict: the filter had to widen. Soft: the selectivity gate dropped a level.
    widenings: int = 0
    levels: dict[str, int] = field(default_factory=dict)

    def reset(self) -> None:
        self.resolved = self.unresolved = self.widenings = 0
        self.levels.clear()

    def record(self, scope: GeoScope | None, used: GeoScope | None) -> None:
        if scope is None:
            self.unresolved += 1
            return
        self.resolved += 1
        level = used.level if used else "none"
        self.levels[level] = self.levels.get(level, 0) + 1
        if used is not None and used.level != scope.level:
            self.widenings += 1
        elif used is None and scope.filters:
            self.widenings += 1


@dataclass(frozen=True)
class GeoScopedRetriever:
    name: str
    resolver: ScopeResolver
    # The full first stage (fusion, optionally reranked) for one scope; called
    # with None for the unfiltered stage.
    build_stage: Callable[[GeoScope | None], Retriever]
    # Chunk id -> primary location of its page; absent when the page is unlocated.
    chunk_locations: Callable[[list[str]], dict[str, PageLocation]]
    mode: GeoMode = "soft"
    # Soft: weight of the geographic score in the convex combination.
    # Strict: depth of the multiplicative distance boost after the filter.
    boost_weight: float = 0.3
    decay_km: float = 50.0
    # Soft only.
    overfetch: int = 4
    max_scope_share: float = 0.9
    scope_share: Callable[[GeoScope], float | None] | None = None
    # Strict only. `include_null` is applied here, not read off the scope: cached
    # scopes carry the policy that was configured when they were stored.
    min_candidates: int = 5
    include_null: bool = False
    stats: GeoStats = field(
        default_factory=GeoStats, init=False, repr=False, compare=False
    )
    _stages: dict[str, Retriever] = field(
        default_factory=dict, init=False, repr=False, compare=False
    )
    _shares: dict[str, float | None] = field(
        default_factory=dict, init=False, repr=False, compare=False
    )

    # --- public -----------------------------------------------------------------

    def retrieve(self, query: str, limit: int) -> list[RankedChunk]:
        scope = self._resolve(query)
        if self.mode == "soft":
            return self._retrieve_soft(query, limit, scope)
        return self._retrieve_strict(query, limit, scope)

    def retrieve_batch(
        self,
        queries: list[str],
        limit: int,
    ) -> dict[int, list[RankedChunk]]:
        scopes = {index: self._resolve(query) for index, query in enumerate(queries)}
        if self.mode == "soft":
            return self._retrieve_batch_soft(queries, limit, scopes)
        return self._retrieve_batch_strict(queries, limit, scopes)

    # --- soft: over-fetch, then fuse text and geography -------------------------

    def _retrieve_soft(
        self, query: str, limit: int, scope: GeoScope | None
    ) -> list[RankedChunk]:
        used = self._effective(scope)
        self.stats.record(scope, used)
        stage = self._stage(None)
        if used is None:
            return stage.retrieve(query, limit)
        chunks = stage.retrieve(query, limit * self.overfetch)
        return self._fuse(chunks, used)[:limit]

    def _retrieve_batch_soft(
        self, queries: list[str], limit: int, scopes: dict[int, GeoScope | None]
    ) -> dict[int, list[RankedChunk]]:
        used = {index: self._effective(scope) for index, scope in scopes.items()}
        for index in scopes:
            self.stats.record(scopes[index], used[index])
        stage = self._stage(None)
        results: dict[int, list[RankedChunk]] = {}
        plain = [index for index, scope in used.items() if scope is None]
        scoped = [index for index, scope in used.items() if scope is not None]
        if plain:
            rankings = stage.retrieve_batch([queries[i] for i in plain], limit)
            for position, index in enumerate(plain):
                results[index] = rankings.get(position, [])
        if scoped:
            rankings = stage.retrieve_batch(
                [queries[i] for i in scoped], limit * self.overfetch
            )
            for position, index in enumerate(scoped):
                results[index] = self._fuse(rankings.get(position, []), used[index])[
                    :limit
                ]
        return results

    def _effective(self, scope: GeoScope | None) -> GeoScope | None:
        """The scope the soft path scores with, after the selectivity gate.

        A point scope scores by distance, which discriminates as long as the
        located pages do not all share one coordinate. A region scope is applied
        at its most specific level that keeps at most `max_scope_share` of the
        located pages; a level that keeps nearly all of them is a no-op here.
        """
        if scope is None:
            return None
        if scope.coordinates is not None:
            return scope
        current: GeoScope | None = scope
        while current is not None and current.filters:
            share = self._share(current)
            if share is not None and share <= self.max_scope_share:
                return current
            current = current.widen()
        return None

    def _fuse(self, chunks: list[RankedChunk], scope: GeoScope) -> list[RankedChunk]:
        if not chunks:
            return chunks
        locations = self.chunk_locations([chunk.id for chunk in chunks])
        scores = [chunk.score for chunk in chunks]
        low, high = min(scores), max(scores)
        span = high - low
        w = self.boost_weight
        fused = [
            RankedChunk(
                id=chunk.id,
                score=(1 - w) * ((chunk.score - low) / span if span > 0 else 1.0)
                + w * self._geo_score(locations.get(chunk.id), scope),
                text=chunk.text,
            )
            for chunk in chunks
        ]
        # Stable: ties keep the text order.
        return sorted(fused, key=lambda chunk: chunk.score, reverse=True)

    def _geo_score(self, location: PageLocation | None, scope: GeoScope) -> float:
        if location is None:
            return 1.0  # unknown footprint is neutral, never "outside"
        if scope.coordinates is not None:
            if location.coordinates is None:
                return 1.0
            return math.exp(
                -haversine_km(scope.coordinates, location.coordinates) / self.decay_km
            )
        return 1.0 if _in_region(location, scope) else 0.0

    def _share(self, scope: GeoScope) -> float | None:
        if self.scope_share is None:
            return None
        key = scope.to_json()
        if key not in self._shares:
            try:
                self._shares[key] = self.scope_share(scope)
            except Exception as exc:  # no pages database in this process
                print(f"geo scope share unavailable, level kept: {exc}", flush=True)
                self._shares[key] = None
        return self._shares[key]

    # --- strict: filter, widen when thin, boost ---------------------------------

    def _retrieve_strict(
        self, query: str, limit: int, scope: GeoScope | None
    ) -> list[RankedChunk]:
        scope = self._with_null_policy(scope)
        current = scope
        while True:
            chunks = self._stage(current).retrieve(query, limit)
            if not _filters(current) or self._enough(chunks, current):
                break
            current = current.widen()
        self.stats.record(scope, current)
        return self._boost(chunks, scope)

    def _retrieve_batch_strict(
        self, queries: list[str], limit: int, scopes: dict[int, GeoScope | None]
    ) -> dict[int, list[RankedChunk]]:
        scopes = {index: self._with_null_policy(s) for index, s in scopes.items()}
        pending: dict[int, GeoScope | None] = dict(scopes)
        results: dict[int, list[RankedChunk]] = {}
        used: dict[int, GeoScope | None] = {}
        while pending:
            groups: dict[str, list[int]] = {}
            for index, scope in pending.items():
                key = scope.to_json() if _filters(scope) else "none"
                groups.setdefault(key, []).append(index)
            for indices in groups.values():
                scope = pending[indices[0]]
                stage = self._stage(scope)
                rankings = stage.retrieve_batch([queries[i] for i in indices], limit)
                for position, index in enumerate(indices):
                    chunks = rankings.get(position, [])
                    if not _filters(scope) or self._enough(chunks, scope):
                        results[index] = chunks
                        used[index] = scope
                        del pending[index]
                    else:
                        pending[index] = scope.widen()
        for index in results:
            self.stats.record(scopes[index], used[index])
        return {index: self._boost(results[index], scopes[index]) for index in results}

    def _with_null_policy(self, scope: GeoScope | None) -> GeoScope | None:
        if scope is None or scope.include_null == self.include_null:
            return scope
        return replace(scope, include_null=self.include_null)

    def _enough(self, chunks: list[RankedChunk], scope: GeoScope) -> bool:
        """Whether the filter contributed: enough located, in-scope chunks.

        With `include_null` the filtered stage always fills up with unlocated
        pages, so counting chunks would never widen; count the ones the scope
        actually selected.
        """
        if len(chunks) < self.min_candidates:
            return False
        if not scope.include_null:
            return True
        locations = self.chunk_locations([chunk.id for chunk in chunks])
        in_scope = sum(
            1
            for chunk in chunks
            if chunk.id in locations and _in_scope(locations[chunk.id], scope)
        )
        return in_scope >= self.min_candidates

    def _boost(
        self, chunks: list[RankedChunk], scope: GeoScope | None
    ) -> list[RankedChunk]:
        if not chunks or scope is None or scope.coordinates is None:
            return chunks
        locations = self.chunk_locations([chunk.id for chunk in chunks])
        if not locations:
            return chunks
        # Multiplying only works on positive scores; a reranker may go negative.
        floor = min(chunk.score for chunk in chunks)
        shift = -floor + 1e-6 if floor <= 0 else 0.0
        boosted = [
            RankedChunk(
                id=chunk.id,
                score=(chunk.score + shift)
                * self._multiplier(locations.get(chunk.id), scope),
                text=chunk.text,
            )
            for chunk in chunks
        ]
        return sorted(boosted, key=lambda chunk: chunk.score, reverse=True)

    def _multiplier(self, location: PageLocation | None, scope: GeoScope) -> float:
        if location is None or location.coordinates is None:
            return 1.0
        return distance_multiplier(
            haversine_km(scope.coordinates, location.coordinates),
            weight=self.boost_weight,
            decay_km=self.decay_km,
        )

    # --- shared -----------------------------------------------------------------

    def _resolve(self, query: str) -> GeoScope | None:
        try:
            return self.resolver.resolve(query)
        except Exception as exc:  # never let a gazetteer outage end a run
            print(f"geo scope resolution failed, unscoped: {exc}", flush=True)
            return None

    def _stage(self, scope: GeoScope | None) -> Retriever:
        """One stage per distinct filter; scopes repeat heavily across a run."""
        key = scope.to_json() if _filters(scope) else "none"
        if key not in self._stages:
            self._stages[key] = self.build_stage(scope if _filters(scope) else None)
        return self._stages[key]


def _filters(scope: GeoScope | None) -> bool:
    return scope is not None and scope.filters


def _in_region(location: PageLocation, scope: GeoScope) -> bool:
    """Whether a located page lies in the scope's most specific code level."""
    level = scope.level
    if level not in _CODE_LEVELS:
        return True
    column = "country_code" if level == "country" else level
    return getattr(location, column) == getattr(scope, column)


def _in_scope(location: PageLocation, scope: GeoScope) -> bool:
    if scope.level == "radius":
        return (
            location.coordinates is not None
            and haversine_km(scope.coordinates, location.coordinates) <= scope.radius_km
        )
    return _in_region(location, scope)

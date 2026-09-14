from __future__ import annotations

from dataclasses import dataclass, replace


__all__ = [
    "AgentSearchState",
    "FallbackEmbeddingModel",
    "HigherLimit",
    "QueryReformulation",
    "WiderGeoFilter",
]


@dataclass(frozen=True)
class AgentSearchState:
    query: str
    limit: int
    embedding_model: str
    geo_filter: dict[str, str] | None = None
    attempt: int = 1
    strategy: str = "initial"


@dataclass(frozen=True)
class QueryReformulation:
    reformulated_query: str | None
    name: str = "query_reformulation"

    def apply(self, state: AgentSearchState) -> AgentSearchState | None:
        query = (self.reformulated_query or "").strip()
        if not query or query == state.query:
            return None
        return _advance(state, self.name, query=query)


@dataclass(frozen=True)
class HigherLimit:
    max_limit: int
    step_factor: float = 2.0
    name: str = "higher_limit"

    def apply(self, state: AgentSearchState) -> AgentSearchState | None:
        next_limit = min(int(state.limit * self.step_factor), self.max_limit)
        if next_limit <= state.limit:
            return None
        return _advance(state, self.name, limit=next_limit)


@dataclass(frozen=True)
class FallbackEmbeddingModel:
    fallback_models: tuple[str, ...]
    name: str = "fallback_model"

    def apply(self, state: AgentSearchState) -> AgentSearchState | None:
        for model in self.fallback_models:
            if model != state.embedding_model:
                return _advance(state, self.name, embedding_model=model)
        return None


@dataclass(frozen=True)
class WiderGeoFilter:
    name: str = "wider_geo_filter"

    def apply(self, state: AgentSearchState) -> AgentSearchState | None:
        if not state.geo_filter:
            return None

        next_filter = dict(state.geo_filter)
        if "nuts2_region" in next_filter:
            next_filter.pop("nuts2_region")
        elif "country_code" in next_filter:
            next_filter.pop("country_code")
        else:
            next_filter = {}

        if next_filter == state.geo_filter:
            return None

        return _advance(state, self.name, geo_filter=next_filter or None)


def _advance(state: AgentSearchState, strategy: str, **changes) -> AgentSearchState:
    return replace(state, attempt=state.attempt + 1, strategy=strategy, **changes)

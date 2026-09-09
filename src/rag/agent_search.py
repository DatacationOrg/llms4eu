from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from src.rag.judge import ChunkSufficiency, SufficiencyJudge
from src.rag.retry import (
    AgentSearchState,
    FallbackEmbeddingModel,
    HigherLimit,
    QueryReformulation,
    RetryStrategy,
    WiderGeoFilter,
)
from src.rag.search import ScoredPlace, search_places_with_model
from src.shared.env import load_yaml


__all__ = ["AgentSearchResult", "agentic_search_places"]

CONFIG = load_yaml(Path(__file__).with_name("config.yaml"))
SEARCH_CONFIG = CONFIG["search"]
JUDGE_CONFIG = CONFIG["judge"]
AGENT_CONFIG = CONFIG["agent_search"]


@dataclass(frozen=True)
class AgentAttempt:
    state: AgentSearchState
    sufficient: bool
    reason: str
    hits: int


@dataclass(frozen=True)
class AgentSearchResult:
    places: list[ScoredPlace]
    attempts: list[AgentAttempt]
    sufficient: bool


def agentic_search_places(
    question: str,
    limit: int | None = None,
    geo_filter: dict[str, str] | None = None,
) -> AgentSearchResult:
    """Retry the legacy places search until the judge is satisfied.

    `geo_filter` is `GeoScope.to_legacy_filter()` for the question's scope; the
    `WiderGeoFilter` strategy widens it between attempts. The places collection
    itself carries no location payload yet, so the filter shapes the retry
    schedule but does not narrow that search; the page-chunk stack
    (`*_hybrid_rerank_geo`) is where the filter is applied.
    """
    state = AgentSearchState(
        query=question,
        limit=limit or AGENT_CONFIG["initial_limit"],
        embedding_model=SEARCH_CONFIG["embedding_model"],
        geo_filter=geo_filter or None,
        attempt=1,
    )
    judge = SufficiencyJudge(
        model=JUDGE_CONFIG["model"],
        min_sufficient_places=JUDGE_CONFIG["min_sufficient_places"],
    )
    max_attempts = AGENT_CONFIG["max_attempts"]

    attempts: list[AgentAttempt] = []
    best_places: list[ScoredPlace] = []
    best_verdict = ChunkSufficiency(
        sufficient=False,
        reason="No successful retrieval attempts yet.",
        reformulated_query=question,
    )

    while True:
        places = search_places_with_model(
            query=state.query,
            embedding_model=state.embedding_model,
            limit=state.limit,
            collection_name=SEARCH_CONFIG["collection_name"],
        )
        verdict = judge.evaluate(state.query, places)
        attempts.append(
            AgentAttempt(
                state=state,
                sufficient=verdict.sufficient,
                reason=verdict.reason,
                hits=len(places),
            )
        )

        if len(places) > len(best_places):
            best_places = places
            best_verdict = verdict

        if verdict.sufficient:
            return AgentSearchResult(
                places=places,
                attempts=attempts,
                sufficient=True,
            )

        if state.attempt >= max_attempts:
            return AgentSearchResult(
                places=best_places,
                attempts=attempts,
                sufficient=False,
            )

        next_state = _next_state(state, verdict)
        if next_state is None:
            return AgentSearchResult(
                places=best_places,
                attempts=attempts,
                sufficient=best_verdict.sufficient,
            )
        state = next_state


def _next_state(
    state: AgentSearchState, verdict: ChunkSufficiency
) -> AgentSearchState | None:
    strategies: tuple[RetryStrategy, ...] = (
        QueryReformulation(reformulated_query=verdict.reformulated_query),
        HigherLimit(max_limit=AGENT_CONFIG["max_limit"]),
        FallbackEmbeddingModel(tuple(AGENT_CONFIG["fallback_embedding_models"])),
        WiderGeoFilter(),
    )
    for strategy in strategies:
        candidate = strategy.apply(state)
        if candidate is not None:
            return candidate
    return None

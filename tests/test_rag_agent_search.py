from src.rag import agent_search
from src.rag.judge import ChunkSufficiency, SufficiencyJudge
from src.rag.retry import (
    AgentSearchState,
    FallbackEmbeddingModel,
    HigherLimit,
    QueryReformulation,
)
from src.rag.search import ScoredPlace


def _place(place_id: str, score: float = 1.0) -> ScoredPlace:
    return ScoredPlace(
        id=place_id,
        summary=f"Summary for {place_id}",
        place_description=f"Description for {place_id}",
        score=score,
    )


def test_sufficiency_judge_short_circuits_when_not_enough_places():
    judge = SufficiencyJudge(model="stub", min_sufficient_places=2)

    verdict = judge.evaluate("Where should I go?", [_place("p1")])

    assert verdict.sufficient is False
    assert "Only 1" in verdict.reason


def test_query_reformulation_updates_query():
    state = AgentSearchState(query="old", limit=3, embedding_model="m1")

    next_state = QueryReformulation(reformulated_query="new query").apply(state)

    assert next_state is not None
    assert next_state.query == "new query"
    assert next_state.attempt == 2


def test_higher_limit_doubles_until_max():
    state = AgentSearchState(query="q", limit=4, embedding_model="m1")

    next_state = HigherLimit(max_limit=6).apply(state)

    assert next_state is not None
    assert next_state.limit == 6
    assert next_state.attempt == 2


def test_fallback_embedding_model_switches_provider():
    state = AgentSearchState(query="q", limit=3, embedding_model="m1")

    next_state = FallbackEmbeddingModel(("m1", "m2")).apply(state)

    assert next_state is not None
    assert next_state.embedding_model == "m2"
    assert next_state.attempt == 2


def test_agentic_search_retries_until_sufficient(monkeypatch):
    monkeypatch.setattr(
        agent_search,
        "SEARCH_CONFIG",
        {"embedding_model": "m1", "collection_name": "places"},
    )
    monkeypatch.setattr(
        agent_search,
        "JUDGE_CONFIG",
        {"model": "judge", "min_sufficient_places": 1},
    )
    monkeypatch.setattr(
        agent_search,
        "AGENT_CONFIG",
        {
            "max_attempts": 3,
            "initial_limit": 3,
            "max_limit": 10,
            "fallback_embedding_models": ["m2"],
        },
    )

    calls = {"count": 0}

    def fake_search(query: str, embedding_model: str, limit: int, collection_name: str):
        calls["count"] += 1
        if calls["count"] == 1:
            return []
        return [_place("p2")]

    monkeypatch.setattr(agent_search, "search_places_with_model", fake_search)

    verdicts = [
        ChunkSufficiency(
            sufficient=False,
            reason="missing context",
            reformulated_query="better query",
        ),
        ChunkSufficiency(
            sufficient=True,
            reason="enough context",
            reformulated_query=None,
        ),
    ]

    class FakeJudge:
        def evaluate(self, question: str, places: list[ScoredPlace]):
            return verdicts.pop(0)

    monkeypatch.setattr(agent_search, "SufficiencyJudge", lambda **_: FakeJudge())

    result = agent_search.agentic_search_places("q")

    assert result.sufficient is True
    assert len(result.attempts) == 2
    assert result.attempts[0].state.strategy == "initial"
    assert result.attempts[1].state.strategy == "query_reformulation"
    assert result.places[0].id == "p2"


def test_wider_geo_filter_drops_nuts2_then_country_then_stops():
    from src.rag.retry import WiderGeoFilter

    state = AgentSearchState(
        query="q",
        limit=3,
        embedding_model="m1",
        geo_filter={"nuts2_region": "SI03", "country_code": "SI"},
    )

    wider = WiderGeoFilter().apply(state)
    assert wider is not None and wider.geo_filter == {"country_code": "SI"}
    assert wider.strategy == "wider_geo_filter"

    widest = WiderGeoFilter().apply(wider)
    assert widest is not None and widest.geo_filter is None

    assert WiderGeoFilter().apply(widest) is None
    assert WiderGeoFilter().apply(AgentSearchState("q", 3, "m1")) is None


def test_wider_geo_filter_carries_foreign_keys_and_drops_them_last():
    from src.rag.retry import WiderGeoFilter

    state = AgentSearchState(
        query="q",
        limit=3,
        embedding_model="m1",
        geo_filter={"country_code": "SI", "extra": "x"},
    )
    wider = WiderGeoFilter().apply(state)
    assert wider is not None and wider.geo_filter == {"extra": "x"}
    widest = WiderGeoFilter().apply(wider)
    assert widest is not None and widest.geo_filter is None
    assert WiderGeoFilter().apply(widest) is None

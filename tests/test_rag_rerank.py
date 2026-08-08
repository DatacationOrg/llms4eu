from dataclasses import dataclass

import httpx
import pytest

from src.retrieval.base import RankedChunk
from src.retrieval.retrievers import rerank
from src.retrieval.retrievers.rerank import (
    AzureCohereRerankRetriever,
    _parse_cohere_results,
    _prompt_kwargs,
)


@dataclass
class StubRetriever:
    name: str = "stub"

    def retrieve(self, query: str, limit: int) -> list[RankedChunk]:
        return [
            RankedChunk(id="first", score=0.9, text=f"{query} first"),
            RankedChunk(id="second", score=0.8, text=f"{query} second"),
        ][:limit]

    def retrieve_batch(
        self,
        queries: list[str],
        limit: int,
    ) -> dict[int, list[RankedChunk]]:
        return {
            index: self.retrieve(query, limit)
            for index, query in enumerate(queries)
        }


class StubClient:
    def __init__(self, response: httpx.Response) -> None:
        self.response = response
        self.calls = []

    def post(self, endpoint: str, *, json: dict) -> httpx.Response:
        self.calls.append((endpoint, json))
        return self.response


def test_prompt_kwargs_omits_empty_prompt_override():
    assert _prompt_kwargs(None, None) == {}
    assert _prompt_kwargs("rag", None) == {}
    assert _prompt_kwargs(None, "prompt") == {}


def test_prompt_kwargs_builds_custom_prompt_override():
    assert _prompt_kwargs("rag", "prompt") == {
        "prompts": {"rag": "prompt"},
        "default_prompt_name": "rag",
    }


def test_azure_cohere_reranker_maps_response_indices_to_candidates(monkeypatch):
    request = httpx.Request("POST", "https://example.test/v2/rerank")
    client = StubClient(
        httpx.Response(
            200,
            request=request,
            json={
                "results": [
                    {"index": 1, "relevance_score": 0.97},
                    {"index": 0, "relevance_score": 0.25},
                ]
            },
        )
    )
    monkeypatch.setattr(rerank, "_cohere_client", lambda *_: client)
    retriever = AzureCohereRerankRetriever(
        name="stub_rerank_cohere",
        model_name="Cohere-rerank-v4.0-pro",
        endpoint="https://example.test/v2/rerank",
        api_key="secret",
        base_retriever=StubRetriever(),
        candidate_limit=30,
    )

    ranked = retriever.retrieve("query", 2)

    assert [(chunk.id, chunk.score) for chunk in ranked] == [
        ("second", 0.97),
        ("first", 0.25),
    ]
    assert client.calls == [
        (
            "https://example.test/v2/rerank",
            {
                "model": "Cohere-rerank-v4.0-pro",
                "query": "query",
                "documents": ["query first", "query second"],
                "top_n": 2,
            },
        )
    ]


def test_azure_cohere_reranker_batch_uses_one_request_per_query(monkeypatch):
    request = httpx.Request("POST", "https://example.test/v2/rerank")
    client = StubClient(
        httpx.Response(
            200,
            request=request,
            json={"results": [{"index": 0, "relevance_score": 0.8}]},
        )
    )
    monkeypatch.setattr(rerank, "_cohere_client", lambda *_: client)
    retriever = AzureCohereRerankRetriever(
        name="stub_rerank_cohere",
        model_name="deployment",
        endpoint="https://example.test/v2/rerank",
        api_key="secret",
        base_retriever=StubRetriever(),
        candidate_limit=30,
    )

    ranked = retriever.retrieve_batch(["one", "two"], 1)

    assert [ranked[0][0].text, ranked[1][0].text] == ["one first", "two first"]
    assert len(client.calls) == 2


@pytest.mark.parametrize(
    "results,match",
    [
        ([{"index": 2, "relevance_score": 0.5}], "out of range"),
        (
            [
                {"index": 0, "relevance_score": 0.5},
                {"index": 0, "relevance_score": 0.4},
            ],
            "duplicate index",
        ),
        ([{"index": 0, "relevance_score": "high"}], "must be numeric"),
    ],
)
def test_parse_cohere_results_rejects_malformed_rankings(results, match):
    with pytest.raises(ValueError, match=match):
        _parse_cohere_results(
            {"results": results},
            document_count=2,
            top_n=len(results),
        )

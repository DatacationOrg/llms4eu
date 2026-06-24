from src.retrieval.base import RankedChunk
from src.retrieval.retrievers import agentic
from src.retrieval.retrievers.agentic import AgenticRetriever, ChunkSufficiency


class StubRetriever:
    name = "stub"

    def __init__(self, responses):
        self.responses = responses
        self.calls = []

    def retrieve(self, query: str, limit: int):
        self.calls.append((query, limit))
        return self.responses[len(self.calls) - 1]

    def retrieve_batch(self, queries: list[str], limit: int):
        return {idx: self.retrieve(query, limit) for idx, query in enumerate(queries)}


def test_agentic_retriever_returns_immediately_when_sufficient(monkeypatch):
    base = StubRetriever([[RankedChunk(id="c1", score=1.0, text="enough context")]])
    retriever = AgenticRetriever(
        name="qwen_agentic",
        base_retriever=base,
        max_attempts=3,
        min_sufficient_chunks=1,
        max_limit=10,
    )

    monkeypatch.setattr(
        AgenticRetriever,
        "_evaluate_sufficiency",
        lambda self, query, chunks: ChunkSufficiency(
            sufficient=True,
            reason="ok",
            reformulated_query=None,
        ),
    )

    chunks = retriever.retrieve("query", 5)

    assert [chunk.id for chunk in chunks] == ["c1"]
    assert base.calls == [("query", 5)]


def test_agentic_retriever_retries_with_reformulation(monkeypatch):
    base = StubRetriever(
        [
            [RankedChunk(id="c1", score=0.4, text="weak")],
            [RankedChunk(id="c2", score=0.9, text="better")],
        ]
    )
    retriever = AgenticRetriever(
        name="qwen_agentic",
        base_retriever=base,
        max_attempts=3,
        min_sufficient_chunks=1,
        max_limit=10,
    )

    verdicts = [
        ChunkSufficiency(
            sufficient=False,
            reason="insufficient",
            reformulated_query="better query",
        ),
        ChunkSufficiency(
            sufficient=True,
            reason="sufficient",
            reformulated_query=None,
        ),
    ]

    monkeypatch.setattr(
        AgenticRetriever,
        "_evaluate_sufficiency",
        lambda self, query, chunks: verdicts.pop(0),
    )

    chunks = retriever.retrieve("query", 3)

    assert [chunk.id for chunk in chunks] == ["c2"]
    assert base.calls == [("query", 3), ("better query", 3)]


def test_agentic_retriever_increases_limit_when_no_reformulation(monkeypatch):
    base = StubRetriever(
        [
            [RankedChunk(id="c1", score=0.4, text="weak")],
            [RankedChunk(id="c2", score=0.6, text="still weak")],
        ]
    )
    retriever = AgenticRetriever(
        name="qwen_agentic",
        base_retriever=base,
        max_attempts=2,
        min_sufficient_chunks=1,
        max_limit=8,
    )

    monkeypatch.setattr(
        AgenticRetriever,
        "_evaluate_sufficiency",
        lambda self, query, chunks: ChunkSufficiency(
            sufficient=False,
            reason="insufficient",
            reformulated_query=None,
        ),
    )

    chunks = retriever.retrieve("query", 4)

    assert [chunk.id for chunk in chunks] == ["c1"]
    assert base.calls == [("query", 4), ("query", 8)]


def test_agentic_retriever_uses_azure_judge_provider(monkeypatch):
    base = StubRetriever([[RankedChunk(id="c1", score=1.0, text="enough")]])
    retriever = AgenticRetriever(
        name="qwen_agentic",
        base_retriever=base,
        judge_retries=5,
        max_attempts=1,
        min_sufficient_chunks=1,
        max_limit=10,
    )

    calls = []

    def fake_azure(prompt: str, retries: int) -> ChunkSufficiency:
        calls.append((prompt, retries))
        return ChunkSufficiency(
            sufficient=True,
            reason="ok",
            reformulated_query=None,
        )

    monkeypatch.setattr(agentic, "_judge_with_azure", fake_azure)

    chunks = retriever.retrieve("query", 3)

    assert [chunk.id for chunk in chunks] == ["c1"]
    assert len(calls) == 1
    assert calls[0][1] == 5


def test_agentic_retriever_uses_5_10_15_limit_schedule(monkeypatch):
    base = StubRetriever(
        [
            [RankedChunk(id="c1", score=0.3, text="weak")],
            [RankedChunk(id="c2", score=0.4, text="still weak")],
            [RankedChunk(id="c3", score=0.5, text="best")],
        ]
    )
    retriever = AgenticRetriever(
        name="qwen_agentic",
        base_retriever=base,
        judge_retries=1,
        max_attempts=3,
        min_sufficient_chunks=1,
        initial_limit=5,
        limit_step=5,
        max_limit=15,
    )

    monkeypatch.setattr(
        AgenticRetriever,
        "_evaluate_sufficiency",
        lambda self, query, chunks: ChunkSufficiency(
            sufficient=False,
            reason="insufficient",
            reformulated_query=None,
        ),
    )

    retriever.retrieve("query", 10)

    assert base.calls == [("query", 5), ("query", 10), ("query", 15)]


def test_agentic_retriever_tracks_queries_per_question(monkeypatch):
    base = StubRetriever(
        [
            [RankedChunk(id="c1", score=0.4, text="weak")],
            [RankedChunk(id="c2", score=0.9, text="better")],
            [RankedChunk(id="c3", score=1.0, text="enough")],
        ]
    )
    retriever = AgenticRetriever(
        name="qwen_agentic",
        base_retriever=base,
        max_attempts=3,
        min_sufficient_chunks=1,
        max_limit=10,
    )

    verdicts = [
        ChunkSufficiency(
            sufficient=False,
            reason="insufficient",
            reformulated_query="better query",
        ),
        ChunkSufficiency(
            sufficient=True,
            reason="sufficient",
            reformulated_query=None,
        ),
        ChunkSufficiency(
            sufficient=True,
            reason="sufficient",
            reformulated_query=None,
        ),
    ]

    monkeypatch.setattr(
        AgenticRetriever,
        "_evaluate_sufficiency",
        lambda self, query, chunks: verdicts.pop(0),
    )

    retriever.retrieve_batch(["query one", "query two"], 3)

    assert retriever.total_queries() == 3
    assert retriever.average_queries_per_question() == 1.5

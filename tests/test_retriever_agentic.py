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


class StubJudge:
    """Structured-output client standing in for the local Ollama judge."""

    def __init__(self) -> None:
        self.calls = []

    def structured_output(self, prompt, output_schema, *, retries=3):
        self.calls.append((prompt, retries))
        return ChunkSufficiency(
            sufficient=True,
            reason="ok",
            reformulated_query=None,
        )


def test_agentic_retriever_uses_its_injected_judge():
    base = StubRetriever([[RankedChunk(id="c1", score=1.0, text="enough")]])
    judge = StubJudge()
    retriever = AgenticRetriever(
        name="qwen_agentic",
        base_retriever=base,
        judge_retries=5,
        max_attempts=1,
        min_sufficient_chunks=1,
        max_limit=10,
        judge=judge,
    )

    chunks = retriever.retrieve("query", 3)

    assert [chunk.id for chunk in chunks] == ["c1"]
    assert len(judge.calls) == 1
    assert judge.calls[0][1] == 5


def test_agentic_retriever_falls_back_to_the_default_local_judge(monkeypatch):
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

    def fake_judge(prompt: str, retries: int, judge=None) -> ChunkSufficiency:
        calls.append((prompt, retries, judge))
        return ChunkSufficiency(
            sufficient=True,
            reason="ok",
            reformulated_query=None,
        )

    monkeypatch.setattr(agentic, "_judge_locally", fake_judge)

    chunks = retriever.retrieve("query", 3)

    assert [chunk.id for chunk in chunks] == ["c1"]
    assert calls == [(calls[0][0], 5, None)]


def test_default_judge_reads_the_retrieval_config():
    judge = agentic.default_judge(
        {
            "agentic_judge_model": "gpt-oss:20b",
            "agentic_judge_num_ctx": 32768,
            "agentic_judge_num_predict": 1024,
            "agentic_judge_reasoning": "low",
            "agentic_judge_structured_method": "function_calling",
        }
    )

    assert judge.model_id == "gpt-oss:20b"
    assert judge.num_ctx == 32768
    assert judge.num_predict == 1024
    assert judge.reasoning == "low"
    assert judge.method == "function_calling"


def test_agentic_retriever_uses_10_15_20_limit_schedule(monkeypatch):
    expanded_chunks = [
        RankedChunk(id=f"c{index}", score=float(index), text="expanded")
        for index in range(20)
    ]
    base = StubRetriever(
        [
            [RankedChunk(id="c1", score=0.3, text="weak")],
            [RankedChunk(id="c2", score=0.4, text="still weak")],
            expanded_chunks,
        ]
    )
    retriever = AgenticRetriever(
        name="qwen_agentic",
        base_retriever=base,
        judge_retries=1,
        max_attempts=3,
        min_sufficient_chunks=1,
        initial_limit=10,
        limit_step=5,
        max_limit=20,
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

    chunks = retriever.retrieve("query", 10)

    assert base.calls == [("query", 10), ("query", 15), ("query", 20)]
    assert chunks == expanded_chunks


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


class _FailingJudge:
    """Judge whose structured call never succeeds."""

    def structured_output(self, prompt, output_schema, *, retries=3):
        raise RuntimeError(f"structured call failed after {retries} attempts")


def test_judge_failure_is_insufficient_rather_than_fatal(capsys):
    """One unjudgeable question must not end a long benchmark run."""
    base = StubRetriever(
        [
            [RankedChunk(id="c1", score=0.4, text="weak")],
            [RankedChunk(id="c2", score=0.5, text="still weak")],
        ]
    )
    retriever = AgenticRetriever(
        name="qwen_agentic",
        base_retriever=base,
        max_attempts=2,
        min_sufficient_chunks=1,
        max_limit=8,
        judge=_FailingJudge(),
    )

    chunks = retriever.retrieve("query", 4)

    assert [chunk.id for chunk in chunks] == ["c1"]
    assert base.calls == [("query", 4), ("query", 8)]
    verdict = retriever.batch_stats.action_log[0]["verdict"]
    assert verdict["sufficient"] is False
    assert "judge unavailable" in verdict["reason"]
    assert "agentic judge failed" in capsys.readouterr().out

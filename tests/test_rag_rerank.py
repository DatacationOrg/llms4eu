from src.retrieval.base import RankedChunk
from src.retrieval.retrievers.rerank import CrossEncoderRerankRetriever


class _StubBase:
    name = "stub"

    def retrieve(self, query, limit):
        return [RankedChunk(id=f"c{i}", score=1.0, text=f"text {i}") for i in range(2)]

    def retrieve_batch(self, queries, limit):
        return {
            index: self.retrieve(query, limit) for index, query in enumerate(queries)
        }


class _StubModel:
    def predict(self, pairs, batch_size=None):
        return [float(len(text)) for _, text in pairs]


def _retriever(monkeypatch):
    monkeypatch.setattr(
        "src.retrieval.retrievers.rerank._cross_encoder", lambda *a: _StubModel()
    )
    return CrossEncoderRerankRetriever(
        name="stub_rerank",
        model_name="m",
        base_retriever=_StubBase(),
        candidate_limit=5,
        device="cpu",
        max_length=128,
        local_files_only=True,
        batch_size=4,
    )


def test_rerank_batch_scores_every_query(monkeypatch):
    """retrieve_batch must call the cross-encoder with the args it accepts."""
    ranked = _retriever(monkeypatch).retrieve_batch(["a", "b"], limit=2)

    assert sorted(ranked) == [0, 1]
    assert [chunk.id for chunk in ranked[0]] == ["c0", "c1"]


def test_rerank_single_and_batch_agree(monkeypatch):
    retriever = _retriever(monkeypatch)

    assert [c.id for c in retriever.retrieve("a", 2)] == [
        c.id for c in retriever.retrieve_batch(["a"], 2)[0]
    ]

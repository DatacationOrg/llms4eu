from src.shared.embedding_cache import cached_embeddings


def test_cache_returns_vectors_in_input_order_and_reuses_hits(monkeypatch, tmp_path):
    """A pairing bug here silently gives one text another text's vector."""
    monkeypatch.setenv("LLMS4EU_DATA", str(tmp_path))
    texts = ["alpha", "beta", "gamma"]
    vectors = {"alpha": [1.0], "beta": [2.0], "gamma": [3.0]}
    calls = []

    def embed_missing(missing: list[str]) -> list[list[float]]:
        calls.append(list(missing))
        return [vectors[text] for text in missing]

    first = cached_embeddings(
        provider="p",
        model="m",
        kind="document",
        texts=texts,
        embed_missing=embed_missing,
    )
    # Reversed, so an order bug cannot hide behind the original ordering.
    second = cached_embeddings(
        provider="p",
        model="m",
        kind="document",
        texts=list(reversed(texts)),
        embed_missing=embed_missing,
    )

    assert first == [vectors[text] for text in texts]
    assert second == [vectors[text] for text in reversed(texts)]
    assert calls == [texts], "second call must be served entirely from cache"

"""The embedding cache must not serve vectors across sequence lengths.

`cached_embeddings` keys on (provider, model, kind, text). Truncating the same
text at 512 and at 2048 gives different vectors, so the length has to reach that
key — otherwise raising `embedding_max_seq_length` returns stale vectors from
the old length and a sequence-length sweep silently measures nothing.
"""

from __future__ import annotations

from src.shared.indexers import SentenceTransformerIndexer, build_indexer

MODEL = "Qwen/Qwen3-Embedding-0.6B"


def _indexer(max_seq_length: int | None) -> SentenceTransformerIndexer:
    return SentenceTransformerIndexer(
        name="qwen", model_name=MODEL, max_seq_length=max_seq_length
    )


def test_the_default_length_keeps_the_bare_model_name():
    # 512 is what every existing cache entry was written under; changing its key
    # would throw away the whole warm cache.
    assert _indexer(512)._cache_model() == MODEL
    assert _indexer(None)._cache_model() == MODEL


def test_a_different_length_gets_a_different_cache_identity():
    assert _indexer(1024)._cache_model() == f"{MODEL}@1024"
    assert _indexer(2048)._cache_model() != _indexer(1024)._cache_model()
    assert _indexer(2048)._cache_model() != _indexer(512)._cache_model()


def test_config_declared_sequence_lengths_are_distinct_providers():
    config = {
        "qwen_embedding_model": MODEL,
        "embedding_max_seq_length": 512,
        "qwen_s2048_embedding_model": MODEL,
        "qwen_s2048_max_seq_length": 2048,
    }

    base = build_indexer("qwen", config)
    long = build_indexer("qwen_s2048", config)

    assert base.max_seq_length == 512
    assert long.max_seq_length == 2048
    # Distinct on both axes that decide where vectors land and get reused.
    assert base.name != long.name
    assert base._cache_model() != long._cache_model()


def test_prompt_overrides_are_carried_from_config():
    config = {
        "e5_embedding_model": "intfloat/multilingual-e5-large",
        "e5_query_prompt": "query: ",
        "e5_document_prompt": "passage: ",
    }

    indexer = build_indexer("e5", config)

    assert indexer.query_prompt == "query: "
    assert indexer.document_prompt == "passage: "


def test_provider_token_limit_respects_the_model_floor():
    from src.shared.tokenizers import provider_token_limit

    # MiniLM is configured at 512 but its sentence_bert_config caps it at 256.
    # Reporting the configured value would halve its measured truncation.
    assert provider_token_limit("english") == 256
    # Qwen reads its model's full declared context, so no chunk size can be
    # truncated and chunk size is the only thing a size sweep varies.
    assert provider_token_limit("qwen") == 32768
    # The historical cap survives as its own provider so its cost stays
    # measurable against the same chunks and labels.
    assert provider_token_limit("qwen_s512") == 512

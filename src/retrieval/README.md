# Retrieval

Reusable chunk retrieval methods.

`methods.py` is the public catalog for eval and RAG callers:
`list_retrievers`, `build_retriever`, and `ensure_retriever_ready`.

`base.py` defines `RankedChunk` and the `Retriever` protocol.

`retrievers/` contains concrete methods:

- `sparse`: local lexical retrieval over page chunks.
- `{english,qwen,qwen4b,azure}`: Chroma vector search over page chunks.
- `*_hybrid`: normalized weighted score fusion over one vector provider plus
  sparse retrieval.
- `*_rerank`: Qwen3 cross-encoder reranker over first-stage candidates.
- `*_hybrid_rerank`: Qwen3 reranker over hybrid candidates.

Retrieval tuning lives in `config.yaml`. Enabled vector providers come from
`src/indexing/config.yaml`; eval defaults come from `src/eval/config.yaml`.

Sparse retrieval uses BM25. `sparse_k1` controls repeated-term saturation, and
`sparse_b` controls length normalization.

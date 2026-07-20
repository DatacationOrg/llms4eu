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

OKF navigation is intentionally not registered as another chunk retriever: it
opens hierarchical indexes and complete concept documents. Comparative
retrieval evaluation therefore keeps native chunk metrics here and adds a
shared source-page evidence view for both systems. Map ranked chunks to page IDs
and visited OKF concepts through `source_page_ids`, preserving first-evidence
order, before computing shared `nDCG@10` and `Recall@10`.

See [`experiments/indexing/README.md`](../../experiments/indexing/README.md) for workload,
latency, quality, and statistical requirements.

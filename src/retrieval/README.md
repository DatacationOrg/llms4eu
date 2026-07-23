# Retrieval

Reusable chunk retrieval methods.

`methods.py` is the public catalog for eval and RAG callers:
`list_retrievers`, `build_retriever`, and `ensure_retriever_ready`.

`base.py` defines `RankedChunk` and the `Retriever` protocol.

`retrievers/` contains concrete methods:

- `sparse`: local lexical retrieval over page chunks.
- `{english,qwen,qwen4b,nemotron,azure}`: Chroma vector search over page chunks.
- `*_hybrid`: normalized weighted score fusion over one vector provider plus
  sparse retrieval.
- `*_rerank`: Qwen3 cross-encoder reranker over first-stage candidates.
- `*_hybrid_rerank`: Qwen3 reranker over hybrid candidates.
- `qwen_hybrid_agentic`, `nemotron_hybrid_agentic`, and `azure_hybrid_agentic`:
  the agentic sufficiency
  and query-reformulation loop over the provider's hybrid-reranked chunks.
- Every dense, hybrid, reranked, and agentic method also has a `_v2` variant,
  such as `nemotron_hybrid_rerank_v2` and
  `nemotron_hybrid_agentic_v2`. These use isolated metadata-context dense and
  sparse indexes. Unsuffixed names retain the legacy v1 behavior.
- `sparse_v2` and `sparse_rerank_v2` use the same metadata-context text as v2
  dense retrieval; v1 sparse methods continue to index raw chunk text.

Retrieval tuning lives in `config.yaml`. Enabled vector providers come from
`src/indexing/config.yaml`; eval defaults come from `src/eval/config.yaml`.

Sparse retrieval uses BM25. `sparse_k1` controls repeated-term saturation, and
`sparse_b` controls length normalization.

See [`experiments/indexing/README.md`](../../experiments/indexing/README.md) for workload,
latency, quality, and statistical requirements.

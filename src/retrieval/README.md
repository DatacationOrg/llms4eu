# Retrieval

Reusable chunk retrieval methods.

`methods.py` is the public catalog for eval and RAG callers:
`list_retrievers`, `build_retriever`, and `ensure_retriever_ready`.

`base.py` defines `RankedChunk` and the `Retriever` protocol.

`retrievers/` contains concrete methods:

- `sparse`: local lexical retrieval over page chunks.
- `{english,qwen,qwen4b,nemotron,embed_v4}`: Chroma vector search over page
  chunks. `embed_v4` uses the internal Azure index configured by
  `AZURE_EMBEDDING_MODEL` (`embed-v-4-0` in the current benchmark).
- `*_hybrid`: normalized weighted score fusion over one vector provider plus
  sparse retrieval.
- `*_rerank`: Qwen3 cross-encoder reranker over first-stage candidates.
- `*_hybrid_rerank`: Qwen3 reranker over hybrid candidates.
- `*_rerank_cohere` and `*_hybrid_rerank_cohere`: Azure-hosted
  `Cohere-rerank-v4.0-pro` over the same first-stage candidates. Set
  `AZURE_COHERE_RERANK_ENDPOINT` to the deployment's complete Cohere v2 rerank
  URL and set `AZURE_COHERE_RERANK_API_KEY`. Override the deployment/model ID
  with `AZURE_COHERE_RERANK_MODEL`; it defaults to
  `Cohere-rerank-v4.0-pro`.
- `embed_v4_hybrid_rerank_cohere` and
  `embed_v4_hybrid_agentic_cohere` use `embed-v-4-0` hybrid candidates and
  Cohere v4 reranking. The agentic variant uses Cohere directly as its
  reranker, not after Qwen3.
- `qwen_hybrid_agentic`, `nemotron_hybrid_agentic`, and
  `embed_v4_hybrid_agentic`:
  the agentic sufficiency
  and query-reformulation loop over the provider's hybrid-reranked chunks.
- Every dense, hybrid, reranked, and agentic method also has a `_v2` variant,
  such as `nemotron_hybrid_rerank_v2` and
  `nemotron_hybrid_agentic_v2`. These use isolated metadata-context dense and
  sparse indexes. Unsuffixed names retain the legacy v1 behavior.
- `sparse_v2` and `sparse_rerank_v2` use the same metadata-context text as v2
  dense retrieval; v1 sparse methods continue to index raw chunk text.
- Cohere methods also have isolated `_v2` variants, for example
  `nemotron_hybrid_rerank_cohere_v2` and `sparse_rerank_cohere_v2`.

Retrieval tuning lives in `config.yaml`. Enabled vector providers come from
`src/indexing/config.yaml`; eval defaults come from `src/eval/config.yaml`.
Agentic retrieval starts with 10 reranked chunks and expands in steps of 5 up
to the configured maximum when the judge requests more context.

Sparse retrieval uses BM25. `sparse_k1` controls repeated-term saturation, and
`sparse_b` controls length normalization.

See [`experiments/indexing/README.md`](../../experiments/indexing/README.md) for workload,
latency, quality, and statistical requirements.

# Architecture Decisions

Long-lived context for retrieval, indexing, eval, and data artifact decisions.

## Retrieval Ownership

Retrieval/RAG owns reusable ranking methods. Eval owns measurement.

- `src.retrieval.base` defines `RankedChunk` and the `Retriever` protocol.
- `src.retrieval.methods` is the public method catalog:
  `list_retrievers`, `build_retriever`, `ensure_retriever_ready`.
- `src.eval` owns datasets, labels, metrics, timing, and reports.

Reason: eval should compare application retrieval methods, not own them.

## Retrieval Names

User-facing names describe method intent, not storage details.

- `sparse`: lexical retrieval, currently BM25 internally.
- `english`, `qwen`, `qwen4b`, `azure`: dense/vector providers.
- `*_hybrid`: dense provider plus sparse retrieval.
- `*_rerank`: rerank candidates from one base retriever.
- `*_hybrid_rerank`: rerank hybrid candidates.

No `_chunk` suffix in public method names because chunk retrieval is current
target. No `sparse_hybrid`; hybrid means dense plus sparse.

## Hybrid History

RRF was tested as hybrid fusion. Weighted normalized score fusion performed
better in measured runs and is current active hybrid strategy.

Historical read:

- Azure weighted hybrid was strongest measured method.
- Qwen 4B weighted hybrid was strongest local method.
- Default hybrid weights are 70% vector, 30% sparse.

Keep RRF as historical context in research docs unless new eval justifies
active support.

## Rerank History

Reranking retrieves `N` candidates, then returns final top `M`. Current defaults:
30 candidates, 10 final results.

Measured Qwen3 reranking was harmful and slow in our setup. Keep rerank methods
available for experiments, but do not treat rerank as default until model usage,
prompt, and input formatting are diagnosed.

## Chunk Storage

SQLite is source of truth for page chunks. Chroma is derived vector cache.

- `page_chunks.text` stores canonical chunk text.
- Chroma stores embeddings plus minimal ids.
- Vector retrieval hydrates chunk text from SQLite.
- Chroma readiness means collection exists and count matches `page_chunks`.

## Chunk Summaries

Chunk summaries were useful experiment artifacts, but are not part of
steady-state DB or retrieval code.

Historical result: chunk+summary often measured well, but summary generation
added extra derived content and operational cost. Current steady-state indexes
title, heading path, and chunk text.

## Indexing Boundary

`src.indexing` orchestrates rebuilds. `src.vector_store` owns Chroma mechanics
and query-time vector search.

Provider-shaped embedding implementations live in `src.shared.indexers`.
Retrieval provider files bind those indexers into `Retriever` instances without
instantiating providers at import time.

## Config Boundary

Experiment/runtime policy belongs in YAML config, not hidden in constructors.

- Retrieval limits, hybrid weights, reranker settings:
  `src/retrieval/config.yaml`.
- Indexing providers, default provider, model batch settings:
  `src/indexing/config.yaml`.
- Eval result limits and metric cutoffs: `src/eval/config.yaml`.

Implementation classes may keep defensive library defaults only when they are
not project policy.

## Data Artifacts

Tracked durable artifacts use descriptive paths:

- `data/db/pages.db`: page SQLite DB.
- `data/cache/chroma/`: regenerable shared vector cache.

`.local/` is private scratch and can be overridden through `.env`.

`just rebuild-vector-cache` rebuilds only the default production provider,
currently `qwen`.

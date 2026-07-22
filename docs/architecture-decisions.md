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
`src.indexing/config.yaml` owns which embedding providers are enabled for this
project. Retrieval builds `VectorChunkRetriever` instances from the enabled
provider list and checks Chroma readiness without owning provider definitions.

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

`just eval-index qwen` rebuilds the default production chunk-vector provider.

## Open Knowledge Format

The OKF experiment is a second knowledge representation over the same canonical
raw pages, not another chunk retriever.

- `page_metadata` and `page_markdown_content` are its read-only source.
- Azure Foundry discovers conservative page proposals, resolves them against a
  global canonical catalog, and enriches the resulting fixed concepts.
- Existing durable concepts seed canonicalization; exact normalized titles are
  resolved before model-assisted batched resolution.
- Generated concepts use minimal OKF frontmatter, retain source URLs in the
  standard body citations section, and keep source page IDs only for evaluation.
- Discovery inventory, canonical catalog, and enrichment checkpoints stay under
  `.local/okf/`; the validated bundle lives in `data/okf/tourism/`.
- Online answering follows bundle indexes and reads complete concept files; it
  does not reopen raw database pages or select source-text windows.
- Normal generation resumes incrementally. A deliberate clean rebuild deletes
  the bundle and all private OKF state first, preventing stale concepts and old
  generated prose from seeding a redesigned catalog.
- Shared retrieval evaluation projects gold and retrieved chunks through their
  source pages to OKF concept IDs. This credits translated and duplicate sibling
  pages when canonicalization assigns them to the same concept.

Reason: exact-page scoring penalizes valid alternate sources, while direct
concept-to-chunk comparison conflates representation granularity.

## OKF Versus RAG Benchmarking

Benchmarking is split into clean build, incremental update, online
retrieval/navigation, shared page-evidence retrieval, and end-to-end answer
quality. There is no composite OKF-versus-RAG score.

- Both representations use one frozen and hashed raw-page snapshot.
- Shared scraping and Markdown extraction are excluded from representation
  build time.
- RAG chunk qrels remain valid for RAG-only retrieval studies.
- Shared retrieval derives golden concepts from existing chunk qrels and OKF
  `source_page_ids`, then projects every method to the same concept ontology.
- Build results include wall time, throughput, coverage, retries, model usage,
  cost, memory, artifact size, and storage amplification.
- Query results include p50/p95/p99 latency, throughput, failures, context use,
  calls, and steps.
- Answer results include factual correctness, faithfulness, citation validity
  and entailment, abstention, and blinded paired judgments.
- Raw per-item observations are retained and paired bootstrap confidence
  intervals accompany material quality claims.

Reason: industry IR and search benchmarks report effectiveness together with
latency, throughput, build cost, and storage. The complete protocol and sources
are in [`experiments/indexing/README.md`](../experiments/indexing/README.md).

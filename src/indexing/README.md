# Indexing

Builds local vector indexes for document chunks.

The indexer boundary is provider-shaped: English MiniLM, Qwen multilingual,
Qwen 4B, and Nemotron 3 Embed 1B all expose the same `embed_documents` /
`embed_query` methods from `src.shared.indexers`. Every provider runs locally
through sentence-transformers.

Chunk vector collections are derived state in Chroma. SQLite remains the source
of truth for page metadata, chunk text, and eval labels. `src.vector_store.chunks`
owns Chroma mechanics and query-time vector search; indexing only orchestrates
rebuilds.

Current chunk collections are storage names, not public retrieval method names:

```text
page_chunks_{english,qwen,qwen4b,nemotron}_chunk
```

Two independently stored chunk-representation versions are available:

- `v1` is the unchanged legacy `TitleHeadingChunkText` representation and keeps
	the existing `page_chunks_{provider}_chunk` collection names.
- `v2` uses `MetadataContextChunkText`: labeled title, heading, source language,
	source collection, page kind, and chunk content. It writes separate
	`page_chunks_v2_{provider}_chunk` collections. The retrieved evidence remains
	the original chunk text.

Both versions use the same chunk boundaries and IDs. The version changes only
the text indexed for dense and sparse retrieval, allowing paired comparisons.

Model and collection settings live in `config.yaml`.

Rebuild one collection:

```bash
uv run python -m src.indexing.chunks --method qwen
```

Build the metadata-context collection without replacing v1:

```bash
uv run python -m src.indexing.chunks --method qwen --chunk-version v2
```

The Nemotron collection uses `nvidia/Nemotron-3-Embed-1B-BF16`, its saved
query/document prompts, BF16 weights, SDPA attention, and a 4096-token indexing
limit. It requires a CUDA-capable NVIDIA GPU for practical inference. Once the
model is cached, rebuild the independent 2048-dimensional collection with:

```bash
uv run python -m src.indexing.chunks --method nemotron
```

Rebuild the default regenerable vector cache:

```bash
just eval-index qwen v1
just eval-index qwen v2
```

Durable reference databases belong under `data/db/`. Regenerable Chroma cache
artifacts belong under `data/cache/chroma/`. Use `.env` overrides when a run
should write to private scratch paths under `.local/`.

## Benchmarking against OKF generation

A Chroma build and an OKF bundle build have different products. Compare them
from the same frozen full-page corpus using clean-build wall time, pages and
source MiB per second, coverage, retries, model calls/tokens/cost, peak memory,
artifact size, and storage amplification. Keep chunking, embedding, persistence,
OKF discovery, canonicalization, enrichment, index generation, and validation
as separately timed stages. Also run incremental updates and attach retrieval
and answer quality to every efficiency result.

See [`experiments/indexing/README.md`](../../experiments/indexing/README.md) for the full
protocol derived from BEIR, ANN benchmark, RAG evaluation, and production search
benchmark practices.

# Indexing

Builds local vector indexes for document chunks.

The indexer boundary is provider-shaped: Qwen multilingual,
Qwen 4B, and Nemotron 3 Embed 1B all expose the same `embed_documents` /
`embed_query` methods from `src.shared.indexers`. Every provider runs locally
through sentence-transformers.

Chunk vector collections are derived state in Chroma. SQLite remains the source
of truth for page metadata, chunk text, and eval labels. `src.vector_store.chunks`
owns Chroma mechanics and query-time vector search; indexing only orchestrates
rebuilds.

Current chunk collections are storage names, not public retrieval method names:

```text
page_chunks_{qwen,qwen4b,nemotron}_chunk
```

Each chunk is embedded as its page title, heading breadcrumbs, and chunk text,
joined by newlines. The retrieved evidence is always the original chunk text.

Model and collection settings live in `config.yaml`.

Rebuild one collection:

```bash
uv run python -m src.indexing.chunks --method qwen
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

Durable reference databases belong under `$LLMS4EU_DATA/db/`. Regenerable Chroma
cache artifacts belong under `$LLMS4EU_DATA/chroma/`. Point `LLMS4EU_DATA` at a
private path when a run should not touch the shared store.

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

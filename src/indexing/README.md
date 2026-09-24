# Indexing

Builds local vector indexes for document chunks.

The indexer boundary is provider-shaped: every entry under `providers` in
`config.yaml` (Qwen3-Embedding 0.6B/4B/8B, Nemotron 3 Embed 1B/8B) exposes the same `embed_documents` /
`embed_query` methods from `src.indexing.embedders`. Every provider runs locally
through sentence-transformers.

Chunk vector collections are derived state in Chroma. SQLite remains the source
of truth for page metadata, chunk text, and eval labels. `src.indexing.store`
owns Chroma mechanics and query-time vector search; indexing only orchestrates
rebuilds.

Current chunk collections are storage names, not public retrieval method names:

```text
page_chunks_<provider>_chunk
```

Each chunk is embedded as its page title, heading breadcrumbs, and chunk text,
joined by newlines. The retrieved evidence is always the original chunk text.

Model and collection settings live in `config.yaml`.

Rebuild one collection:

```bash
uv run python -m src.indexing --method qwen
```

Adding a model is a config entry: the keys are `SentenceTransformerIndexer`
fields (`model_name`, `batch_size`, `dtype`, `attn_implementation`, `revision`,
`local_files_only`). The Nemotron models are pinned to a revision and load in
BF16 with SDPA attention; the 8B models need a CUDA GPU.

Rebuild the default regenerable vector cache:

```bash
just index qwen
```

Durable reference databases belong under `$LLMS4EU_DATA/db/`. Regenerable Chroma
cache artifacts belong under `$LLMS4EU_DATA/chroma/`. Point `LLMS4EU_DATA` at a
private path when a run should not touch the shared store.

# Indexing

Builds local vector indexes for document chunks.

The indexer boundary is provider-shaped: English MiniLM, Qwen multilingual,
Qwen 4B, and Azure embeddings all expose the same `embed_documents` /
`embed_query` methods from `src.shared.indexers`.

Chunk vector collections are derived state in Chroma. SQLite remains the source
of truth for page metadata, chunk text, and eval labels. `src.vector_store.chunks`
owns Chroma mechanics and query-time vector search; indexing only orchestrates
rebuilds.

Current chunk collections are storage names, not public retrieval method names:

```text
page_chunks_{english,qwen,qwen4b,azure}_chunk
```

Chunk indexing embeds title, heading path, and chunk text via
`TitleHeadingChunkText`.

Model and collection settings live in `config.yaml`.

Azure embeddings are batched. Larger batches reduce request-per-minute pressure,
but total token usage is unchanged.

Rebuild one collection:

```bash
uv run python -m src.indexing.chunks --method qwen
```

Azure indexing prompts for typed confirmation before sending embedding requests.
Use `--yes` only for deliberate non-interactive runs.

Rebuild the default regenerable vector cache:

```bash
just eval-index qwen
```

Durable reference databases belong under `data/db/`. Regenerable Chroma cache
artifacts belong under `data/cache/chroma/`. Use `.env` overrides when a run
should write to private scratch paths under `.local/`.

# Indexing

Builds and queries local vector indexes for document chunks.

The indexer boundary is provider-shaped: English MiniLM, Qwen multilingual, and
Azure embeddings all expose the same `embed_documents` / `embed_query` methods
from `src.shared.indexers`.

Chunk vector collections are derived state in Chroma. SQLite remains the source
of truth for page metadata, chunk text, summaries, and eval labels.
Chroma stores only embeddings and minimal ids; retrieval hydrates chunk text
from SQLite before returning results or reranking candidates.

Current chunk collections are named as:

```text
page_chunks_{english,qwen,azure}_{chunk,summary,chunk_summary}
```

The content modes mean:

- `chunk`: embed title, heading path, and chunk text.
- `summary`: embed title, heading path, and summary; return the original chunk.
- `chunk_summary`: embed title, heading path, summary, and chunk text.

Model and collection settings live in `config.yaml`.

Rebuild one collection:

```bash
uv run python -m src.indexing.chunks --method qwen
```

Rebuild one content mode:

```bash
uv run python -m src.indexing.chunks --method qwen --content-mode summary
```

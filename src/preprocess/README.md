# Preprocess

Builds derived artifacts from SQLite.

`chunks.py` turns scraped Markdown pages into stable, heading-aware page chunks.
Chunking is reusable preprocessing for indexing, retrieval, and eval.
It appends chunks for newly scraped pages and leaves already chunked pages alone,
so existing eval labels keep pointing at valid chunk ids.

Chunk summaries were useful retrieval experiments, but are not part of the
steady-state page chunk path. Historical summary results live in
`docs/retrieval-results.md`.

`index.py` embeds each place text with `sentence-transformers/all-MiniLM-L6-v2`
and recreates the `places` Chroma collection. MiniLM is small, local, and fast
enough for place search.

Chroma stores vectors plus ids. Full place rows and canonical page chunk text
stay in SQLite.

Place and chunk vector collections are rebuilt from scratch because datasets are
small and Chroma is derived state.

Rebuild the Chroma vector index from SQLite:

```python
from src.preprocess.index import rebuild_vector_index

rebuild_vector_index()
```

Inside `rebuild_vector_index()`, Chroma upsert inserts one vector point per
place. Each point stores the embedding and only this metadata:

```python
{"id": place.id}
```

# Vector Store

Chroma operations for derived chunk vector indexes.

SQLite owns canonical page chunk text. Chunk collections store embeddings and
minimal ids; `src.vector_store.chunks` hydrates chunk text from SQLite at query
time and reports readiness only when the collection exists and its count matches
`page_chunks`.

```python
from src.vector_store.chunks import query_chunk_vectors, rebuild_chunk_collection
```

Collections are rebuilt from scratch: the dataset is small and Chroma is derived
state.

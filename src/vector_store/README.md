# Vector Store

Chroma operations for derived vector indexes.

SQLite owns the full place rows. Chroma stores embeddings plus `{"id": place_id}`
so RAG can search by vector similarity, then fetch full text from SQLite.

SQLite also owns canonical page chunk text. Chunk collections store embeddings
and minimal ids; `src.vector_store.chunks` hydrates `PageChunk.text` from SQLite
at query time and reports readiness only when the collection exists and its
count matches `page_chunks`.

Main functions:

```python
from src.vector_store.places import recreate_places_collection, search_place_vectors
from src.vector_store.chunks import query_chunk_vectors, rebuild_chunk_collection
```

Chroma build time, collection size, readiness coverage, query latency, and
throughput are RAG-side benchmark observations. OKF has no vector collection;
its comparable offline artifact is the validated concept bundle and its online
operation is hierarchical navigation. Compare quality-efficiency frontiers
rather than vector QPS alone. See
[`experiments/indexing/README.md`](../../experiments/indexing/README.md).

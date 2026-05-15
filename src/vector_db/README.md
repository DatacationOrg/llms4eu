# Vector DB

Chroma operations for the derived place vector index.

SQLite owns the full place rows. Chroma stores embeddings plus `{"id": place_id}`
so RAG can search by vector similarity, then fetch full text from SQLite.

Main functions:

```python
from src.vector_db.places import recreate_places_collection, search_place_vectors
```

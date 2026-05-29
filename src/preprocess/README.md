# Preprocess

Builds derived vector data from SQLite.

`chunks.py` turns scraped Markdown pages into stable, heading-aware page chunks.
Chunking belongs here because it is a reusable preprocessing step, not an eval
concern.

`summaries.py` fills missing chunk summaries through the Azure Foundry chat
endpoint configured in `.env`.

Embeds each place text with `sentence-transformers/all-MiniLM-L6-v2` and
recreates the `places` Chroma collection. MiniLM is small, local and fast enough
for this retrieval sketch.

Chroma is used as the local vector index because it stays inside the Python
environment. It stores vectors plus `id`; full place text stays in SQLite.

The collection is rebuilt from scratch because the dataset is tiny and Chroma is
derived state.

Rebuild the Chroma vector index from SQLite:

```python
from src.preprocess.index import rebuild_vector_index

rebuild_vector_index()
```

Inside `rebuild_vector_index()`, Chroma upsert inserts or updates one vector
point per place. Each point stores the embedding and only this metadata:

```python
{"id": place.id}
```

# RAG

Place search and answer generation.

Place search embeds the query, asks Chroma for ids, then fetches full rows from
SQLite. Answer generation retrieves places, prints inspected context, then calls
a local structured-output LLM.

Config lives in `config.yaml`.

Retrieve from Chroma, then fetch rows from SQLite:

```python
from src.rag.search import search_places

places = search_places("quiet forest walk near water", limit=10)
print(places[0].score, places[0].summary)
```

Place search flow:

```text
query text -> MiniLM embedding -> Chroma top-k search -> ids -> SQLite rows
```

Reusable chunk retrievers live in `src.retrieval`.

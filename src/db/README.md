# DB

Initializes and queries local SQLite databases. Vector indexing lives outside
this package.

`initialize_db()` runs `sql/init.sql`, validates JSONL rows with Pydantic,
truncates local `places`, then inserts `data/places.jsonl`.

Read-only place queries live in `places.py`.

Raw scraped pages, page chunks, and eval labels live in `data/db/pages.db`.
Connection and schema helpers for that database live in `pages.py`, because
chunking and indexing need those artifacts outside eval.

OKF generation opens the raw-page tables in this database read-only and consumes
complete Markdown, never `page_chunks`. For comparative runs, freeze and hash
one database snapshot and record the exact eligible page IDs so RAG and OKF see
the same corpus. See
[`experiments/indexing/README.md`](../../experiments/indexing/README.md).

If a field is added, update both `sql/init.sql` and `src/shared/schema.py`.

Read places from SQLite:

```python
from src.db.places import load_places

places = load_places()
print(places[0].summary)
```

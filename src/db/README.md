# DB

Initializes the local SQLite table from the tracked fixture. In this repo,
`db` means source rows only; vector indexing lives in `src/preprocess`.

`initialize_db()` runs `sql/init.sql`, validates JSONL rows with Pydantic,
truncates local `places`, then inserts `data/places.jsonl`.

Read-only place queries live in `places.py`.

If a field is added, update both `sql/init.sql` and `src/shared/schema.py`.

Read places from SQLite:

```python
from src.db.places import load_places

places = load_places()
print(places[0].summary)
```

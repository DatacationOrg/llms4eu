# Data

`places.jsonl` is tracked dummy data for local development.

Each row must match `src.shared.schema.Place`: `id`, `place_description`, and
`summary`. `initialize_db()` treats this file as the whole local fixture and
resets the `places` table before inserting it.

`db/pages.db` is the canonical reference SQLite database for scraped pages,
chunks, and eval labels.

`cache/` holds regenerable artifacts such as Chroma vector indexes. It is not
tracked. Rebuild the default qwen vector cache with:

```bash
just eval-index qwen
```

Scraped raw text outputs are written under `data/scraped/` and are not tracked.

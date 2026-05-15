# Data

`places.jsonl` is tracked dummy data for local development.

Each row must match `src.shared.schema.Place`: `id`, `place_description`, and
`summary`. `initialize_db()` treats this file as the whole local fixture and
resets the `places` table before inserting it.

Scraped raw text outputs are written under `data/scraped/` and are not tracked.

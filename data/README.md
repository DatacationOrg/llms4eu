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

Large raw benchmark observations belong under `.local/`; reviewed aggregate
reports belong under `docs/`.

`geo/` holds the Eurostat NUTS 2024 boundaries (`NUTS_RG_03M_2024_4326.geojson`,
1:3M, all levels, ~16 MB). It is not tracked; fetch it with `just fetch-nuts`.
Page locations are derived against it, so re-running `just locate-pages` after a
NUTS revision recomputes the region codes from the stored coordinates.

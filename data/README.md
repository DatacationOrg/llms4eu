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

`okf/tourism/` is the generated Open Knowledge Format bundle built from complete
pages in `db/pages.db`. Each concept carries source page ids, URLs, source names,
and languages in frontmatter. OKF generation and navigation do not consume the
derived `page_chunks` table.

Resumable OKF discovery, canonical-catalog, and enrichment state is private
derived state under `.local/okf/`, not part of this directory. The validated OKF
bundle is durable because it is human-readable and can be consumed without the
generation service.

Benchmark runs must identify the exact `pages.db` snapshot used by both RAG and
OKF. Large raw observations belong under `.local/`; reviewed aggregate reports
belong under `docs/`. The expected benchmark record layout and metrics are in
[`experiments/indexing/README.md`](../experiments/indexing/README.md).

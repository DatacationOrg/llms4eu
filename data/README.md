# Data

`brestanica.json` is the tracked source list: 176 Slovenian tourism URLs as
`{source, url}` rows, the input to `just fetch-pages`. It is Jernej Hribar's
list and the labelled gold-standard corpus.

The expanded corpus (these 176 plus nine localities in eight countries,
1,338 pages, 2026-09-13) is listed in `/data/llms4eu/corpus-urls.json`, fetched
with `just fetch-pages /data/llms4eu/corpus-urls.json`.
Their rows also carry a `language`, which fetching ignores: each page's language
is read from the page itself.

Everything derived lives outside the repo under `$LLMS4EU_DATA` (default
`/data/llms4eu`): `db/pages.db` holds scraped pages, chunks and eval labels;
`chroma/` holds regenerable vector indexes; `embeddings/` and `checkpoints/`
hold run caches.

`/data/llms4eu/db/pages.db` is the expanded corpus (1,338 pages, 11,181 chunks,
688 located pages) and has no eval questions yet. `/data/llms4eu/archive/` is
the store before the expansion: `db/pages.db` there holds the 176 Brestanica
pages with the **3,476 approved eval questions and their labels**, the only
copy. The rest of `archive/` is Gerson's working material (per-locality seed
lists, NUTS boundaries, OKF data, caches).

Reviewed aggregate reports belong under `docs/`.

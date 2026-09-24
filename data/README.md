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

`/data/llms4eu/archive/` holds Gerson's stores from before the fresh start:
`db/pages.db` (the 176 Brestanica pages with 3,476 approved eval questions and
labels), `db-expanded/pages.db` (his 1,338-page fetch of 2026-09-16), and his
working material (per-locality seed lists, NUTS boundaries, OKF data, caches).
Both databases use main's older schema; the pipeline here builds its own
`db/pages.db` from the URL lists.

Reviewed aggregate reports belong under `docs/`.

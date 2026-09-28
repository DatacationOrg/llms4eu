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

`/data/llms4eu/wiki/` is a separate store for the Wikipedia places corpus
(`just wiki-*`): `urls.jsonl` (seed rows with QID, country, point, categories),
`db/pages.db` (its own page database, so the main corpus is never touched),
`fetch.log`, `wikidata/` (cached query answers, only needed to re-collect) and
the exported `pages.jsonl`.

`pages.jsonl` is complete as of 2026-09-26: 133,164 rows, one per EU27 castle,
lake, cave, mountain, park, … (a Wikidata item), each its Wikipedia article in
the place's own country language, extracted with trafilatura. Fields: `id`,
`wikiname`, `wikidata_id`, `title`, `url`, `final_url`, `in_language`
(requested), `language` (declared by the page), `language_ok`, `country`,
`country_languages`, `latitude`, `longitude`, `sitelinks`, `categories`,
`char_count`, `word_count`, `fetched_at`, `text`. Languages by rows: sv 68,955,
de 15,145, fi 13,003, fr 6,081, it 5,408, pl 5,351, es 3,010, lt 2,859, sk 2,579,
cs 2,309, et 1,446, nl 1,122, then 14 more under 1,000. Before use: keep
`language_ok` rows (one Swedish stub was replaced by a linked PDF); most
Swedish and many Finnish rows are bot-written lake stubs, unfiltered; there
are no groups of related places yet (`wikidata_id` is the only identity).

Reviewed aggregate reports belong under `docs/`.

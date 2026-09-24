# Data

`brestanica.json` is the tracked source list: 176 Slovenian tourism URLs as
`{source, url}` rows, the input to `just fetch-pages`. It is Jernej Hribar's
list and the labelled gold-standard corpus.

The expanded corpus (nine more localities in eight countries, 2026-09-13) stays
on the shared store: `/data/llms4eu/data/seeds/<cluster>.json` and
`/data/llms4eu/v2/seeds/corpus-v2.json`, fetched with `just fetch-pages <file>`.
Their rows also carry a `language`, which fetching ignores: each page's language
is read from the page itself.

Everything derived lives outside the repo under `$LLMS4EU_DATA` (default
`/data/llms4eu`): `db/pages.db` holds scraped pages, chunks and eval labels;
`chroma/` holds regenerable vector indexes; `embeddings/` and `checkpoints/`
hold run caches.

Reviewed aggregate reports belong under `docs/`.

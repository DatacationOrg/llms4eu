# Data

`brestanica.json` is the tracked source list: 176 Slovenian tourism URLs as
`{source, url}` rows, the input to `just fetch-pages`.

Everything derived lives outside the repo under `$LLMS4EU_DATA` (default
`/data/llms4eu`): `db/pages.db` holds scraped pages, chunks and eval labels;
`chroma/` holds regenerable vector indexes; `embeddings/` and `checkpoints/`
hold run caches.

Reviewed aggregate reports belong under `docs/`.

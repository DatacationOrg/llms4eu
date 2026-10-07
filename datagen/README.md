# Data generation (not used by the pipeline)

How the wiki dataset was made, kept as a record. Nothing in `src/` imports from here;
the pipeline only reads the resulting files through `src/db/dataset.py`. Exact reruns
are not possible: Wikipedia changes, and the free models are stochastic and change
behind their endpoints.

| folder / script | made | how |
|---|---|---|
| `corpus/wiki_places.py` | `urls.jsonl`, `pages.jsonl` | Wikidata SPARQL (12 place types x EU27, `corpus/config.yaml`), then the repo's scraper |
| `wiki_qa/` | `qa/*.parquet` | 52 scripts in 12 stages, see `wiki_qa/README.md` |

Where the pages come from: Wikidata, one local-language Wikipedia article per place.
Not from a Hugging Face dataset: `audiala/audiala-places` was checked first (places with
Wikidata ids but no text, 11 languages), and `ilsp/llms4eu_synthetic_qa` was only the
model for the page metadata's field names.

Inputs on thebeast, under `/data/llms4eu/wiki/`:

- `wikidata/`: the 324 saved SPARQL answers. `collect` reads them instead of asking
  again, and rebuilds `urls.jsonl` byte for byte.
- `urls.jsonl`, `collect.log`, `fetch.log`, `db/pages.db` (the fetch as it ran,
  in the old SQLite page store), `pages.jsonl`.
- `datagen/wiki_qa/`: LoRA adapters, SFT training sets, label pools and every
  blind check's input and result.

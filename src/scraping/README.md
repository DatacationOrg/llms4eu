# Scraping

Scraping fetches tourism pages, converts them to Markdown, and plugs results
into the local SQLite and Chroma flow.

`fetch_pages.py` reads JSON/JSONL `{source, url}` rows, fetches each page, turns
the result into Markdown with Trafilatura, and saves that final Markdown in
SQLite. If normal fetch + extraction produces empty/suspicious Markdown, it
renders the page with Playwright and runs the same extraction again.

Listing/index pages are saved as structural Markdown from repeated items in the
main content instead of forcing them through prose extraction. The chosen
`page_kind` is stored with page metadata.

If a page still has no prose but links to a document from its main content,
Docling downloads the first supported document and converts it to Markdown.
Document fallback is capped at 20 MB and 100 pages, with OCR disabled by default.

Stored page kinds are `prose`, `listing`, `document`, and `empty`.

Fetch, fallback, document, and listing thresholds live in `config.yaml` under
`fetch_pages`.

A seed row may carry a `language`; `fetch_pages.py` records it in
`page_sources`, so a source from another country does not inherit the
configured Slovenian default.

## Seed files

`data/brestanica.json` is the reference seed: one small heritage locality seen
from four kinds of source: the attraction's own website (`castle_rajhenburg`),
the town's website (`brestanica_webpage`), the national biographical lexicon's
entries for people from the area (`svn_biography`), and the national-language
Wikipedia articles the town article links to (`wikipedia`).

`seed_urls.py` builds seeds of the same shape for localities in neighbouring
countries, declared in `seeds.yaml`. Per cluster it discovers the site URLs
(sitemaps, else a shallow crawl), the Wikipedia articles (the seed articles'
wikilinks in reading order plus a people category) and the lexicon entries
(Wikidata: people born in the town or its district that have an entry, most
linked first, URL from the property's formatter), fetches every candidate once
and keeps those that answer. Output is `data/seeds/<cluster>.json` and the
merged `data/eu_neighbours.json`:

```bash
just seed-urls                      # all clusters
just seed-urls --cluster hr_zagorje # one cluster, no merged file
just fetch-pages data/eu_neighbours.json
```

Single-site sources also need a `source_locations` entry in
`src/preprocess/config.yaml`, the Wikidata item the geo tier assigns to every
page of that source.

The Brestanica sources are the gold standard. `seed_quality.py` scores each
new source against the Brestanica source of its kind (the `reference` map in
`seeds.yaml`) on page count, median length, prose share, stub share,
duplicate pages and detected-versus-declared language, and flags a source
that is thin, stub-heavy or duplicated relative to its reference. Reports
live under `docs/reports/scraping/`.

```bash
just seed-quality --output docs/reports/scraping/seed-quality-$(date +%F).md
just seed-quality --prune-below 300   # drop new-source stubs; never a reference page
```

`scraper.py` crawls sites and can save raw `.txt` outputs under `data/scraped/`.

`transform.py` turns scraped site/page content into `Place` rows.

`ingest.py` upserts scraped places into SQLite without truncating existing rows.

`scrape.py` orchestrates scrape -> transform -> upsert -> reindex.

`web/` contains the local FastAPI interface for running scrape jobs.

Scraping and Markdown extraction are common upstream preparation for both chunk
RAG and OKF. Comparative representation benchmarks freeze their output and do
not charge this shared work to either build. Record the source database hash,
eligible page IDs, page count, languages, and source bytes as the benchmark
corpus manifest. See
[`experiments/indexing/README.md`](../../experiments/indexing/README.md).

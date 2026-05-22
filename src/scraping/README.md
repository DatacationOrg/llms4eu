# Scraping

Scraping vendors the existing crawler and small web UI, then plugs the results
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

`scraper.py` crawls sites and can save raw `.txt` outputs under `data/scraped/`.

`transform.py` turns scraped site/page content into `Place` rows.

`ingest.py` upserts scraped places into SQLite without truncating existing rows.

`scrape.py` orchestrates scrape -> transform -> upsert -> reindex.

`web/` contains the local FastAPI interface for running scrape jobs.

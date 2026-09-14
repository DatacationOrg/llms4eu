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

Scraping and Markdown extraction are the shared upstream stage for every
retrieval experiment. Freeze their output and record the source database
hash, page count and languages as the benchmark corpus manifest.

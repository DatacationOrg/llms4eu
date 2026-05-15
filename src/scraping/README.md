# Scraping

Scraping vendors the existing crawler and small web UI, then plugs the results
into the local SQLite and Chroma flow.

`scraper.py` crawls sites and can save raw `.txt` outputs under `data/scraped/`.

`transform.py` turns scraped site/page content into `Place` rows.

`ingest.py` upserts scraped places into SQLite without truncating existing rows.

`scrape.py` orchestrates scrape -> transform -> upsert -> reindex.

`web/` contains the local FastAPI interface for running scrape jobs.

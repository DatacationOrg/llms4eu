# DB

Connection and schema helpers for the page database.

Scraped pages, page chunks, and eval labels live in `$LLMS4EU_DATA/db/pages.db`.
`pages.py` owns the connection and applies `sql/raw_pages.sql` and
`sql/eval.sql`; chunking, indexing and eval all read through it.

Vector indexing lives in `src/indexing`.

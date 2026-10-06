# SQL

`raw_pages.sql` stores scraped page metadata and final extracted Markdown. Raw
HTML is not stored in the working scrape database. `page_kind` records whether
the saved Markdown came from prose, listing, document, or empty-page handling.
For successful rows, `content_hash` is the SHA-256 hash of the final stored
Markdown.

`eval.sql` stores generated eval questions and gold relevant chunk ids. Eval
labels belong beside page chunks because relevance points to `page_chunks.id`.
`page_chunks.variant` keeps alternative cuts of the pages side by side for
chunk-size experiments; `migrate_chunk_variants.sql` is the one-off, manual
rebuild that adds the column to an older database.

The SQL stays portable so moving to Postgres later should stay small.

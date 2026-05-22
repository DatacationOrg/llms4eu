# SQL

`init.sql` is the storage schema for the prototype.

`raw_pages.sql` stores scraped page metadata and final extracted Markdown. Raw
HTML is not stored in the working scrape database. `page_kind` records whether
the saved Markdown came from prose, listing, document, or empty-page handling.
For successful rows, `content_hash` is the SHA-256 hash of the final stored
Markdown.

SQLite owns full place data for now. The SQL stays portable so moving to
Postgres later should stay small. The vector index stores only the place `id`,
so changing place fields starts here and in `src.shared.schema.Place`.

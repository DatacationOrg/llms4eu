# Preprocess

`chunks.py` turns scraped Markdown pages into stable, heading-aware page chunks.
Chunking is reusable preprocessing for indexing, retrieval, and eval.

It appends chunks for newly scraped pages and leaves already chunked pages
alone, so existing eval labels keep pointing at valid chunk ids.

SQLite owns the canonical chunk text; Chroma collections are derived indexes
over it.

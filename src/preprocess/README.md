# Preprocess

`chunker.py` turns scraped Markdown pages into heading-aware page chunks.
Chunking is reusable preprocessing for indexing, retrieval, and eval.

`config.yaml` sets `chunk_size` and `chunk_overlap` in characters (one chunk
set serves every embedder's tokenizer). Paragraphs are packed per section up to
`chunk_size`; a longer paragraph is split at sentence, then word boundaries.
Consecutive chunks share up to `chunk_overlap` characters of whole trailing
paragraphs. A chunk under `chunk_min_chars` is merged into the one before it,
with its heading path as a line, so no page text is dropped.

`just chunk` appends chunks for newly scraped pages and leaves already chunked
pages alone, so existing eval labels keep pointing at valid chunk ids.
`just rechunk` rechunks every page after a config change and moves the eval
labels onto the new chunks through their evidence quotes (see `src/eval`).

`locations.py` (`just locate-pages`) stores one point per page in
`page_locations`: a single-site source's configured Wikidata item
(`source_locations` in `config.yaml`), or a Wikipedia page's own article item.
Items without coordinates, and people, give no point; other pages stay
unlocated, which geo retrieval treats as neutral.

SQLite owns the canonical chunk text; Chroma collections are derived indexes
over it. A collection records a digest of the chunks it indexed, so after a
rechunk it reports itself stale until `just index` rebuilds it.

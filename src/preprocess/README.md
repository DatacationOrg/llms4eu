# Preprocess

`pages.py` (`just pages`) turns the scraped `pages.jsonl` into `wikipages.parquet`;
a rerun keeps the summaries already there.

`chunker.py` (`just chunk`) cuts every page of `wikipages.parquet` into
`chunks.parquet` at every size in `SIZES` (tokens of the default embedding model),
one row per chunk with its `size` (see `src/db/README.md`).

How a page is cut, following aihub-core's chunker:

1. Split at the `#`/`##`/`###` headings; drop Wikipedia edit links
   (`[Bearbeiten | Quelltext bearbeiten]`) and sections that held nothing else.
2. Pack neighbouring sections up to the size. A packed chunk's `breadcrumb` is the
   heading trail its sections share; the headings below it stay inline.
3. A section too big on its own is split at paragraph, line, then sentence ends.
4. A piece under 50 tokens joins its neighbour. No overlap.

The embedded text is `represent(title, breadcrumb, text)` in `src/db/schemas/chunk.py`.

Rerunning after a change to the chunker keeps every chunk whose embedded text did not
change, with its `role` and its vectors in every `embeddings/<provider>/<size>.npy`;
the other rows are NaN until `just index` embeds them.

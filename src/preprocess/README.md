# Preprocess

Builds derived artifacts from SQLite.

`chunks.py` turns scraped Markdown pages into stable, heading-aware page chunks.
Chunking is reusable preprocessing for indexing, retrieval, and eval.
It appends chunks for newly scraped pages and leaves already chunked pages alone,
so existing eval labels keep pointing at valid chunk ids.

## Chunkers

`chunkers.py` exposes `build_chunker(strategy, size, overlap, unit, provider)`
over `langchain-text-splitters`. `MarkdownHeaderTextSplitter` sections a page and
supplies the heading path; `RecursiveCharacterTextSplitter` fills it, sized in
characters or — via `from_huggingface_tokenizer` — in the embedding provider's
own tokens. Three things are project-specific and live here rather than in the
library: NFC normalization of the page before splitting, resolving each chunk's
character span (`add_start_index` returns `-1` whenever a piece is not found
verbatim), and reserving the v1 representation prefix from the size budget so
the *indexed document*, not merely the chunk text, fits the sequence limit.

Token sizing is what makes a size budget mean the same thing in every language.
Slovenian runs about 0.41 tokens per character against roughly 0.25 for English,
so a fixed character target silently produces different real sizes per language.
Measured on this corpus: the `base` character-sized cut has a median of 594 Qwen
tokens with 66% of chunks over the 512 limit; a 512-token cut has a median of
408 and a maximum of 494, so nothing is truncated.

`legacy_chunker.py` is frozen. It exists only to reproduce the `base` variant
byte-for-byte, which is what keeps the 3,476 approved labels, every published
report and every checkpoint valid. Verified 2026-08-11: all 726 base chunks are
reproduced with identical ids, text and character counts. New work uses
`recursive` or `markdown`.

## Variants

A *variant* is a cutting; a *version* is an embedding representation. They are
independent. Variants live side by side in `page_chunks`, discriminated by a
`variant` column, and `base` keeps its bare `{page_id}:{index}` ids.

Chunking work never runs against `data/db/pages.db`: `eval_relevant_chunks`
cascades on delete from `page_chunks`, so a re-cut there destroys the approved
labels. Snapshot first, then point `PAGES_DB_PATH` at the copy.

```bash
just chunk-sweep-db                      # data/db/pages.db -> .local/db/pages-variants.db
export PAGES_DB_PATH=.local/db/pages-variants.db
just chunk-variant tok512 512 0 tokens qwen
just relabel tok512                      # point the approved questions at it
just eval-index qwen v1 tok512           # its own Chroma collection
```

`--backfill-spans` fills `start_char`/`end_char` for chunks stored before spans
existed, writing a span only where the regenerated text matches the stored text
exactly.

## Languages

`languages.py` detects each page's language with `lingua`, restricted to the
EU-24. It is read-only until `--apply`, because language is written into the chunk
embedding text and into every chunk's Chroma metadata. Markdown boilerplate is
stripped first and at least 120 characters of prose are required: raw Markdown
makes an address-and-phone-number page look Croatian, and a 60-character list of
Slovenian church names cannot be told apart from Croatian at all.

Detection found this corpus is **not** monolingual: 12 `castle_rajhenburg` pages
are English at 1.00 confidence, though every source is stamped `sl`.

Use `just audit-chunk-tokens` to measure canonical chunks after v1 embedding
context and model document prompts are applied. The audit is read-only and does
not change chunk boundaries or positional IDs.

Chunk summaries were useful retrieval experiments, but are not part of the
steady-state page chunk path. Historical summary results live in
[`docs/reports/agentic/agentic-findings.md`](../../docs/reports/agentic/agentic-findings.md).

`index.py` embeds each place text with `sentence-transformers/all-MiniLM-L6-v2`
and recreates the `places` Chroma collection. MiniLM is small, local, and fast
enough for place search.

Chroma stores vectors plus ids. Full place rows and canonical page chunk text
stay in SQLite.

Place and chunk vector collections are rebuilt from scratch because datasets are
small and Chroma is derived state.

Rebuild the Chroma vector index from SQLite:

```python
from src.preprocess.index import rebuild_vector_index

rebuild_vector_index()
```

Inside `rebuild_vector_index()`, Chroma upsert inserts one vector point per
place. Each point stores the embedding and only this metadata:

```python
{"id": place.id}
```

Page chunking is a measured RAG build stage when benchmarking against OKF. OKF
starts from the same complete Markdown rows but does not run `chunks.py` or
consume `page_chunks`. Shared scraping and Markdown extraction are measured once
outside both representation builds. See
[`experiments/indexing/README.md`](../../experiments/indexing/README.md).

## Page locations

`just locate-pages` (`src/preprocess/locations.py`) writes `page_locations`: the
place each page is about, as coordinates plus derived ISO and NUTS codes. Three
tiers, cheapest first: Wikidata for Wikipedia pages, a configured QID per
single-site source, then an LLM naming places that Nominatim verifies. Read-only
until `--apply`; adds rows only, never touches `page_chunks`. Needs the NUTS
boundaries from `just fetch-nuts`. The 2026-09-08 dry run on the corpus: 89 of
176 pages get a primary location from the first two tiers alone; the 51
biographies and the concept pages (Bazilika, UTC+01:00, Nemščina) correctly
get none.

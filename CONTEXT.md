# Context

## Terms

**Page**
A Wikipedia page about an EU place (`wikipages.parquet`, `Page`): metadata and its
Markdown `text`, which is what gets chunked and searched.

**Markdown page**
The normalized text representation produced by scraping. It is the input to
chunking and should not depend on any retrieval method.

**Chunk**
A heading-aware slice of a page (`chunks.parquet`, `Chunk`), cut at several target
sizes (`SIZES`, see `src/db/README.md`). Indexing, retrieval and eval work on one
size at a time (`CHUNK_SIZE`).

**Summary and role**
Ling 3.1 Flash notes: a page `summary` (at most 3 sentences) and a chunk `role`
(one sentence on what kind of information the chunk adds to its page). Derived
columns for experiments, made by `datagen/`, not a replacement for the text.

**Indexer**
A provider-specific embedding backend behind a common interface. Current
indexers are Qwen3-Embedding 0.6B/4B/8B and Nemotron 3 Embed 1B/8B.

**Ranking method**
A named retrieval strategy used in experiments to return ranked chunk ids for a
query. Examples: sparse, vector-only, hybrid vector+sparse, and reranked
variants.

**Retriever**
Reusable application code that implements a ranking method. Retrievers belong to
the retrieval/RAG layer; eval compares them but does not own their
implementation.

**Eval dataset**
The wiki QA questions (`qa/wiki_qa_rag.parquet`): a gold page and evidence quotes
per question. A chunk is relevant when it is on the gold page and holds a quote.
Eval owns relevance and metrics; it should not own reusable pipeline steps.

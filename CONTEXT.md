# Context

## Terms

**Raw page**
A fetched source page stored in the local raw-pages SQLite database with metadata
and Markdown content.

**Markdown page**
The normalized text representation produced by scraping. It is the input to
chunking and should not depend on any retrieval method.

**Page chunk**
A heading-aware slice of a Markdown page. Current defaults target 1,800
characters, allow up to 2,600 characters for long paragraphs, and drop tiny
chunks below 300 characters when a page produces multiple chunks.

**Chunk variant**
One named way of cutting the pages. `base` is the cut the pipeline runs on;
other variants (`c900`, ...) are alternative cuts stored beside it, each with
its own chunk ids, labels and vector collections, so chunk size can be
compared without disturbing the pipeline.

**Chunk summary**
A short search-oriented description of a page chunk. It is derived content used
for past embedding and retrieval experiments, not a replacement for the chunk
text. Chunk summaries were removed from the steady-state database path after the
experiment.

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
Approved factual questions, answers, and gold chunk ids used to compare ranking
methods. Eval owns labels and metrics; it should not own reusable pipeline steps.

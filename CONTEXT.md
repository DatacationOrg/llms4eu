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

**Chunk summary**  
A short search-oriented description of a page chunk. It is derived content used
for embedding and retrieval experiments, not a replacement for the chunk text.

**Indexer**  
A provider-specific embedding backend behind a common interface. Current
indexers are English MiniLM, Qwen multilingual, and Azure embeddings.

**Ranking method**  
An evaluation-time retrieval strategy that returns ranked chunk ids for a query.
Examples: vector-only, BM25, vector+BM25 fusion, and reranked variants.

**Eval dataset**  
Approved factual questions, answers, and gold chunk ids used to compare ranking
methods. Eval owns labels and metrics; it should not own reusable pipeline steps.

# Retrieval Pipeline Plan

## Current Branch Delta

Compared with `main`, this branch adds an experimental retrieval-eval pipeline:
page chunking, chunk summaries, generated QA labels, BM25, vector retrieval,
hybrid reciprocal-rank fusion, Qwen reranking, and Chroma chunk indexes.

The branch also expands `.local/raw_pages.db`. Before pushing DB changes, run:

```bash
sqlite3 .local/raw_pages.db "vacuum;"
```

## Target Shape

The code should read as a pipeline, not as one large `eval` feature:

1. `src/scraping`: fetch pages and convert them to Markdown.
2. `src/db`: own SQLite access for places and raw-page artifacts.
3. `src/preprocess`: turn Markdown into reusable derived text artifacts, starting
   with page chunks.
4. `src/shared`: provider interfaces for structured LLM calls and embedding
   indexers.
5. `src/indexing`: build/query vector indexes for chunks with English MiniLM,
   Qwen multilingual, and Azure embeddings.
6. `src/eval`: own only eval labels, ranking method composition, metrics, and
   reports.

## Critical Notes

Chunks are not currently 1,000 characters. The implemented defaults target
1,800 characters, cap most splits at 2,600, and filter chunks under 300
characters when there is more than one chunk. That may be fine, but we should
evaluate it rather than assume.

For reranking, top 10 is too shallow as a candidate pool. Use top 50 as the
starting point, then report hit@1, hit@5, hit@10, mrr@10, and timing. Azure AI
Search semantic ranker uses the top 50 first-stage results, and retrieve-rerank
pipelines commonly use about 20-100 candidates before selecting the top 5-10.

Indexing “chunks and summaries, both” is now an explicit experiment. The vector
indexer can build separate collections for:

- `chunk`: title, heading path, and chunk text
- `summary`: title, heading path, and the generated summary
- `chunk_summary`: title, heading path, generated summary, and chunk text

`summary` retrieval still returns the original chunk text as evidence. The
summary is only a search representation, not the answer source.

Chroma is treated as a derived vector index, not a second document store. It
stores embeddings plus minimal ids, then retrieval hydrates the canonical chunk
text from SQLite.

## Proposed First Eval Matrix

Baseline methods:

- `bm25`
- `{english,qwen,azure}_{chunk,summary,chunk_summary}`
- `{english,qwen,azure}_{chunk,summary,chunk_summary}_bm25`

Reranked methods:

- `{english,qwen,azure}_{chunk,summary,chunk_summary}_rerank`
- `{english,qwen,azure}_{chunk,summary,chunk_summary}_rerank_hybrid`

Report latency per method as total time and average milliseconds per query.
Where local models are used, record device information separately so GPU and CPU
runs are not compared as if they were equivalent.

## Current Decision

Keep separate vector collections for chunk-only, summary-only, and
chunk-plus-summary. Compare all three before deciding which representation to
keep as the default retrieval path.

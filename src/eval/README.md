# Eval

Experimental retrieval evaluation over raw scraped Markdown pages.

The stored data is the eval dataset only: factual questions, short answers, and
gold chunk ids. Evaluation run results are printed in the terminal and not saved.

```bash
uv run python -m src.preprocess.chunks
uv run python -m src.eval.generate_dataset --summarize-missing
uv run python -m src.indexing.chunks --method qwen --content-mode chunk
uv run python -m src.indexing.chunks --method qwen --content-mode summary
uv run python -m src.indexing.chunks --method qwen --content-mode chunk_summary
uv run python -m src.eval.generate_dataset --limit 10
uv run python -m src.eval.generate_dataset --limit 10 --model gemma4:26b
uv run python -m src.eval.generate_dataset --limit 10 --reasoning
uv run python -m src.eval.inspect_dataset --limit 20
uv run python -m src.eval.reset_dataset
uv run python -m src.eval.import_questions reviewed.jsonl
uv run python -m src.eval.evaluate --methods qwen_chunk_summary
uv run python -m src.eval.evaluate --methods qwen_chunk,qwen_summary,qwen_chunk_summary,bm25
uv run python -m src.eval.evaluate --methods qwen_chunk_summary_bm25,qwen_chunk_summary_rerank_hybrid
```

`--summarize-missing` fills nullable `page_chunks.summary` values via the Azure
Foundry chat endpoint configured by `.env` (`AZURE_AI_ENDPOINT`,
`AZURE_AI_API_KEY`, `AZURE_AI_MODEL`). Summaries target about 40 words, use the
chunk's language, and preserve searchable names/facts. Summary retrieval embeds
the summary but still returns the original chunk text as evidence.

Embedding and reranker inference is local. `SentenceTransformer`/`CrossEncoder`
loads use `local_files_only=True`, so eval will not contact the Hugging Face Hub
at runtime. Download/cache model weights explicitly before offline eval, or
point the embedding models in `src/indexing/config.yaml` and the reranker model
in `src/eval/config.yaml` at local model directories. Azure-backed question
generation, Azure embeddings, and chunk summarization may call Azure when
selected.

The first report compares overall `hit@1`, `hit@5`, `hit@10`, and `mrr@10`.
The category report shows `hit@5` by question type.

Ranking methods live in `src/eval/ranking`. The public entrypoints are
`get_ranking_method(name)` and `available_ranking_methods()`, mirroring the
simple name-based style used elsewhere.

Current method names:

- `{english,qwen,azure}_{chunk,summary,chunk_summary}`: Chroma vector search
  over a specific embedding representation.
- `bm25`: local lexical BM25 over page chunks.
- `*_bm25`: reciprocal-rank fusion over one vector representation plus BM25.
- `*_rerank`: Qwen3 reranker over vector candidates.
- `*_rerank_hybrid`: Qwen3 reranker over fused vector/BM25 candidates.

The reranker currently receives the top 50 first-stage candidates. This follows
the common retrieve-and-rerank shape: retrieve broadly, rerank down to the top
5-10 consumed by the answerer.

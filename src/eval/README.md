# Eval

Experimental retrieval evaluation over raw scraped Markdown pages.

The stored data is the eval dataset only: factual questions, short answers, and
gold chunk ids. Evaluation run results are printed in the terminal and not saved.

```bash
uv run python -m src.eval.chunks
uv run python -m src.eval.vector_index
uv run python -m src.eval.generate_dataset --limit 10
uv run python -m src.eval.generate_dataset --limit 10 --model gemma4:26b
uv run python -m src.eval.generate_dataset --limit 10 --reasoning
uv run python -m src.eval.inspect_dataset --limit 20
uv run python -m src.eval.reset_dataset
uv run python -m src.eval.import_questions reviewed.jsonl
uv run python -m src.eval.evaluate --methods vector
uv run python -m src.eval.evaluate --methods vector,bm25,vector_bm25
uv run python -m src.eval.evaluate --methods bm25,vector_bm25,qwen3_rerank_hybrid
```

The first report compares overall `hit@1`, `hit@5`, `hit@10`, and `mrr@10`.
The category report shows `hit@5` by question type.

Ranking methods live in `src/eval/ranking`. The public entrypoints are
`get_ranking_method(name)` and `available_ranking_methods()`, mirroring the
simple name-based style used elsewhere.

Current method names:

- `vector`: Chroma vector search over page chunks.
- `bm25`: local lexical BM25 over page chunks.
- `vector_bm25`: reciprocal-rank fusion over vector and BM25 candidates.
- `qwen3_rerank`: Qwen3 reranker over vector candidates.
- `qwen3_rerank_hybrid`: Qwen3 reranker over fused vector/BM25 candidates.

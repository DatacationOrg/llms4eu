# Eval

Retrieval evaluation over scraped Markdown page chunks.

Eval owns factual questions, short answers, gold chunk ids, metrics, timings,
and reports. It compares retrievers from `src.retrieval`.

```bash
uv run python -m src.preprocess.chunks
uv run python -m src.indexing.chunks --method qwen
uv run python -m src.eval.generate_dataset --limit 10
uv run python -m src.eval.generate_dataset --limit 10 --model gemma4:26b
uv run python -m src.eval.generate_dataset --limit 10 --reasoning
uv run python -m src.eval.inspect_dataset --limit 20
uv run python -m src.eval.reset_dataset
uv run python -m src.eval.import_questions reviewed.jsonl
uv run python -m src.eval.evaluate --methods qwen
uv run python -m src.eval.evaluate --methods qwen4b,sparse
uv run python -m src.eval.evaluate --methods qwen --category crosslingual
```

Chunk summaries are historical retrieval experiments, not steady-state eval
inputs. Results live in `docs/retrieval-results.md`.

Embedding and reranker inference is local unless an Azure retriever or question
generator is selected. Azure index builds pause for typed `yes`.

Embedding calls are cached in `.local/embedding_cache.sqlite`. Chroma stores
derived indexes; SQLite stores canonical chunk text.

Eval reports `hit@1`, `hit@5`, `hit@10`, and `mrr@10`; it requests top 10 even
if an answer path uses a different limit.

Retriever method names and tuning live in `src/retrieval`.

# Eval

Retrieval evaluation over scraped Markdown page chunks.

Eval owns factual questions, short answers, gold chunk ids, metrics, timings,
and reports. It compares retrievers from `src.retrieval`.

```bash
just eval-chunks
just eval-index qwen
just eval-generate 10
just eval qwen
just eval-inspect 20
```

Useful direct commands for less common workflows:

```bash
uv run python -m src.eval.generate_dataset --limit 10 --model gemma4:26b
uv run python -m src.eval.generate_dataset --limit 10 --reasoning
uv run python -m src.eval.import_questions reviewed.jsonl
uv run python -m src.eval.reset_dataset
uv run python -m src.eval.evaluate --methods qwen --category crosslingual
```

Chunk summaries are historical retrieval experiments, not steady-state eval
inputs. Current results live in
[`docs/retrieval-results-agentic.md`](../../docs/retrieval-results-agentic.md).

All inference is local: embeddings, reranking, question generation, and the
agentic sufficiency judge run through sentence-transformers and Ollama.
Eval checks that requested vector indexes already exist and reports the build
commands when they are missing; it does not build indexes while measuring.

Embedding calls are cached in `.local/embedding_cache.sqlite`. Chroma stores
derived indexes; SQLite stores canonical chunk text.

Default eval compares the `qwen` retriever. Eval reports `hit@1`, `hit@5`,
`hit@10`, and `mrr@10`; it requests top 10 even if an answer path uses a
different limit.

Retriever method names and tuning live in `src/retrieval`.

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

Embedding and reranker inference is local unless an Azure retriever or question
generator is selected. Azure index builds pause for typed `yes`.
Eval checks that requested vector indexes already exist and reports the build
commands when they are missing; it does not build indexes while measuring.

Embedding calls are cached in `.local/embedding_cache.sqlite`. Chroma stores
derived indexes; SQLite stores canonical chunk text.

Default eval compares the `qwen` retriever. Eval reports `hit@1`, `hit@5`,
`hit@10`, and `mrr@10`; it requests top 10 even if an answer path uses a
different limit.

Retriever method names and tuning live in `src/retrieval`.

## OKF comparison boundary

These chunk qrels remain the correct benchmark for ranked RAG retrieval, but
they cannot directly score OKF concept IDs. The shared comparison derives a
golden concept from each gold chunk's page using OKF `source_page_ids`. RAG
rankings are projected through chunk → page → concept, while OKF rankings are
already concept IDs. This gives full concept recall to language variants and
sibling pages assigned to the same concept. Native chunk metrics remain
separate, as do end-to-end factual correctness, faithfulness, citation support,
latency, and cost.

The complete protocol is in
[`experiments/indexing/README.md`](../../experiments/indexing/README.md). Keep current
chunk-level reports separate from future page-evidence and answer-level reports.

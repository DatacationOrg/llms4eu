# Eval

Retrieval evaluation over scraped Markdown page chunks.

Eval owns factual questions, short answers, gold chunk ids, metrics, timings,
and reports. It compares retrievers from `src.retrieval`.

```bash
just chunk
just index qwen
just eval-generate 10
just eval --methods qwen
just eval-inspect 20
```

Less common workflows, run directly:

```bash
uv run python -m src.eval.generate_dataset --limit 10 --model gemma4:26b
uv run python -m src.eval.generate_dataset --limit 10 --reasoning
uv run python -m src.eval.evaluate --methods qwen --category cross_language
uv run python -m src.eval.evaluate --methods all --checkpoint
```

Two question types: `same_language`, asked in the page's own language, and
`cross_language`, asked in one other language sampled per chunk. The page's
language comes from what the page declares, not from a default.

Labels are chunk ids, which a rechunk destroys, so each question can also carry
a verbatim quote from its page (`eval_evidence`). `just eval-evidence` asks the
question model for it once and keeps it only if it appears in the gold chunk;
failures are left for a rerun. `just rechunk` refuses until every question has
one, rechunks, and `relabel` points each question at the chunks now containing
its quote (or, for a quote cut by a boundary, any of its sentences).

`--checkpoint` makes a long run resumable: finished methods are skipped on a
re-run, keyed on the question set so a different `--limit` or `--category`
starts fresh.

Eval scores one chunk variant at a time (`CHUNK_VARIANT`, default `base`; see
`src/preprocess`): only labels on that variant's chunks count, and a question
whose quote was not found in them is left out of that variant's run. Each
variant has its own checkpoint. `just chunk-compare base,c900 --methods qwen`
chunks, relabels, indexes and scores each variant and writes one table to
`.local/reports/`. Beside the usual metrics it shows each variant's chunk count,
mean chunk length and `chars@5`, the characters a reader gets back at k, because
`hit@k` on whole chunks favours a longer cut by construction; the earlier
chunk-size sweeps in `docs/reports/chunking/` explain that trade-off.

Eval checks that requested vector indexes already exist and reports the build
commands when they are missing; it does not build indexes while measuring.
All inference is local, through sentence-transformers and Ollama.

Embedding calls are cached in `$LLMS4EU_DATA/embeddings/cache.sqlite`. Chroma
stores derived indexes; SQLite stores canonical chunk text.

Eval reports `hit@1`, `hit@5`, `hit@10`, `recall@10` and `mrr@10`. Defaults
live in `config.yaml`; `just eval` overrides them with `--methods qwen`.
Retriever names and tuning live in `src/retrieval`. Result reports from past
runs are indexed in [`docs/README.md`](../../docs/README.md).

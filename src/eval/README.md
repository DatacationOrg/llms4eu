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

## Comparing chunkings

Two things make a chunk-size comparison mean what it looks like it means, and
neither is on by default in a single-variant eval.

**Question density.** `--density` asks a chunk for one question per 256 of its
own tokens, so a 256-token cut gets one, 512 two and 1,024 four, and every
variant ends up with the same number of questions over the same corpus. Without
it a chunk gets one question per type whatever its size, which probes the
smallest cut four times more densely than the largest (17.6 questions per 10,000
indexed characters against 4.2) and hands the largest the more salient facts as
well. `sweep_status` prints the density per variant so the claim is checkable.

**Retrieval cost.** `run_eval(..., store_metrics=True)` adds `store_share@k` —
the percentage of the variant's whole index one query returns — and
`recall_per_share@k`, which is `char_recall@k` divided by it. Every quality
metric here can be bought with chunk size; these are what charges a cut for the
text it returned. Lower is better for the share, higher for the ratio.

```bash
export PAGES_DB_PATH=.local/db/pages-variants.db   # never the durable database
just variant-questions tok1024                     # density-normalised
uv run python -m src.eval.generate_dataset --variant tok1024 --density --fill-missing
uv run python -m src.eval.sweep_status             # q/10k ch should match
just sweep-overview --design per-variant-density --variants tok256,tok512,tok1024 ...
```

The top-up needs `--density` too: without it it offers every type for every
chunk, restoring the per-type design with the counts still looking right. It is
worth running — the model delivers 49 of 49 requested questions for a 256-token
cut but 172 of 215 for a 1,024-token one, because one call asked for five pairs
fails more often than one asked for one.

Neither removes home turf: each variant's questions were still written from its
own chunks. Anchor them (`src.eval.anchors`) and project them onto every variant
(`src.eval.relabel`) to pool them into one shared set, and run the grid as
`--design shared`. Equal counts are what make that pool balanced.

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

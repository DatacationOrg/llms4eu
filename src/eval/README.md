# Eval

Retrieval evaluation on the wiki QA test set (`qa/wiki_qa_rag.parquet`, see
`src/db`). It compares retrievers from `src.retrieval`.

```bash
just chunk                                  # once, or after a chunker change
just index qwen                             # vectors for every chunk size
just eval --methods qwen,sparse --limit 500
CHUNK_SIZE=<size> just eval --methods qwen  # another chunk size
just chunk-compare --methods qwen           # every size in one table
```

Questions are the `answer_ok` rows of the `test` split, in a stable random order,
at most `question_limit` unless `--limit` says otherwise. `--category`
keeps one `qtype` (identify, location, number, date, ...).

A chunk is relevant to a question when it is on the question's page and holds one
of its evidence quotes (whitespace-normalised; a quote cut by a chunk boundary
labels each side by sentence). A question none of whose quotes is found is left out.

Eval reports `hit@1`, `hit@5`, `hit@10`, `recall@10` and `mrr@10`, the speed of
each method, and `hit@5` per question type. `--checkpoint` makes a long run
resumable: finished methods are skipped, keyed on the question set and chunk size.
`chunk-compare` adds each size's chunk count, mean length and `chars@5`, the
characters a reader gets back at k, because `hit@k` on whole chunks favours longer
chunks by construction.

Eval checks that the requested vectors exist and prints the `just index` command
when they do not. Defaults live in `config.yaml`. Reports go to `.local/reports/`;
past ones are indexed in [`docs/README.md`](../../docs/README.md).

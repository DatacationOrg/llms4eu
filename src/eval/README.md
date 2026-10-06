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

Eval checks that requested vector indexes already exist and reports the build
commands when they are missing; it does not build indexes while measuring.
All inference is local, through sentence-transformers and Ollama.

Embedding calls are cached in `$LLMS4EU_DATA/embeddings/cache.sqlite`. Chroma
stores derived indexes; SQLite stores canonical chunk text.

Eval reports `hit@1`, `hit@5`, `hit@10`, `recall@10` and `mrr@10`. Defaults
live in `config.yaml`; `just eval` overrides them with `--methods qwen`.
Retriever names and tuning live in `src/retrieval`. Result reports from past
runs are indexed in [`docs/README.md`](../../docs/README.md).

## Wiki places QA test set

Questions about 133k Wikipedia pages of EU places (24 languages), on thebeast:

| path | what |
|---|---|
| `/data/llms4eu/wiki/pages.jsonl` | the pages to index: `id`, `title`, `text`, metadata |
| `/data/llms4eu/wiki/qa/wiki_qa_*.parquet` | the questions, one file per kind |

| model in `wiki_qa.py` | rows | a question that |
|---|---|---|
| `Rag` | 531k | is about one page: easy (names the place) or hard (describes it) |
| `Unanswerable` | 31k | no page answers |
| `Compare` | 10k | needs two pages |
| `Meta` | 4k | has a set of pages as its answer (list, geo) |
| `Tables` | 339 | aggregates a table in a page |

Each model lists its file's columns. Test on `ok` rows (`answer_ok` in `Rag`), tune on `split == "dev"`.

```python
from src.eval.wiki_qa import Rag, Unanswerable, load, read, pages

hard = load(Rag, ["id", "question"], answer_ok=True, kind="challenge").to_pandas()  # fast, columns
for q in read(Unanswerable, ok=True):  # validated models
    print(q.question, q.why)
texts = pages(hard.id)  # page id -> page
```

Not confidential, copy it where you need it (`WIKI_QA_DIR=<copy of qa/>`); do not publish it.
How it was made: `docs/reports/wiki-qgen/wiki-qa-dataset.md`.

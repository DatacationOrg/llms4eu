# DB

The project's data and how to read it.

## Wiki places dataset (the main one)

Questions about 133k Wikipedia pages of EU places (24 languages), on thebeast:

| path | what |
|---|---|
| `/data/llms4eu/wiki/pages.jsonl` | the pages to index: `id`, `title`, `text`, metadata |
| `/data/llms4eu/wiki/qa/wiki_qa_*.parquet` | the questions, one file per kind |

| model (`schemas/<name>.py`) | rows | a question that |
|---|---|---|
| `Rag` | 531k | is about one page: easy (names the place) or hard (describes it) |
| `Unanswerable` | 31k | no page answers |
| `Compare` | 10k | needs two pages |
| `Meta` | 4k | has a set of pages as its answer (list, geo) |
| `Tables` | 339 | aggregates a table in a page |

Each model lists its file's columns; `wiki_qa.py` loads them. Test on `ok` rows (`answer_ok` in `Rag`), tune on `split == "dev"`.

```python
from src.db.wiki_qa import Rag, Unanswerable, load, read, pages

hard = load(Rag, ["id", "question"], answer_ok=True, kind="challenge").to_pandas()  # fast, columns
for q in read(Unanswerable, ok=True):  # validated models
    print(q.question, q.why)
texts = pages(hard.id)  # page id -> page
```

Not confidential, copy it where you need it (`WIKI_QA_DIR=<copy of qa/>`); do not publish it.
How it was made: `docs/reports/wiki-qgen/wiki-qa-dataset.md`.

## Legacy: Slovenian page database

Scraped pages, page chunks, and eval labels of the first Slovenian corpus live in
`$LLMS4EU_DATA/db/pages.db`. `legacy/pages.py` owns the connection and applies `sql/raw_pages.sql` and
`sql/eval.sql`; chunking, indexing and eval still read through it.

Vector indexing lives in `src/indexing`.

# DB

The project's data and how to read it.

## Wiki places dataset (the main one)

133k Wikipedia pages of EU places (24 languages), cut into chunks, embedded, and questions
about them. On thebeast under `/data/llms4eu/wiki/` (`WIKI_DIR` for a copy):

| path | model (`schemas/<name>.py`) | rows | what |
|---|---|---|---|
| `pages.parquet` | `Page` | 133k | the pages: `id`, `title`, `text` (Markdown), metadata, Ling `summary` |
| `chunks.parquet` | `Chunk` | 1.8M | the pages cut at 256 / 512 / 1024 / 2048 tokens (`size`), with `breadcrumb` and Ling `role` |
| `embeddings/<model>/<size>.npy` | | | float16 unit vectors, row i = chunk i of that size |
| `qa/wiki_qa_rag.parquet` | `Rag` | 531k | a question about one page: easy (names the place) or hard (describes it) |
| `qa/wiki_qa_unanswerable.parquet` | `Unanswerable` | 31k | a question no page answers |
| `qa/wiki_qa_compare.parquet` | `Compare` | 10k | a question that needs two pages |
| `qa/wiki_qa_meta.parquet` | `Meta` | 4k | a question with a set of pages as its answer (list, geo) |
| `qa/wiki_qa_tables.parquet` | `Tables` | 339 | a question that aggregates a table in a page |

Each model lists its file's columns; `wiki_qa.py` loads them. Questions point at pages by
`id` (`svwiki/Q123`). Test on `ok` rows (`answer_ok` in `Rag`), tune on `split == "dev"`.

```python
from src.db.wiki_qa import Chunk, Rag, Unanswerable, embeddings, load, pages, read

hard = load(Rag, ["id", "question"], answer_ok=True, kind="challenge").to_pandas()
for q in read(Unanswerable, ok=True):  # validated models
    print(q.question, q.why)
texts = pages(hard.id)  # page id -> page
# embed chunk.embedded(): title, breadcrumb, text
chunks = load(Chunk, size=512).to_pandas()
ids, vectors = embeddings("qwen3-embedding-0.6b", 512)  # same rows as `chunks`
```

Embedding models: `qwen3-embedding-0.6b` (1024 dims, local GPU) and `nemotron-3-embed-1b`
(2048 dims, free on OpenRouter; prefix queries with `query: `). Qwen3 queries take its
`query` prompt. Rows still NaN are not embedded yet.

How the files are made (each step resumable, `pages.jsonl` is only read):

| step | command | writes |
|---|---|---|
| chunk | `just wiki-chunks` (`wiki_chunks.py`) | `pages.parquet`, `chunks.parquet`; rerun to copy in new notes |
| notes | `just wiki-notes summaries`, `just wiki-notes roles --size 512` (`wiki_notes.py`) | `notes.db` |
| embed | `just wiki-embed qwen3-embedding-0.6b` (`wiki_embed.py`) | `embeddings/` |

Chunking follows aihub-core: split at the page's `##`/`###` headings, pack neighbouring
sections up to the size, split only a section too big alone; no overlap.

Not confidential, copy it where you need it; do not publish it.
How the questions were made: `docs/reports/wiki-qgen/wiki-qa-dataset.md`.

## Legacy: Slovenian page database

Scraped pages, page chunks, and eval labels of the first Slovenian corpus live in
`$LLMS4EU_DATA/db/pages.db`. `legacy/pages.py` owns the connection and applies `sql/raw_pages.sql` and
`sql/eval.sql`; chunking, indexing and eval still read through it.

Vector indexing lives in `src/indexing`.

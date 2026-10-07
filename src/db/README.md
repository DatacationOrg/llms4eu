# DB

The project's data and how to read it.

## Wiki places dataset (the main one)

133k Wikipedia pages of EU places (24 languages), cut into chunks, embedded, and questions
about them. On thebeast under `/data/llms4eu/wiki/` (`DATASET_DIR` for a copy):

| path | model (`schemas/<name>.py`) | rows | what |
|---|---|---|---|
| `wikipages.parquet` | `Page` | 133k | the pages: `id`, `title`, `text` (Markdown), metadata, Ling `summary` |
| `chunks.parquet` | `Chunk` | 1.7M | the pages cut at every size in `SIZES` (`size`), with `breadcrumb` and Ling `role` |
| `embeddings/<provider>/<size>.npy` | | | float16 unit vectors, row i = chunk i of that size |
| `qa/wiki_qa_rag.parquet` | `Rag` | 531k | a question about one page: easy (names the place) or hard (describes it) |
| `qa/wiki_qa_unanswerable.parquet` | `Unanswerable` | 31k | a question no page answers |
| `qa/wiki_qa_compare.parquet` | `Compare` | 10k | a question that needs two pages |
| `qa/wiki_qa_meta.parquet` | `Meta` | 4k | a question with a set of pages as its answer (list, geo) |
| `qa/wiki_qa_tables.parquet` | `Tables` | 339 | a question that aggregates a table in a page |

Each model lists its file's columns; `dataset.py` loads them. Questions point at pages by
`id` (`svwiki/Q123`). Test on `ok` rows (`answer_ok` in `Rag`), tune on `split == "dev"`.

```python
from src.db.dataset import Chunk, Rag, Unanswerable, chunk_size, embeddings
from src.db.dataset import load, pages, read

hard = load(Rag, ["id", "question"], answer_ok=True, kind="challenge").to_pandas()
for q in read(Unanswerable, ok=True):  # validated models
    print(q.question, q.why)
texts = pages(hard.id)  # page id -> page
# embedded text: represent(title, breadcrumb, text) in schemas/chunk.py
chunks = load(Chunk, size=chunk_size()).to_pandas()
ids, vectors = embeddings("qwen", chunk_size())  # same rows as `chunks`
```

Chunk sizes are `SIZES = (256, 512, 1024, 2048)` tokens in `dataset.py`; the
pipeline works on one, `CHUNK_SIZE` (default `DEFAULT_SIZE`, 512).

Embeddings are named after the indexing provider: `qwen` (Qwen3-Embedding-0.6B, 1024
dims) and `nemotron` (Nemotron 3 Embed 1B, 2048 dims), more as they are run. Queries
take the model's `query` prompt. Rows still NaN are not embedded yet (`missing`).

How the files are made (`pages.jsonl` and `qa/` are only read):

| step | command | writes |
|---|---|---|
| pages | `just pages` (`src/preprocess/pages.py`) | `wikipages.parquet` |
| chunk | `just chunk` (`src/preprocess/chunker.py`) | `chunks.parquet` |
| embed | `just index qwen`, `just index nemotron --api` (`src/indexing`) | `embeddings/<provider>/<size>.npy` |
| notes | `datagen/notes.py`, not part of the pipeline | `summary`, `role` |

Not confidential, copy it where you need it; do not publish it.
How the questions were made: `docs/reports/wiki-qgen/wiki-qa-dataset.md`.

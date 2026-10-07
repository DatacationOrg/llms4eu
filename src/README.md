# Source

Service-shaped folders with one root `pyproject.toml`.

`scraping` fetches source URLs, converts pages to Markdown, and appends them to a
JSONL of pages.

`db` owns the data and how to read it: `dataset.py` and one pydantic model per
Parquet file in `schemas/` (pages, chunks, questions), plus the stored embeddings.

`preprocess` cuts the pages into chunks (`chunks.parquet`).

`indexing` owns the embedding providers, embeds the chunks into
`embeddings/<provider>/<size>.npy`, and answers vector queries over them.

`retrieval` owns reusable chunk retrievers and the public retrieval catalog.

`eval` owns relevance from the questions' evidence, metrics, timing, and reports. It
asks `retrieval` for named retrievers to compare.

`shared` is not a service. It holds code used by more than one script, including
`env.py`, which resolves every generated artifact under `LLMS4EU_DATA`
(`data_path(...)`), so a location is stated once. The dataset folder is
`DATASET_DIR` (`src/db/dataset.py`).

Generated columns (summaries, roles) were made by `datagen/`, outside `src/`, which
the pipeline never imports.

Orchestration folders can import infrastructure folders (`db`, `indexing`)
and shared helpers. Avoid cross-imports between orchestration folders except
when a later pipeline stage consumes an earlier artifact.

No `__init__.py` files are needed; namespace packages are enough here.

Benchmark orchestration belongs in `eval` or a dedicated experiment, not in a
retriever implementation.

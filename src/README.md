# Source

Service-shaped folders with one root `pyproject.toml`.

`scraping` fetches source URLs, converts pages to Markdown, and stores them in
SQLite.

`db` owns the data: the wiki places dataset (`wiki_qa.py`, `schemas/`) and the
legacy page-database connection (`legacy/`).

`preprocess` rebuilds derived artifacts from SQL rows, currently page chunks.

`indexing` orchestrates chunk vector index rebuilds.

`indexing` owns embedding providers and the Chroma collections they fill.

`retrieval` owns reusable chunk retrievers and the public retrieval catalog.

`eval` owns labels, metrics, timing, and reports. It asks `retrieval` for named
retrievers to compare.

`shared` is not a service. It holds code used by more than one script, including
`env.py`, which resolves every generated artifact under `LLMS4EU_DATA`. Use
`pages_db()`, `chroma_path()` or `data_path(...)` instead of spelling a path in
a config file, so the location is stated once.

Orchestration folders can import infrastructure folders (`db`, `indexing`)
and shared helpers. Avoid cross-imports between orchestration folders except
when a later pipeline stage consumes an earlier artifact.

No `__init__.py` files are needed; namespace packages are enough here.

Benchmark orchestration belongs in `eval` or a dedicated experiment, not in a
retriever implementation.

# Source

Service-shaped folders with one root `pyproject.toml`.

`db` owns SQLite initialization and place row queries.

`preprocess` reads SQL rows and rebuilds derived artifacts such as Markdown page
chunks.

`indexing` orchestrates chunk vector index rebuilds.

`vector_store` owns Chroma mechanics for place and chunk vectors.

`retrieval` owns reusable chunk retrievers and the public retrieval catalog.

`scraping` owns website crawling, scrape-to-place transformation, incremental
SQLite ingest, and the small local scrape UI.

`rag` owns place search and answer generation.

`eval` owns labels, metrics, timing, and reports. It asks `retrieval` for named
retrievers to compare.

`shared` is not a service. It holds code used by more than one script.

Orchestration folders can import infrastructure folders (`db`, `vector_store`)
and shared helpers. Avoid cross-imports between orchestration folders except
when a later pipeline stage consumes an earlier artifact.

No `__init__.py` files are needed; namespace packages are enough here.

Benchmark orchestration belongs in `eval` or a dedicated experiment, not in a
retriever implementation.

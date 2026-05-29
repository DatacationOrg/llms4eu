# Source

Service-shaped folders with one root `pyproject.toml`.

`db` owns SQLite initialization and place row queries.

`preprocess` reads SQL rows and rebuilds derived artifacts. Today that is only
place indexing plus Markdown page chunking.

`indexing` owns provider-shaped embedding indexers and Chroma collections for
chunk retrieval experiments.

`vector_db` owns Chroma operations for place vectors.

`scraping` owns website crawling, scrape-to-place transformation, incremental
SQLite ingest, and the small local scrape UI.

`rag` queries Chroma, fetches rows from SQLite, and asks the local model to
answer from that context.

`shared` is not a service. It holds code used by more than one script.

Orchestration folders (`preprocess`, `indexing`, `rag`, `eval`) can import
infrastructure folders (`db`, `vector_db`) and shared helpers. They should not
import each other except when a later pipeline stage consumes the artifact of an
earlier one.

No `__init__.py` files are needed; namespace packages are enough here.

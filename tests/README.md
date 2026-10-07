# Tests

The tests stay narrow on purpose: they cover what could break silently, and
nothing that only restates a one-line wrapper.

The data contract, read only on the real files: `test_dataset.py` checks every
Parquet file is in place with its model's columns and valid rows, chunk and
question ids point at pages that exist, chunk sizes, evidence spans, summary and
role lengths, and one unit vector per chunk in every embeddings file. Every other
test runs on a scratch `DATASET_DIR` set in `conftest.py`, so no test can write to
the real data.

`test_prompts.py` checks every file in `prompts/` renders and that a missing
variable raises instead of shipping a literal `$name`.

Silent-corruption risks: `test_embedding_cache.py` asks for the same texts in
reverse so a cache-merge ordering bug cannot hide; `test_page_extract.py`
covers the branch order that decides `page_kind` and the language a page
declares.

Retrieval stays local and model-free: `test_rag_methods.py` checks the method
catalog and readiness wiring, `test_ranking_fusion.py` weighted score fusion,
`test_rag_rerank.py` the rerank batch path, `test_indexing_store.py` vector
and BM25 search over a two-chunk dataset, `test_chunker.py` heading-aware
splitting and which vectors survive a rechunk, `test_structured_markdown.py` listing classification.

`test_eval_metrics.py` covers metric math only; it runs no inference.

Performance is not a unit test. Timings come from `just eval-all`, which
records ms/query per method in its report.

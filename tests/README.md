# Tests

The tests stay narrow on purpose: they cover what could break silently, and
nothing that only restates a one-line wrapper.

Contracts that span two files, where drift is invisible until runtime:
`test_page_schema.py` checks `PageMetadata`'s fields against the columns in
`sql/raw_pages.sql`; `test_prompts.py` checks every file in `prompts/` renders
and that a missing variable raises instead of shipping a literal `$name`.

Silent-corruption risks: `test_embedding_cache.py` asks for the same texts in
reverse so a cache-merge ordering bug cannot hide; `test_page_extract.py`
covers the branch order that decides `page_kind` and the language a page
declares.

Retrieval stays local and model-free: `test_rag_methods.py` checks the method
catalog and readiness wiring, `test_ranking_fusion.py` weighted score fusion,
`test_rag_rerank.py` the rerank batch path, `test_indexing_store.py` chunk
vector rebuild and query with a stub indexer, `test_chunker.py` heading-aware
splitting, `test_structured_markdown.py` listing classification.

`test_eval_metrics.py` covers metric math only; it runs no inference.

Performance is not a unit test. Timings come from `just eval-all`, which
records ms/query per method in its report.

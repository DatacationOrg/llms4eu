# Tests

The tests stay narrow on purpose.

`test_seed_data.py` checks that the tracked fixture and SQL still match the
Pydantic place model. `test_rag_search.py` keeps local retrieval deterministic
without model downloads or services. `test_scraping_transform.py` keeps the
scrape-to-place conversion deterministic without live crawling or LLM calls.
`test_structured_markdown.py` covers only tiny HTML snippets for listing
classification.

Retrieval architecture tests stay local: `test_rag_methods.py` checks public
method names and readiness wiring, `test_ranking_fusion.py` checks weighted
score fusion, and `test_vector_store_chunks.py` checks chunk vector rebuild/query
behavior with a stub indexer.

Eval tests cover metric math only; they do not run model inference.

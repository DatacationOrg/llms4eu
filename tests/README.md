# Tests

The tests stay narrow on purpose.

`test_seed_data.py` checks that the tracked fixture and SQL still match the
Pydantic place model. `test_rag_search.py` keeps local retrieval deterministic
without model downloads or services. `test_scraping_transform.py` keeps the
scrape-to-place conversion deterministic without live crawling or LLM calls.

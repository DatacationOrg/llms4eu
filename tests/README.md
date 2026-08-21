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

Performance measurements are not unit tests. The comparative protocol requires
controlled repeated runs and raw observations described in
[`experiments/indexing/README.md`](../experiments/indexing/README.md).

## Scraper-arena tests

`test_scraper_arena_*.py` cover the research code under
[`research/scrapers/`](../research/scrapers/README.md) — the blind A/B extractor
benchmark — not the production pipeline. They live here because `testpaths` in
`pyproject.toml` collects only this directory, and `research.scrapers` imports from
the repo root via `pythonpath = ["."]`.

| file | covers |
| --- | --- |
| `test_scraper_arena_marks.py` | whole-document diff marking (`arena/marks.py`) |
| `test_scraper_arena_judge.py` | brief clipping, matchup sampling, report helpers |
| `test_scraper_arena_inventory.py` | entrant registry and focus-set consistency |
| `test_scraper_arena_rating.py` | Elo / Bradley-Terry and rank correlation |
| `test_scraper_arena_store.py` | vote storage, judge pools, standings |
| `test_scraper_arena_vote_api.py` | the vote endpoints, including cross-site refusal |

`test_scraper_arena_inventory.py` is deliberately not a unit test: it asserts on
source text, vote counts and focus-set membership, so it is a ledger of research
decisions and *will* fail for anyone who changes the arena's configuration. That is
the intent — the numbers in the report and the deck depend on those choices.

None of them need the network. Four are pure functions over literals; the two that
use the database (`store`, `vote_api`) monkeypatch `store.ROOT` to `tmp_path` and
replace `store.config()`, so the real `.local/scraper-arena` store is never opened —
without both redirections they would mutate a live judging session.

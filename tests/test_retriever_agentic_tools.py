import sqlite3

import pytest

from src.retrieval.base import RankedChunk
from src.retrieval.retrievers import agentic_tools, page_tools
from src.retrieval.retrievers.agentic_tools import (
    AgenticToolRetriever,
    ToolAction,
    _merge,
)


@pytest.fixture
def pages_db(monkeypatch, tmp_path):
    """Two pages of chunks, only some of which the base retriever returns."""
    path = tmp_path / "pages.db"
    conn = sqlite3.connect(path)
    conn.executescript(
        """
        create table page_metadata (
          id text primary key, source text, url text, title text
        );
        create table page_chunks (
          id text primary key, page_id text, chunk_index integer,
          heading_path text, text text
        );
        insert into page_metadata values
          ('p1', 'castle', 'https://x.test/castle', 'Rajhenburg Castle'),
          ('p2', 'other', 'https://x.test/other', 'Other Page');
        insert into page_chunks values
          ('p1-0', 'p1', 0, 'Intro', 'The castle stands above Brestanica.'),
          ('p1-1', 'p1', 1, 'Opening hours', 'Open Tuesday to Sunday 10:00-18:00.'),
          ('p1-2', 'p1', 2, 'Opening hours', 'Closed on Mondays.'),
          ('p2-0', 'p2', 0, 'Intro', 'Unrelated content.');
        """
    )
    conn.commit()
    conn.close()

    def _connect():
        connection = sqlite3.connect(path)
        connection.row_factory = sqlite3.Row
        return connection

    monkeypatch.setattr(page_tools, "connect", _connect)
    return path


class StubRetriever:
    name = "stub"

    def __init__(self, responses):
        self.responses = responses
        self.calls = []

    def retrieve(self, query: str, limit: int):
        self.calls.append((query, limit))
        return self.responses[min(len(self.calls) - 1, len(self.responses) - 1)]

    def retrieve_batch(self, queries: list[str], limit: int):
        return {idx: self.retrieve(query, limit) for idx, query in enumerate(queries)}


class ScriptedJudge:
    """Judge returning a fixed sequence of tool actions."""

    def __init__(self, actions):
        self.actions = list(actions)
        self.prompts = []

    def structured_output(self, prompt, output_schema, *, retries=3):
        self.prompts.append(prompt)
        return self.actions.pop(0)


def test_list_sections_returns_table_of_contents(pages_db):
    sections = page_tools.list_sections("p1")

    assert [(s.heading_path, s.chunk_count) for s in sections] == [
        ("Intro", 1),
        ("Opening hours", 2),
    ]


def test_search_in_page_returns_scoreable_chunk_ids(pages_db):
    hits = page_tools.search_in_page("p1", "Mondays")

    assert [hit.id for hit in hits] == ["p1-2"]
    assert hits[0].heading_path == "Opening hours"


def test_search_in_page_is_scoped_to_one_page(pages_db):
    assert page_tools.search_in_page("p2", "castle") == []
    assert page_tools.search_in_page("p1", "") == []


def test_search_in_page_treats_wildcards_as_literal_text(pages_db):
    """A bare % must not match every chunk on the page."""
    assert page_tools.search_in_page("p1", "%") == []


def test_agent_promotes_sibling_chunks_it_finds(pages_db):
    base = StubRetriever([[RankedChunk(id="p1-0", score=0.9, text="intro")]])
    judge = ScriptedJudge(
        [
            ToolAction(
                action="search_in_page",
                page_id="p1",
                term="Mondays",
                reason="wrong section",
            ),
            ToolAction(action="sufficient", reason="found it"),
        ]
    )
    retriever = AgenticToolRetriever(
        name="qwen_hybrid_agentic_tools",
        base_retriever=base,
        max_attempts=4,
        min_sufficient_chunks=1,
        judge=judge,
    )

    chunks = retriever.retrieve("opening hours", 10)

    # The found sibling sits directly after its anchor, so it can score at hit@2.
    assert [chunk.id for chunk in chunks] == ["p1-0", "p1-2"]
    assert len(base.calls) == 1  # a tool step reuses the current ranking


def test_agent_records_tool_actions_in_the_action_log(pages_db):
    base = StubRetriever([[RankedChunk(id="p1-0", score=0.9, text="intro")]])
    judge = ScriptedJudge(
        [
            ToolAction(action="list_sections", page_id="p1", reason="check structure"),
            ToolAction(action="sufficient", reason="done"),
        ]
    )
    retriever = AgenticToolRetriever(
        name="qwen_hybrid_agentic_tools",
        base_retriever=base,
        max_attempts=4,
        min_sufficient_chunks=1,
        judge=judge,
    )

    retriever.retrieve("query", 10)

    log = retriever.batch_stats.action_log
    assert [entry["verdict"]["action"] for entry in log] == [
        "list_sections",
        "sufficient",
    ]
    # The sufficiency shape stays intact for the existing agentic diagnostics.
    assert log[0]["verdict"]["sufficient"] is False
    assert log[1]["verdict"]["sufficient"] is True
    assert "list_sections(p1)" in judge.prompts[1]


def test_tool_budget_stops_runaway_tool_use(pages_db):
    """Distinct calls, so the budget is what stops them rather than the dedup."""
    base = StubRetriever([[RankedChunk(id="p1-0", score=0.9, text="intro")]])
    judge = ScriptedJudge(
        [
            ToolAction(action="search_in_page", page_id="p1", term=term)
            for term in ("castle", "Open", "Closed", "Brestanica", "Sunday", "Intro")
        ]
    )
    retriever = AgenticToolRetriever(
        name="qwen_hybrid_agentic_tools",
        base_retriever=base,
        max_attempts=6,
        min_sufficient_chunks=1,
        max_tool_calls=2,
        judge=judge,
    )

    retriever.retrieve("query", 10)

    asked = [
        entry
        for entry in retriever.batch_stats.action_log
        if entry["verdict"]["action"] == "search_in_page"
    ]
    assert len(asked) == 6  # the model may keep asking...
    assert judge.prompts[-1].count("search_in_page(p1,") == 2  # ...but only 2 ran
    assert "Tool budget exhausted" in judge.prompts[-1]


def test_repeated_identical_search_is_not_rerun(pages_db):
    """A repeat of the same call must not burn tool budget on a known answer."""
    base = StubRetriever([[RankedChunk(id="p1-0", score=0.9, text="intro")]])
    repeat = ToolAction(
        action="search_in_page", page_id="p1", term="Mondays", reason="again"
    )
    judge = ScriptedJudge([repeat, repeat, repeat, repeat])
    retriever = AgenticToolRetriever(
        name="qwen_hybrid_agentic_tools",
        base_retriever=base,
        max_attempts=4,
        min_sufficient_chunks=1,
        max_tool_calls=3,
        judge=judge,
    )

    chunks = retriever.retrieve("query", 10)

    # Ran once; the later identical calls were answered without touching SQL.
    assert judge.prompts[-1].count('search_in_page(p1, "Mondays"):') == 1
    assert "You already ran search_in_page" in judge.prompts[-1]
    assert [chunk.id for chunk in chunks] == ["p1-0", "p1-2"]


def test_a_different_term_on_the_same_page_still_runs(pages_db):
    base = StubRetriever([[RankedChunk(id="p1-0", score=0.9, text="intro")]])
    judge = ScriptedJudge(
        [
            ToolAction(action="search_in_page", page_id="p1", term="Mondays"),
            ToolAction(action="search_in_page", page_id="p1", term="Tuesday"),
            ToolAction(action="sufficient", reason="done"),
        ]
    )
    retriever = AgenticToolRetriever(
        name="qwen_hybrid_agentic_tools",
        base_retriever=base,
        max_attempts=4,
        min_sufficient_chunks=1,
        judge=judge,
    )

    chunks = retriever.retrieve("query", 10)

    assert {chunk.id for chunk in chunks} == {"p1-0", "p1-1", "p1-2"}


def test_judge_failure_falls_back_to_expanding(pages_db, capsys):
    class FailingJudge:
        def structured_output(self, prompt, output_schema, *, retries=3):
            raise RuntimeError("structured call failed after 3 attempts")

    base = StubRetriever([[RankedChunk(id="p1-0", score=0.9, text="intro")]])
    retriever = AgenticToolRetriever(
        name="qwen_hybrid_agentic_tools",
        base_retriever=base,
        max_attempts=2,
        min_sufficient_chunks=1,
        judge=FailingJudge(),
    )

    chunks = retriever.retrieve("query", 10)

    assert [chunk.id for chunk in chunks] == ["p1-0"]
    assert "agentic tool judge failed" in capsys.readouterr().out


def test_merge_keeps_base_order_when_nothing_was_found(pages_db):
    chunks = [
        RankedChunk(id="p1-0", score=0.9, text="a"),
        RankedChunk(id="p2-0", score=0.5, text="b"),
    ]

    assert _merge(chunks, []) == chunks


def test_merge_never_duplicates_a_chunk(pages_db):
    chunks = [RankedChunk(id="p1-0", score=0.9, text="a")]
    promoted = [("p1", RankedChunk(id="p1-0", score=0.0, text="a"))]

    assert [chunk.id for chunk in _merge(chunks, promoted)] == ["p1-0"]


def test_tool_agent_pairs_with_the_plain_reranked_baseline():
    from src.eval.agentic_diagnostics import baseline_for_agent

    assert baseline_for_agent("qwen_hybrid_agentic_tools") == "qwen_hybrid_rerank"
    assert (
        baseline_for_agent("nemotron_hybrid_agentic_tools_v2")
        == "nemotron_hybrid_rerank_v2"
    )


def test_tool_actions_cover_the_documented_action_set():
    assert set(agentic_tools.TOOL_ACTIONS) == {
        "sufficient",
        "reformulate",
        "expand",
        "list_sections",
        "search_in_page",
        "find_pages_near",
        "pages_in_region",
    }


def test_find_pages_near_promotes_located_pages(pages_db, monkeypatch):
    from src.retrieval.retrievers.page_tools import LocatedPage
    from src.shared.geo_scope import GeoScope

    class StubGazetteer:
        default_radius_km = 25.0

        def lookup(self, place, granularity="point", radius_km=None):
            assert place == "Brestanica"
            return GeoScope(latitude=45.99, longitude=15.47, radius_km=radius_km)

    def fake_near(coordinates, radius_km, limit, variant):
        assert radius_km == 10.0 and variant == "tok512"
        return [
            LocatedPage(
                "p1",
                "Rajhenburg Castle",
                "Grad Rajhenburg",
                "SI036",
                1.2,
                "p1-0",
                "The castle stands above Brestanica.",
            )
        ]

    monkeypatch.setattr(agentic_tools, "located_pages_near", fake_near)
    base = StubRetriever([[RankedChunk(id="p2-0", score=0.9, text="unrelated")]])
    judge = ScriptedJudge(
        [
            ToolAction(action="find_pages_near", place="Brestanica", radius_km=10),
            ToolAction(action="sufficient", reason="found it"),
        ]
    )
    retriever = AgenticToolRetriever(
        name="qwen_hybrid_agentic_tools",
        base_retriever=base,
        max_attempts=4,
        min_sufficient_chunks=1,
        judge=judge,
        gazetteer=StubGazetteer(),
        variant="tok512",
    )

    chunks = retriever.retrieve("gradovi blizu Brestanice", 10)

    assert [chunk.id for chunk in chunks] == ["p2-0", "p1-0"]
    assert 'find_pages_near("Brestanica", 10 km)' in judge.prompts[1]
    assert "1.2 km away" in judge.prompts[1]


def test_geo_tools_without_a_gazetteer_explain_themselves(pages_db):
    base = StubRetriever([[RankedChunk(id="p2-0", score=0.9, text="unrelated")]])
    judge = ScriptedJudge(
        [
            ToolAction(action="find_pages_near", place="Brestanica"),
            ToolAction(action="sufficient", reason="give up"),
        ]
    )
    retriever = AgenticToolRetriever(
        name="qwen_hybrid_agentic_tools",
        base_retriever=base,
        max_attempts=4,
        min_sufficient_chunks=1,
        judge=judge,
    )

    retriever.retrieve("query", 10)

    assert "not available" in judge.prompts[1]


def test_judge_and_geo_suffixes_pair_with_the_right_baseline():
    from src.eval.agentic_diagnostics import baseline_for_agent, strip_reasoning_suffix

    assert strip_reasoning_suffix("qwen_hybrid_agentic_tools_gptoss") == (
        "qwen_hybrid_agentic_tools",
        "gptoss",
    )
    assert baseline_for_agent("qwen_hybrid_agentic_gemma_v2") == "qwen_hybrid_rerank_v2"
    assert baseline_for_agent("qwen_hybrid_agentic_geo") == "qwen_hybrid_rerank_geo"
    assert (
        baseline_for_agent("qwen_hybrid_agentic_tools_geo") == "qwen_hybrid_rerank_geo"
    )


def test_wider_radius_is_a_new_call_not_a_repeat(monkeypatch):
    from src.shared.geo_scope import GeoScope

    class StubGazetteer:
        default_radius_km = 25.0

        def lookup(self, place, granularity="point", radius_km=None):
            return GeoScope(latitude=45.99, longitude=15.47, radius_km=radius_km)

    radii = []

    def fake_near(coordinates, radius_km, limit, variant):
        radii.append(radius_km)
        return []

    monkeypatch.setattr(agentic_tools, "located_pages_near", fake_near)
    base = StubRetriever([[RankedChunk(id="p2-0", score=0.9, text="unrelated")]])
    judge = ScriptedJudge(
        [
            ToolAction(action="find_pages_near", place="Brestanica", radius_km=5),
            ToolAction(action="find_pages_near", place="Brestanica", radius_km=30),
            ToolAction(action="find_pages_near", place="Brestanica", radius_km=30),
            ToolAction(action="sufficient"),
        ]
    )
    retriever = AgenticToolRetriever(
        name="t",
        base_retriever=base,
        max_attempts=5,
        min_sufficient_chunks=1,
        judge=judge,
        gazetteer=StubGazetteer(),
    )

    retriever.retrieve("q", 10)

    assert radii == [5.0, 30.0]
    assert "You already ran find_pages_near" in judge.prompts[3]


def test_tool_agent_without_gazetteer_neither_offers_nor_runs_geo_tools():
    base = StubRetriever([[RankedChunk(id="p2-0", score=0.9, text="unrelated")]])
    judge = ScriptedJudge(
        [
            ToolAction(action="pages_in_region", region_code="SI036"),
            ToolAction(action="sufficient"),
        ]
    )
    retriever = AgenticToolRetriever(
        name="t",
        base_retriever=base,
        max_attempts=3,
        min_sufficient_chunks=1,
        judge=judge,
    )

    retriever.retrieve("q", 10)

    assert "find_pages_near" not in judge.prompts[0]
    assert "pages_in_region" not in judge.prompts[0]
    assert "not available in this configuration" in judge.prompts[1]


def test_unsupported_region_code_shape_is_named_not_reported_as_empty(monkeypatch):
    class StubGazetteer:
        default_radius_km = 25.0

    calls = []
    monkeypatch.setattr(
        agentic_tools,
        "located_pages_in_region",
        lambda code, limit, variant: calls.append(code) or [],
    )
    base = StubRetriever([[RankedChunk(id="p2-0", score=0.9, text="unrelated")]])
    judge = ScriptedJudge(
        [
            ToolAction(action="pages_in_region", region_code="SI0"),
            ToolAction(action="pages_in_region", region_code="Posavje"),
            ToolAction(action="sufficient"),
        ]
    )
    retriever = AgenticToolRetriever(
        name="t",
        base_retriever=base,
        max_attempts=4,
        min_sufficient_chunks=1,
        judge=judge,
        gazetteer=StubGazetteer(),
    )

    retriever.retrieve("q", 10)

    assert calls == ["SI0"]  # NUTS-1 is a supported shape
    assert "give a NUTS code" in judge.prompts[2]

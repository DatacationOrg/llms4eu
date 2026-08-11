import sqlite3

import pytest

from src.retrieval import workspace as workspace_module
from src.retrieval.base import RankedChunk
from src.retrieval.retrievers import dci
from src.retrieval.retrievers.dci import (
    CorpusAction,
    DirectCorpusRetriever,
    _ranking,
    _shortlist_documents,
)
from src.retrieval.workspace import build_workspace


@pytest.fixture
def corpus(monkeypatch, tmp_path):
    """Two pages whose chunks are distinguishable by exact wording."""
    db = tmp_path / "pages.db"
    conn = sqlite3.connect(db)
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
          ('p2', 'town', 'https://x.test/town', 'Brestanica');
        insert into page_chunks values
          ('p1-0', 'p1', 0, 'Intro', 'The castle stands above Brestanica.'),
          ('p1-1', 'p1', 1, 'Hours', 'Open Tuesday to Sunday.'),
          ('p1-2', 'p1', 2, 'Hours', 'Closed on Mondays in winter.'),
          ('p2-0', 'p2', 0, 'Intro', 'Brestanica is a settlement.');
        """
    )
    conn.commit()
    conn.close()

    def _connect():
        connection = sqlite3.connect(db)
        connection.row_factory = sqlite3.Row
        return connection

    monkeypatch.setattr(workspace_module, "connect", _connect)
    return build_workspace(tmp_path / "ws")


class StubShortlist:
    name = "sparse"

    def __init__(self, chunks):
        self.chunks = chunks
        self.calls = []

    def retrieve(self, query, limit):
        self.calls.append((query, limit))
        return self.chunks[:limit]

    def retrieve_batch(self, queries, limit):
        return {i: self.retrieve(q, limit) for i, q in enumerate(queries)}


class ScriptedAgent:
    def __init__(self, actions):
        self.actions = list(actions)
        self.prompts = []

    def structured_output(self, prompt, output_schema, *, retries=3):
        self.prompts.append(prompt)
        return self.actions.pop(0)


def _chunks(*ids):
    return [RankedChunk(id=i, score=1.0, text=i) for i in ids]


def test_workspace_maps_every_line_back_to_one_chunk(corpus):
    document = corpus.by_page_id["p1"]

    assert [span.chunk_id for span in document.spans] == ["p1-0", "p1-1", "p1-2"]
    for span in document.spans:
        assert corpus.chunk_for(document.path, span.start_line) == span.chunk_id
    assert "lines" in document.toc()


def test_workspace_rebuild_only_rewrites_changed_pages(corpus, capsys, monkeypatch):
    capsys.readouterr()
    build_workspace(corpus.root)

    # Nothing changed, so no page is rewritten and no build line is printed.
    assert "wrote" not in capsys.readouterr().out


def test_workspace_drops_pages_that_disappear(corpus, monkeypatch, tmp_path):
    rows = [
        {
            "id": "p1-0",
            "page_id": "p1",
            "chunk_index": 0,
            "heading_path": "Intro",
            "text": "kept",
            "title": "t",
            "source": "castle",
            "url": "u",
        }
    ]
    monkeypatch.setattr(
        workspace_module,
        "_pages",
        lambda: iter([{"page_id": "p1", "chunks": rows}]),
    )
    rebuilt = build_workspace(corpus.root)

    assert [d.page_id for d in rebuilt.documents] == ["p1"]
    assert not (corpus.root / "town/p2.md").exists()


def test_shortlist_is_capped_at_k(corpus):
    shortlist = _chunks("p1-0", "p2-0")

    assert len(_shortlist_documents(corpus, shortlist, 2)) == 2
    assert [d.page_id for d in _shortlist_documents(corpus, shortlist, 1)] == ["p1"]


def test_agent_search_finds_the_chunk_id_behind_a_line(corpus):
    shortlist = StubShortlist(_chunks("p1-0", "p2-0"))
    agent = ScriptedAgent(
        [
            CorpusAction(action="search", pattern="Mondays"),
            CorpusAction(action="answer", chunk_ids=["p1-2"]),
        ]
    )
    retriever = DirectCorpusRetriever(
        name="dci",
        shortlist_retriever=shortlist,
        judge=agent,
        workspace=corpus,
    )

    ranked = retriever.retrieve("when is it closed", 10)

    assert ranked[0].id == "p1-2"
    assert "[chunk p1-2]" in agent.prompts[1]


def test_repeated_identical_search_is_not_rerun(corpus):
    shortlist = StubShortlist(_chunks("p1-0"))
    repeat = CorpusAction(action="search", pattern="Mondays")
    agent = ScriptedAgent([repeat, repeat, CorpusAction(action="answer")])
    retriever = DirectCorpusRetriever(
        name="dci",
        shortlist_retriever=shortlist,
        judge=agent,
        workspace=corpus,
        max_steps=3,
    )

    retriever.retrieve("query", 10)

    assert agent.prompts[-1].count('search("Mondays"): 1 matches') == 1
    assert "You already ran that exact search" in agent.prompts[-1]


def test_agent_cannot_read_outside_the_working_set(corpus):
    shortlist = StubShortlist(_chunks("p1-0"))
    agent = ScriptedAgent(
        [
            CorpusAction(action="read", path="../../etc/passwd", start_line=1),
            CorpusAction(action="read", path="town/p2.md", start_line=1),
            CorpusAction(action="answer", chunk_ids=["p1-0"]),
        ]
    )
    retriever = DirectCorpusRetriever(
        name="dci",
        shortlist_retriever=shortlist,
        judge=agent,
        workspace=corpus,
        max_documents=1,
        max_steps=3,
    )

    retriever.retrieve("query", 10)

    # Escaping the workspace and reading a page outside the shortlist both fail.
    assert "is not in the working set" in agent.prompts[1]
    assert "is not in the working set" in agent.prompts[2]


def test_hallucinated_chunk_ids_never_enter_the_ranking(corpus):
    ranked = _ranking(
        cited=["does-not-exist", "p1-2"],
        fallback=_chunks("p1-0"),
        texts={"p1-2": "Closed on Mondays in winter."},
        workspace=corpus,
        limit=10,
    )

    assert [chunk.id for chunk in ranked] == ["p1-2", "p1-0"]


def test_ranking_falls_back_to_bm25_when_the_agent_cites_nothing(corpus):
    ranked = _ranking(
        cited=[],
        fallback=_chunks("p1-0", "p2-0"),
        texts={},
        workspace=corpus,
        limit=10,
    )

    assert [chunk.id for chunk in ranked] == ["p1-0", "p2-0"]


def test_agent_failure_falls_back_to_the_shortlist(corpus, capsys):
    class FailingAgent:
        def structured_output(self, prompt, output_schema, *, retries=3):
            raise RuntimeError("structured call failed after 3 attempts")

    retriever = DirectCorpusRetriever(
        name="dci",
        shortlist_retriever=StubShortlist(_chunks("p1-0", "p2-0")),
        judge=FailingAgent(),
        workspace=corpus,
    )

    ranked = retriever.retrieve("query", 10)

    assert [chunk.id for chunk in ranked] == ["p1-0", "p2-0"]
    assert "corpus agent failed" in capsys.readouterr().out


def test_dci_actions_cover_the_documented_set():
    assert set(dci.DCI_ACTIONS) == {"search", "read", "toc", "answer"}

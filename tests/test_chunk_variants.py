import sqlite3

import pytest

import src.preprocess.chunker as chunker
from src.db.pages import initialize_page_artifacts_db
from src.eval.compare_chunkings import chars_at_k
from src.eval.evaluate import load_eval_rows
from src.eval.evidence import relabel
from src.indexing.store import _collection_name
from src.preprocess.chunker import rebuild_page_chunks, variant_settings
from src.retrieval.retrievers.sparse import SparseRetriever
from src.shared.env import ROOT

QUOTE = "The castle was first mentioned in 1256."


@pytest.fixture
def tiny_variant(monkeypatch):
    monkeypatch.setitem(
        chunker.CONFIG["chunk_variants"],
        "tiny",
        {"chunk_size": 60, "chunk_overlap": 0, "chunk_min_chars": 0},
    )
    return "tiny"


def test_variant_settings_override_the_base_values(tiny_variant):
    assert variant_settings("base") == {"size": 1800, "overlap": 0, "min_chars": 300}
    assert variant_settings("c900ov") == {"size": 900, "overlap": 200, "min_chars": 300}
    with pytest.raises(SystemExit, match="Unknown chunk variant"):
        variant_settings("nope")


def test_variant_chunks_live_beside_base_with_their_own_ids(page_db, tiny_variant):
    _seed(page_db)
    rebuild_page_chunks()
    rebuild_page_chunks(variant=tiny_variant)
    with sqlite3.connect(page_db) as conn:
        ids = {
            row[1]: [r for r in row[0].split("\n")]
            for row in conn.execute(
                "select group_concat(id, char(10)), variant from page_chunks "
                "group by variant"
            )
        }
    assert ids["base"] == ["page:0"]
    assert len(ids[tiny_variant]) > 1
    assert all(chunk_id.startswith("page:tiny:") for chunk_id in ids[tiny_variant])


def test_each_variant_carries_its_own_labels_and_questions(
    page_db, tiny_variant, monkeypatch
):
    _seed(page_db, with_evidence=True)
    rebuild_page_chunks()
    relabel()
    monkeypatch.setenv("CHUNK_VARIANT", tiny_variant)
    rebuild_page_chunks()
    relabel()
    with sqlite3.connect(page_db) as conn:
        labelled = conn.execute(
            "select c.variant, count(*) from eval_relevant_chunks r "
            "join page_chunks c on c.id = r.chunk_id group by c.variant"
        ).fetchall()
    assert dict(labelled) == {"base": 1, tiny_variant: 1}

    questions, relevance = load_eval_rows()
    assert [q["id"] for q in questions] == ["q"]
    assert all(row["chunk_id"].startswith("page:tiny:") for row in relevance)
    assert all(
        hit.id.startswith("page:tiny:")
        for hit in SparseRetriever().retrieve("castle", limit=5)
    )
    assert _collection_name("qwen") == "page_chunks_qwen_tiny_chunk"

    monkeypatch.setenv("CHUNK_VARIANT", "base")
    assert _collection_name("qwen") == "page_chunks_qwen_chunk"
    questions, relevance = load_eval_rows()
    assert [row["chunk_id"] for row in relevance] == ["page:0"]


def test_variant_eval_skips_questions_its_chunks_do_not_label(page_db, monkeypatch):
    _seed(page_db)
    rebuild_page_chunks()
    monkeypatch.setenv("CHUNK_VARIANT", "c900")
    assert load_eval_rows() == ([], [])


def test_chars_at_k_sums_the_top_k_results():
    rankings = {"q1": ["a", "b", "c"], "q2": ["c"]}
    counts = {"a": 100, "b": 50, "c": 10}
    assert chars_at_k(rankings, counts, k=2) == (150 + 10) / 2


def test_old_database_is_refused_until_the_manual_migration_ran(tmp_path, monkeypatch):
    monkeypatch.setenv("LLMS4EU_DATA", str(tmp_path))
    path = tmp_path / "db" / "pages.db"
    path.parent.mkdir(parents=True)
    old_schema = (
        (ROOT / "sql" / "eval.sql")
        .read_text()
        .replace("  variant text not null default 'base',\n", "")
        .replace(
            "unique(page_id, variant, chunk_index)", "unique(page_id, chunk_index)"
        )
        .replace("create index if not exists idx_page_chunks_variant\n", "")
        .replace("  on page_chunks(variant);\n", "")
    )
    with sqlite3.connect(path) as conn:
        conn.executescript((ROOT / "sql" / "raw_pages.sql").read_text())
        conn.executescript(old_schema)
        _seed_rows(conn, with_evidence=False)
        conn.execute(
            "insert into page_chunks (id, page_id, chunk_index, text, char_count) "
            "values ('page:0', 'page', 0, 'old', 3)"
        )
        conn.execute("insert into eval_relevant_chunks values ('q', 'page:0')")

    with pytest.raises(SystemExit, match="migrate_chunk_variants"):
        initialize_page_artifacts_db()

    with sqlite3.connect(path) as conn:
        conn.executescript((ROOT / "sql" / "migrate_chunk_variants.sql").read_text())
    initialize_page_artifacts_db()

    with sqlite3.connect(path) as conn:
        conn.execute("pragma foreign_keys = on")
        assert conn.execute("select variant from page_chunks").fetchall() == [("base",)]
        assert conn.execute("select chunk_id from eval_relevant_chunks").fetchall() == [
            ("page:0",)
        ]
        with pytest.raises(sqlite3.IntegrityError):
            conn.execute("insert into eval_relevant_chunks values ('q', 'missing')")
        migrated = conn.execute("pragma table_info(page_chunks)").fetchall()
    with sqlite3.connect(":memory:") as fresh:
        fresh.executescript((ROOT / "sql" / "raw_pages.sql").read_text())
        fresh.executescript((ROOT / "sql" / "eval.sql").read_text())
        assert fresh.execute("pragma table_info(page_chunks)").fetchall() == migrated


def _seed(path, with_evidence=False):
    with sqlite3.connect(path) as conn:
        _seed_rows(conn, with_evidence)


def _seed_rows(conn, with_evidence):
    markdown = f"# Castle\n\n{QUOTE} " + "Filler sentence here. " * 20
    conn.execute(
        "insert into page_metadata (id, source, url, fetched_at) "
        "values ('page', 'fixture', 'https://example.test', '2026-01-01')"
    )
    conn.execute("insert into page_markdown_content values ('page', ?)", (markdown,))
    conn.execute(
        "insert into eval_questions (id, question, answer, question_type, "
        "question_language) values ('q', 'When?', '1256', 'same_language', 'en')"
    )
    if with_evidence:
        conn.execute("insert into eval_evidence values ('q', 'page', ?)", (QUOTE,))

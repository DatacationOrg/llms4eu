"""Tests for vote storage and the leaderboard's pool handling.

These run against a throwaway database: `store.data_dir()` is derived from the
config, so the config cache and the config value are both redirected at a tmp
path. Without that these tests would mutate the real `.local/scraper-arena`
database and destroy a judging session.
"""

import pytest

from research.scrapers import stats, store


@pytest.fixture
def arena(tmp_path, monkeypatch):
    monkeypatch.setattr(store, "ROOT", tmp_path)
    config = dict(store.config())
    config["data_dir"] = "arena"
    monkeypatch.setattr(store, "config", lambda: config)
    store.initialize()
    with store.connect() as connection:
        for index, (bucket, is_wiki) in enumerate(
            [("en-wiki", 1), ("en-other", 0), ("eu-wiki", 1)]
        ):
            connection.execute(
                "insert into pages (id, url, bucket, lang, shape, is_wiki, title)"
                " values (?, ?, ?, 'en', 'prose', ?, 'T')",
                (f"p{index}", f"https://example.org/{index}", bucket, is_wiki),
            )
    return store


def test_invalidate_votes_removes_rows_and_journals_them(arena):
    keep = arena.append_vote("p0", "trafilatura", "goose3", "a", judge=arena.LLM)
    drop = arena.append_vote("p1", "trafilatura", "goose3", "b", judge=arena.LLM)

    removed = arena.invalidate_votes([drop], "output regenerated")

    assert removed == 1
    remaining = [vote["id"] for vote in arena.load_votes(judge=arena.LLM)]
    assert remaining == [keep]

    # The mirror stays append-only: the retraction is recorded, not erased, so the
    # decision to discard evidence is itself auditable.
    lines = arena.votes_jsonl().read_text(encoding="utf-8").splitlines()
    journal = [line for line in lines if "invalidated" in line]
    assert len(journal) == 1
    assert "output regenerated" in journal[0]
    assert f'"id": {drop}' in journal[0]


def test_invalidate_votes_is_a_no_op_for_an_empty_list(arena):
    arena.append_vote("p0", "trafilatura", "goose3", "a", judge=arena.LLM)
    before = arena.votes_jsonl().read_text(encoding="utf-8")

    assert arena.invalidate_votes([], "nothing to do") == 0
    assert arena.votes_jsonl().read_text(encoding="utf-8") == before


def test_pools_stay_separate(arena):
    arena.append_vote("p0", "trafilatura", "goose3", "a", judge=arena.HUMAN)
    arena.append_vote("p0", "trafilatura", "goose3", "b", judge=arena.LLM)
    arena.append_vote("p1", "trafilatura", "goose3", "b", judge=arena.LLM)

    assert arena.judge_pools() == {arena.HUMAN: 1, arena.LLM: 2}
    assert len(arena.load_votes(judge=arena.HUMAN)) == 1
    assert len(arena.load_votes(judge=arena.LLM)) == 2


def test_default_pool_prefers_human_but_never_shows_a_meaningless_board():
    """The bug this guards: a flat-1500 leaderboard reads as a measurement."""
    from research.scrapers.arena.app import default_pool

    assert default_pool({store.HUMAN: 40, store.LLM: 200}) == store.HUMAN
    assert default_pool({store.LLM: 200}) == store.LLM
    assert default_pool({store.HUMAN: 0, store.LLM: 200}) == store.LLM
    # A handful of human votes is almost all prior, so it must not displace a
    # populated automated pool the moment the reviewer casts their first vote.
    assert default_pool({store.HUMAN: 1, store.LLM: 200}) == store.LLM
    assert default_pool({store.HUMAN: store.MIN_POOL_VOTES, store.LLM: 200}) == (
        store.HUMAN
    )
    # Nothing anywhere: fall back to the primary pool rather than inventing one.
    assert default_pool({}) == store.HUMAN


def test_record_snapshot_updates_only_the_fields_passed(arena):
    """Re-hashing a saved body must not erase how it was fetched.

    The destructive version wrote NULL over every omitted column, so a re-hash pass
    silently discarded the path, status code and latency of the original fetch --
    and with them the ability to recompute anything from disk.
    """
    arena.record_snapshot(
        "p0",
        "raw",
        path="/tmp/raw.html",
        bytes=1234,
        sha256="old",
        status_code=200,
        fetch_ms=181.5,
    )
    arena.record_snapshot("p0", "raw", sha256="new")

    row = next(r for r in arena.load_snapshots("raw") if r["page_id"] == "p0")
    assert row["sha256"] == "new"
    assert row["path"] == "/tmp/raw.html"
    assert row["bytes"] == 1234
    assert row["status_code"] == 200
    assert row["fetch_ms"] == 181.5


def test_record_snapshot_keeps_fetched_at_when_only_rehashing(arena):
    """The timestamp records when bytes were retrieved, not when they were re-read."""
    arena.record_snapshot("p0", "raw", path="/tmp/raw.html", sha256="old")
    first = next(r for r in arena.load_snapshots("raw") if r["page_id"] == "p0")

    arena.record_snapshot("p0", "raw", sha256="new")
    after = next(r for r in arena.load_snapshots("raw") if r["page_id"] == "p0")
    assert after["fetched_at"] == first["fetched_at"]


def test_record_snapshot_rejects_unknown_fields(arena):
    """A typo'd field name silently did nothing before."""
    with pytest.raises(ValueError, match="unknown snapshot fields"):
        arena.record_snapshot("p0", "raw", shas256="typo")


# --------------------------------------------------- cell-level agreement


def test_cell_agreement_is_empty_without_overlap(arena):
    """The state the arena starts in: two pools, disjoint cells, no evidence.

    Reporting 0% or 100% here would both be wrong. The measure must say "not
    measurable" and count how many cells are available to re-check.
    """

    arena.append_vote("p0", "trafilatura", "goose3", "a", judge=arena.LLM)
    arena.append_vote("p1", "trafilatura", "goose3", "b", judge=arena.HUMAN)

    result = stats.judge_cell_agreement()
    assert result["cells"] == 0
    assert result["equivalent"] is None
    assert result["kappa"] is None
    assert result["candidates"] == 1


def test_cell_agreement_counts_a_shared_cell_regardless_of_side(arena):
    """Same page, same pair, same winner -- but shown in opposite orders.

    The winner is compared by entrant name, not by slot, so a flipped presentation
    still counts as agreement. Getting this wrong would make agreement track the
    randomised side assignment instead of the judgement.
    """

    arena.append_vote("p0", "trafilatura", "goose3", "a", judge=arena.LLM)
    arena.append_vote("p0", "goose3", "trafilatura", "b", judge=arena.HUMAN)

    result = stats.judge_cell_agreement()
    assert result["cells"] == 1
    assert result["equivalent"] == 100.0
    assert result["decisive"] == 100.0
    # Exact must not be side-sensitive. The sampler randomises the side independently
    # per pool, so a measure that demanded the same slot would report roughly half the
    # true agreement rate -- it did exactly that before this assertion existed.
    assert result["exact"] == 100.0
    assert result["flipped_order_n"] == 1
    assert result["same_order_n"] == 0


def test_cell_agreement_treats_both_bad_as_equivalent_to_a_draw(arena):
    """`rating.py` scores both as half a win, so agreement must score them alike.

    Counting these as a disagreement would penalise the panel for a vocabulary
    difference rather than a difference of opinion.
    """

    arena.append_vote("p0", "justext", "goose3", "draw", judge=arena.LLM)
    arena.append_vote("p0", "justext", "goose3", "both_bad", judge=arena.HUMAN)

    result = stats.judge_cell_agreement()
    assert result["equivalent"] == 100.0
    # `exact` is the one measure that still separates the two no-winner vocabularies,
    # which is the whole reason it is reported next to `equivalent`.
    assert result["exact"] == 0.0
    # Neither side named a winner, so the decisive-only measure has nothing to say.
    assert result["decisive_n"] == 0
    assert result["decisive"] is None


def test_cell_agreement_detects_a_real_disagreement(arena):

    arena.append_vote("p0", "trafilatura", "goose3", "a", judge=arena.LLM)
    arena.append_vote("p0", "trafilatura", "goose3", "b", judge=arena.HUMAN)

    result = stats.judge_cell_agreement()
    assert result["cells"] == 1
    assert result["equivalent"] == 0.0
    assert len(result["disagreements"]) == 1
    row = result["disagreements"][0]
    assert row["human"] == "goose3"
    assert row["llm"] == "trafilatura"


def test_cell_agreement_ignores_auto_draws(arena):
    """Byte-identical output is not a judgement, so it cannot be agreed with."""

    arena.append_vote("p0", "trafilatura", "ours", "draw", judge=arena.LLM, auto=True)
    arena.append_vote("p0", "trafilatura", "ours", "a", judge=arena.HUMAN)

    result = stats.judge_cell_agreement()
    assert result["cells"] == 0
    assert result["candidates"] == 0


# ------------------------------------------------------- head-to-head by shape


def test_head_to_head_splits_index_like_from_article_pages(arena):
    """The split is the finding, not a breakdown of it.

    `favor_precision=True` empties link-index pages and slightly enriches prose, so a
    single win rate over both populations averages the effect to nothing.
    """

    with arena.connect() as connection:
        connection.execute("update pages set shape = 'listing' where id = 'p0'")
        connection.execute("update pages set shape = 'biography' where id = 'p1'")

    arena.append_vote("p0", "ours", "trafilatura", "b", reason="duel")
    arena.append_vote("p1", "trafilatura", "ours", "b", reason="duel")

    result = stats.head_to_head_by_shape("ours", "trafilatura")
    assert result["games"] == 2
    assert result["duel_votes"] == 2

    index_like = result["split"]["index-like"]
    article = result["split"]["article"]
    # On the listing page trafilatura won; on the article page `ours` did. Both votes
    # are "b", so a naive tally that ignored which side each entrant sat on would
    # report the same winner twice.
    assert (index_like["wins_a"], index_like["wins_b"]) == (0, 1)
    assert (article["wins_a"], article["wins_b"]) == (1, 0)
    assert index_like["win_rate_a"] == 0.0
    assert article["win_rate_a"] == 100.0


def test_head_to_head_ignores_other_pairs_and_pools(arena):

    arena.append_vote("p0", "ours", "goose3", "a")
    arena.append_vote("p1", "ours", "trafilatura", "a", judge=arena.LLM)

    result = stats.head_to_head_by_shape("ours", "trafilatura")
    assert result["games"] == 0
    assert stats.head_to_head_by_shape("ours", "trafilatura", arena.LLM)["games"] == 1


def test_head_to_head_marks_which_votes_were_semi_blind(arena):
    """Duel votes name the pair up front, so they must stay distinguishable."""

    arena.append_vote("p0", "ours", "trafilatura", "a", reason="duel")
    arena.append_vote("p1", "ours", "trafilatura", "a")

    result = stats.head_to_head_by_shape("ours", "trafilatura")
    assert result["games"] == 2
    assert result["duel_votes"] == 1
    flags = sorted(
        row["duel"] for entry in result["split"].values() for row in entry["pages"]
    )
    assert flags == [False, True]


# ----------------------------------------------------------- subset standings


def _wiki_trio(arena):
    """Three entrants over two wiki pages, plus a page wikiextractor never ran on."""
    with arena.connect() as connection:
        connection.execute("update pages set is_wiki = 1, lang = 'en' where id = 'p0'")
        connection.execute("update pages set is_wiki = 1, lang = 'nl' where id = 'p1'")
        for page_id in ("p0", "p1"):
            for name in ("ours", "trafilatura", "wikiextractor-v2"):
                connection.execute(
                    "insert into runs (page_id, entrant, variant, status,"
                    " output_sha256, output_chars, ran_at)"
                    " values (?, ?, 'raw', 'ok', ?, 100, ?)",
                    (page_id, name, f"{page_id}-{name}", arena.now()),
                )
        # p2 has no wikitext, so only the two HTML entrants ran there.
        for name in ("ours", "trafilatura"):
            connection.execute(
                "insert into runs (page_id, entrant, variant, status,"
                " output_sha256, output_chars, ran_at)"
                " values ('p2', ?, 'raw', 'ok', ?, 100, ?)",
                (name, f"p2-{name}", arena.now()),
            )


def test_subset_standings_uses_only_the_shared_page_population(arena):
    """A vote on a page outside the shared population must not enter the standings.

    That population is the entire reason a 3-way including wikiextractor-v2 is valid:
    every entrant is rated over the same documents. Letting an out-of-population vote
    leak in would quietly reintroduce the bias the restriction removes.
    """

    _wiki_trio(arena)
    arena.append_vote("p0", "ours", "trafilatura", "b", reason="focus")
    arena.append_vote("p2", "ours", "trafilatura", "a")  # outside the trio population

    result = stats.subset_standings(("ours", "trafilatura", "wikiextractor-v2"))
    assert result["pages"] == 2
    assert result["votes"] == 1
    assert result["focus_votes"] == 1


def test_subset_standings_ranks_and_splits_by_language(arena):

    _wiki_trio(arena)
    arena.append_vote("p0", "wikiextractor-v2", "ours", "a", reason="focus")
    arena.append_vote("p1", "ours", "wikiextractor-v2", "b", reason="focus")

    result = stats.subset_standings(("ours", "trafilatura", "wikiextractor-v2"))
    assert [row["entrant"] for row in result["rows"]][0] == "wikiextractor-v2"
    assert set(result["per_language"]) == {"en", "other"}
    assert result["per_language"]["en"]["votes"] == 1
    assert result["per_language"]["other"]["votes"] == 1


def test_subset_standings_ignores_votes_touching_outsiders(arena):

    _wiki_trio(arena)
    arena.append_vote("p0", "ours", "goose3", "a")

    result = stats.subset_standings(("ours", "trafilatura", "wikiextractor-v2"))
    assert result["votes"] == 0

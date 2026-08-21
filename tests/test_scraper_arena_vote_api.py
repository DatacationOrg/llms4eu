"""Does POST /api/vote actually tag a duel vote? Against a throwaway database."""
import pytest
from fastapi.testclient import TestClient

from research.scrapers import store


@pytest.fixture
def client(tmp_path, monkeypatch):
    monkeypatch.setattr(store, "ROOT", tmp_path)
    config = dict(store.config())
    config["data_dir"] = "arena"
    monkeypatch.setattr(store, "config", lambda: config)
    store.initialize()
    with store.connect() as connection:
        connection.execute(
            "insert into pages (id, url, bucket, lang, shape, is_wiki, title)"
            " values ('p0', 'https://example.org/0', 'en-other', 'en', 'listing', 0, 'T')"
        )
    from research.scrapers.arena.app import app

    return TestClient(app)


def test_duel_vote_is_tagged(client):
    response = client.post(
        "/api/vote",
        json={
            "page_id": "p0",
            "entrant_a": "ours",
            "entrant_b": "trafilatura",
            "winner": "a",
            "mode": "duel",
        },
    )
    assert response.status_code == 200, response.text
    votes = store.load_votes(judge=store.HUMAN)
    assert [v["reason"] for v in votes] == ["duel"]


def test_explore_vote_is_not_tagged(client):
    client.post(
        "/api/vote",
        json={
            "page_id": "p0",
            "entrant_a": "ours",
            "entrant_b": "trafilatura",
            "winner": "b",
        },
    )
    assert [v["reason"] for v in store.load_votes(judge=store.HUMAN)] == [None]


def test_cross_site_vote_is_refused(client):
    """A page on another origin must not be able to append to the vote log.

    The server binds 127.0.0.1, which keeps other machines out but not other pages:
    any site open in the reviewer's browser can POST to localhost. `Sec-Fetch-Site`
    is set by the browser, not by page script, so it is the check.
    """
    response = client.post(
        "/api/vote",
        headers={"Sec-Fetch-Site": "cross-site"},
        json={
            "page_id": "p0",
            "entrant_a": "ours",
            "entrant_b": "trafilatura",
            "winner": "a",
        },
    )
    assert response.status_code == 403
    assert store.load_votes(judge=store.HUMAN) == []


def test_same_origin_vote_is_accepted(client):
    response = client.post(
        "/api/vote",
        headers={"Sec-Fetch-Site": "same-origin"},
        json={
            "page_id": "p0",
            "entrant_a": "ours",
            "entrant_b": "trafilatura",
            "winner": "a",
        },
    )
    assert response.status_code == 200, response.text
    assert len(store.load_votes(judge=store.HUMAN)) == 1


def test_cross_site_undo_is_refused(client):
    assert (
        client.post("/api/undo", headers={"Sec-Fetch-Site": "cross-site"}).status_code
        == 403
    )


def test_unknown_focus_mode_is_404_not_the_default_round(client):
    """A typo must not silently serve the default mode.

    That would look like the round being reviewed while recording votes against a
    different entrant set.
    """
    assert client.get("/focus/content3").status_code == 200
    assert client.get("/focus/no-such-mode").status_code == 404


def test_snapshot_rejects_an_unknown_page_id(client):
    assert client.get("/snapshot/p0?variant=raw").status_code == 404  # no file on disk
    assert client.get("/snapshot/not-a-page?variant=raw").status_code == 404
    assert client.get("/snapshot/p0?variant=bogus").status_code == 400

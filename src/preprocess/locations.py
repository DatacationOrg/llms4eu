"""Give each page the point it is about, where Wikidata knows one.

A single-site source (a castle's own website) is about one configured item; a
Wikipedia page is about its article's item. Everything else stays unlocated,
which geo retrieval treats as neutral. People get no point by construction.
"""

from __future__ import annotations

import re
from pathlib import Path
from urllib.parse import unquote, urlparse

from src.db.legacy.pages import connect_pages as connect
from src.db.legacy.pages import initialize_page_artifacts_db
from src.shared import wikidata
from src.shared.env import load_yaml

CONFIG = load_yaml(Path(__file__).with_name("config.yaml"))
WIKIPEDIA = re.compile(r"^([a-z-]+)\.(?:m\.)?wikipedia\.org$")


def page_qid(source: str, url: str) -> str | None:
    parsed = urlparse(url)
    wiki = WIKIPEDIA.match(parsed.netloc)
    if wiki and parsed.path.startswith("/wiki/"):
        title = unquote(parsed.path.removeprefix("/wiki/")).replace("_", " ")
        return wikidata.qid_for_title(wiki.group(1), title)
    return CONFIG["source_locations"].get(source)


def locate_pages() -> None:
    initialize_page_artifacts_db()
    with connect() as conn:
        pages = conn.execute(
            "select id, source, coalesce(final_url, url) as url from page_metadata"
        ).fetchall()
    qids = {page["id"]: page_qid(page["source"], page["url"]) for page in pages}
    points = wikidata.coordinates([qid for qid in qids.values() if qid])
    rows = [
        (page_id, qid, *points[qid]) for page_id, qid in qids.items() if qid in points
    ]
    with connect() as conn:
        conn.execute("delete from page_locations")
        conn.executemany("insert into page_locations values (?, ?, ?, ?)", rows)
    print(f"located {len(rows)} of {len(pages)} pages")


if __name__ == "__main__":
    locate_pages()

"""SQLite storage for the scraper arena.

Everything the arena needs is derived from four tables. Votes are append-only so
ratings can always be recomputed from scratch; see `rating.py`.
"""

from __future__ import annotations

import json
import sqlite3
import uuid
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterator

from src.shared.env import ROOT, load_yaml


CONFIG_PATH = Path(__file__).resolve().parent / "config.yaml"

SCHEMA = """
create table if not exists pages (
  id        text primary key,
  url       text not null unique,
  bucket    text not null,
  lang      text not null,
  shape     text not null default 'prose',
  is_wiki   integer not null default 0,
  title     text,
  note      text,
  dropped   integer not null default 0,
  drop_reason text
);

create table if not exists snapshots (
  page_id      text not null references pages(id) on delete cascade,
  variant      text not null,
  path         text,
  bytes        integer not null default 0,
  sha256       text,
  status_code  integer,
  final_url    text,
  content_type text,
  charset      text,
  fetch_ms     real,
  fetched_at   text not null,
  error        text,
  -- 1 = the live URL can be shown in an <iframe>, 0 = X-Frame-Options or a
  -- restrictive CSP frame-ancestors blocks it, null = not measured. The arena's
  -- reference pane is a live iframe where this allows it and the saved DOM where it
  -- does not; 29 of the corpus's 52 domains block framing, all Wikipedia hosts allow it.
  frameable    integer,
  primary key (page_id, variant)
);

create table if not exists runs (
  page_id       text not null references pages(id) on delete cascade,
  entrant       text not null,
  variant       text not null,
  output        text,
  output_chars  integer not null default 0,
  output_sha256 text,
  extract_ms    real,
  extract_ms_all text,
  input_ms      real,
  status        text not null,
  error         text,
  ran_at        text not null,
  primary key (page_id, entrant)
);

-- Append-only. entrant_a/entrant_b are the LEFT/RIGHT columns as displayed, so
-- position bias stays measurable after the fact.
--
-- `judge` partitions votes into independent pools ('human', 'llm'). The pools are
-- never mixed into one rating: an LLM panel and a human reviewer are different
-- instruments, and averaging them would produce a number that describes neither.
-- Keeping them apart also means automated judging cannot consume the human's
-- unseen matchups, because each pool samples against only its own history.
create table if not exists votes (
  id         integer primary key autoincrement,
  page_id    text not null,
  entrant_a  text not null,
  entrant_b  text not null,
  winner     text not null,
  auto       integer not null default 0,
  latency_ms real,
  voted_at   text not null,
  judge      text not null default 'human',
  reason     text
);

"""

# Indexes are applied after _migrate(), because an index over a column added by a
# migration cannot be created until that column exists.
INDEXES = """
create index if not exists votes_pair on votes (entrant_a, entrant_b);
create index if not exists votes_judge on votes (judge);
create index if not exists votes_page on votes (page_id);
create index if not exists runs_entrant on runs (entrant);
"""


def config() -> dict:
    return load_yaml(CONFIG_PATH)


def data_dir() -> Path:
    path = ROOT / config()["data_dir"]
    path.mkdir(parents=True, exist_ok=True)
    return path


def db_path() -> Path:
    return data_dir() / "arena.db"


def snapshot_dir(page_id: str) -> Path:
    path = data_dir() / "snapshots" / page_id
    path.mkdir(parents=True, exist_ok=True)
    return path


HUMAN = "human"
LLM = "llm"


def votes_jsonl() -> Path:
    return data_dir() / "votes.jsonl"


def fetch_layer_json() -> Path:
    """Where `just arena-fetchers` writes its measurements.

    One accessor because four modules read this file and each was spelling the path
    itself; the shapes they want out of it differ, but the location does not.
    """
    return data_dir() / "fetch_layer.json"


def page_id(url: str) -> str:
    """Same scheme as src/scraping/page_fetch.py, so ids line up across the repo."""
    return str(uuid.uuid5(uuid.NAMESPACE_URL, url))


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


@contextmanager
def connect() -> Iterator[sqlite3.Connection]:
    connection = sqlite3.connect(db_path(), timeout=30.0)
    connection.row_factory = sqlite3.Row
    connection.execute("pragma journal_mode=wal")
    connection.execute("pragma foreign_keys=on")
    try:
        yield connection
        connection.commit()
    finally:
        connection.close()


def initialize() -> None:
    with connect() as connection:
        connection.executescript(SCHEMA)
        _migrate(connection)
        connection.executescript(INDEXES)


def _migrate(connection: sqlite3.Connection) -> None:
    """Add columns that postdate the original schema.

    `create table if not exists` is a no-op on an existing table, so new columns
    have to be added explicitly. Votes are the one thing here that cannot be
    regenerated -- snapshots and runs can always be recomputed -- so this migrates
    in place rather than asking for a rebuild.
    """
    existing = {row["name"] for row in connection.execute("pragma table_info(votes)")}
    if "judge" not in existing:
        connection.execute(
            "alter table votes add column judge text not null default 'human'"
        )
    if "reason" not in existing:
        connection.execute("alter table votes add column reason text")

    snaps = {row["name"] for row in connection.execute("pragma table_info(snapshots)")}
    if "frameable" not in snaps:
        connection.execute("alter table snapshots add column frameable integer")


@dataclass(frozen=True)
class Page:
    id: str
    url: str
    bucket: str
    lang: str
    shape: str
    is_wiki: bool
    title: str | None = None
    note: str | None = None


def upsert_pages(pages: list[Page]) -> None:
    with connect() as connection:
        connection.executemany(
            """insert into pages (id, url, bucket, lang, shape, is_wiki, title, note)
               values (?, ?, ?, ?, ?, ?, ?, ?)
               on conflict(id) do update set
                 bucket=excluded.bucket, lang=excluded.lang, shape=excluded.shape,
                 is_wiki=excluded.is_wiki, note=excluded.note""",
            [
                (
                    p.id,
                    p.url,
                    p.bucket,
                    p.lang,
                    p.shape,
                    int(p.is_wiki),
                    p.title,
                    p.note,
                )
                for p in pages
            ],
        )


def load_pages(include_dropped: bool = False) -> list[dict]:
    query = "select * from pages"
    if not include_dropped:
        query += " where dropped = 0"
    with connect() as connection:
        return [
            dict(row) for row in connection.execute(query + " order by bucket, url")
        ]


def mark_dropped(page_id_value: str, reason: str) -> None:
    with connect() as connection:
        connection.execute(
            "update pages set dropped = 1, drop_reason = ? where id = ?",
            (reason, page_id_value),
        )


def set_title(page_id_value: str, title: str | None) -> None:
    with connect() as connection:
        connection.execute(
            "update pages set title = coalesce(?, title) where id = ?",
            (title, page_id_value),
        )


SNAPSHOT_COLUMNS = (
    "path",
    "bytes",
    "sha256",
    "status_code",
    "final_url",
    "content_type",
    "charset",
    "fetch_ms",
    "error",
    "frameable",
)


def record_snapshot(page_id_value: str, variant: str, **fields) -> None:
    """Insert a snapshot row, or update only the fields actually passed.

    Updating just the passed fields, rather than every column, is deliberate. The
    obvious implementation writes `None` over every field the caller omitted, which
    turns an innocent-looking `record_snapshot(page, variant, sha256=...)` -- exactly
    what re-hashing saved bodies wants to do -- into silent destruction of the path,
    status code and latency measured by the original fetch. That happened.
    """
    unknown = set(fields) - set(SNAPSHOT_COLUMNS)
    if unknown:
        raise ValueError(f"unknown snapshot fields: {sorted(unknown)}")
    if "bytes" in fields and fields["bytes"] is None:
        fields["bytes"] = 0

    insert_columns = list(SNAPSHOT_COLUMNS)
    insert_values = [fields.get(column) for column in insert_columns]
    if fields.get("bytes") is None:
        insert_values[insert_columns.index("bytes")] = 0
    updates = ", ".join(f"{column}=excluded.{column}" for column in fields)
    # An update that carries no new fetch does not restamp fetched_at either: the
    # timestamp records when the bytes were retrieved, not when they were re-read.
    if "path" in fields or "error" in fields or "fetch_ms" in fields:
        updates = f"fetched_at=excluded.fetched_at, {updates}" if updates else ""
    with connect() as connection:
        connection.execute(
            f"""insert into snapshots
                  (page_id, variant, fetched_at, {", ".join(insert_columns)})
                values (?, ?, ?, {", ".join("?" * len(insert_columns))})
                on conflict(page_id, variant) do update set
                  {updates or "sha256=snapshots.sha256"}""",
            [page_id_value, variant, now(), *insert_values],
        )


def load_snapshots(variant: str | None = None) -> list[dict]:
    query = "select * from snapshots"
    params: list = []
    if variant:
        query += " where variant = ?"
        params.append(variant)
    with connect() as connection:
        return [dict(row) for row in connection.execute(query, params)]


def load_snapshots_for(page_id_value: str) -> list[dict]:
    """Every snapshot variant for one page."""
    with connect() as connection:
        return [
            dict(row)
            for row in connection.execute(
                "select * from snapshots where page_id = ?", (page_id_value,)
            )
        ]


def snapshot_text(page_id_value: str, variant: str) -> str | None:
    """Read a snapshot's bytes off disk, or None if it is missing or errored."""
    with connect() as connection:
        row = connection.execute(
            "select path, error from snapshots where page_id = ? and variant = ?",
            (page_id_value, variant),
        ).fetchone()
    if not row or row["error"] or not row["path"]:
        return None
    path = ROOT / row["path"]
    if not path.exists():
        return None
    return path.read_text(encoding="utf-8", errors="replace")


def record_run(page_id_value: str, entrant: str, **fields) -> None:
    columns = [
        "variant",
        "output",
        "output_chars",
        "output_sha256",
        "extract_ms",
        "extract_ms_all",
        "input_ms",
        "status",
        "error",
    ]
    if isinstance(fields.get("extract_ms_all"), list):
        fields["extract_ms_all"] = json.dumps(fields["extract_ms_all"])
    if fields.get("output_chars") is None:
        fields["output_chars"] = 0
    values = [fields.get(column) for column in columns]
    with connect() as connection:
        connection.execute(
            f"""insert into runs (page_id, entrant, ran_at, {", ".join(columns)})
                values (?, ?, ?, {", ".join("?" * len(columns))})
                on conflict(page_id, entrant) do update set
                  ran_at=excluded.ran_at,
                  {", ".join(f"{c}=excluded.{c}" for c in columns)}""",
            [page_id_value, entrant, now(), *values],
        )


def load_runs(page_id_value: str | None = None) -> list[dict]:
    query = "select * from runs"
    params: list = []
    if page_id_value:
        query += " where page_id = ?"
        params.append(page_id_value)
    with connect() as connection:
        return [dict(row) for row in connection.execute(query, params)]


def load_run_index() -> list[dict]:
    """Run metadata without the output text.

    The arena calls this on every matchup; pulling the `output` column too would
    move tens of megabytes per request for no reason.
    """
    with connect() as connection:
        return [
            dict(row)
            for row in connection.execute(
                """select page_id, entrant, variant, status, output_sha256,
                          output_chars, extract_ms, extract_ms_all, input_ms
                   from runs"""
            )
        ]


def run_output(page_id_value: str, entrant: str) -> str | None:
    with connect() as connection:
        row = connection.execute(
            "select output from runs where page_id = ? and entrant = ?",
            (page_id_value, entrant),
        ).fetchone()
    return row["output"] if row else None


# Below this a pool cannot meaningfully order eleven entrants -- the ratings are
# mostly the 1500 prior -- so it is not shown as the primary leaderboard. Matches the
# vote count at which the report starts computing bootstrap intervals.
MIN_POOL_VOTES = 10


def append_vote(
    page_id_value: str,
    entrant_a: str,
    entrant_b: str,
    winner: str,
    auto: bool = False,
    latency_ms: float | None = None,
    judge: str = HUMAN,
    reason: str | None = None,
) -> int:
    """Insert a vote and mirror it to JSONL, so a DB mishap never loses judgements."""
    stamped = now()
    with connect() as connection:
        cursor = connection.execute(
            """insert into votes (page_id, entrant_a, entrant_b, winner, auto,
                                  latency_ms, voted_at, judge, reason)
               values (?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (
                page_id_value,
                entrant_a,
                entrant_b,
                winner,
                int(auto),
                latency_ms,
                stamped,
                judge,
                reason,
            ),
        )
        vote_id = int(cursor.lastrowid or 0)

    record = {
        "id": vote_id,
        "page_id": page_id_value,
        "entrant_a": entrant_a,
        "entrant_b": entrant_b,
        "winner": winner,
        "auto": int(auto),
        "latency_ms": latency_ms,
        "voted_at": stamped,
        "judge": judge,
        "reason": reason,
    }
    with votes_jsonl().open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record) + "\n")
    return vote_id


def invalidate_votes(vote_ids: list[int], reason: str) -> int:
    """Retract votes cast on output that has since changed.

    Needed whenever an extractor adapter is fixed after judging has started: a
    vote compares two specific strings, so once one of them is regenerated the
    vote is evidence about output that no longer exists. Keeping it would quietly
    poison the ratings.

    The rows leave the `votes` table -- the rating code has no notion of a
    retracted vote and should not grow one -- but an `invalidated` event is
    appended to the JSONL mirror, so the audit trail stays append-only and the
    retraction itself is recoverable.
    """
    if not vote_ids:
        return 0
    placeholders = ",".join("?" for _ in vote_ids)
    with connect() as connection:
        rows = [
            dict(row)
            for row in connection.execute(
                f"select * from votes where id in ({placeholders})", vote_ids
            )
        ]
        connection.execute(f"delete from votes where id in ({placeholders})", vote_ids)

    with votes_jsonl().open("a", encoding="utf-8") as handle:
        handle.write(
            json.dumps(
                {
                    "invalidated": [row["id"] for row in rows],
                    "reason": reason,
                    "votes": rows,
                    "invalidated_at": now(),
                }
            )
            + "\n"
        )
    return len(rows)


def reset_votes(judge: str | None = None) -> int:
    """Drop every vote, archiving the JSONL mirror rather than deleting it.

    Used to clear smoke-test votes before a real session. Snapshots and extractor
    runs are untouched, so no re-fetching or re-extraction is needed.
    """
    where = "" if judge is None else " where judge = ?"
    params: list = [] if judge is None else [judge]
    with connect() as connection:
        removed = int(
            connection.execute(f"select count(*) from votes{where}", params).fetchone()[
                0
            ]
        )
        connection.execute(f"delete from votes{where}", params)
        if judge is None:
            connection.execute("delete from sqlite_sequence where name = 'votes'")

    mirror = votes_jsonl()
    if judge is None:
        # Whole log cleared: rotate the mirror, keeping it rather than deleting it.
        if mirror.exists():
            stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
            mirror.rename(mirror.with_suffix(f".{stamp}.jsonl"))
    elif removed:
        # One pool cleared. Rotating here would strip the *other* pool's records
        # from the live mirror while they are still in the database, leaving the two
        # silently inconsistent. Append the clearance as an event instead.
        with mirror.open("a", encoding="utf-8") as handle:
            handle.write(
                json.dumps(
                    {"cleared_pool": judge, "removed": removed, "cleared_at": now()}
                )
                + "\n"
            )
    scope = "all pools" if judge is None else f"pool {judge!r}"
    print(f"cleared {removed} votes ({scope}); snapshots and runs kept")
    return removed


def undo_last_vote(judge: str = HUMAN) -> dict | None:
    """Remove the most recent non-auto vote.

    Votes are append-only by design so ratings stay recomputable, but over a
    session of hundreds of judgements a misclick is inevitable, and an unremovable
    wrong vote is worse for data quality than a deletion that is written down.
    The removal is appended to the JSONL mirror as an explicit `undo` record
    rather than erasing history.
    """
    with connect() as connection:
        row = connection.execute(
            "select * from votes where auto = 0 and judge = ? order by id desc limit 1",
            (judge,),
        ).fetchone()
        if row is None:
            return None
        removed = dict(row)
        connection.execute("delete from votes where id = ?", (removed["id"],))

    with votes_jsonl().open("a", encoding="utf-8") as handle:
        handle.write(json.dumps({"undo": removed, "undone_at": now()}) + "\n")
    return removed


def load_votes(include_auto: bool = True, judge: str | None = HUMAN) -> list[dict]:
    """Votes for one judge pool.

    `judge` defaults to the human pool rather than to everything: an accidental
    `load_votes()` that silently blended an LLM panel into the human leaderboard
    would be a hard error to notice by eye. Pass `judge=None` to get all pools.
    """
    clauses = []
    params: list = []
    if not include_auto:
        clauses.append("auto = 0")
    if judge is not None:
        clauses.append("judge = ?")
        params.append(judge)
    query = "select * from votes"
    if clauses:
        query += " where " + " and ".join(clauses)
    with connect() as connection:
        return [dict(row) for row in connection.execute(query + " order by id", params)]


def vote_count(judge: str | None = HUMAN) -> tuple[int, int]:
    """Return (judged votes, auto-draws) for one pool."""
    query = (
        "select coalesce(sum(auto = 0), 0) as judged, "
        "coalesce(sum(auto = 1), 0) as auto from votes"
    )
    params: list = []
    if judge is not None:
        query += " where judge = ?"
        params.append(judge)
    with connect() as connection:
        row = connection.execute(query, params).fetchone()
    return int(row["judged"]), int(row["auto"])


def judge_pools() -> dict[str, int]:
    """Vote count per pool, for the report and for sanity-checking a session."""
    with connect() as connection:
        return {
            row["judge"]: int(row["n"])
            for row in connection.execute(
                "select judge, count(*) as n from votes group by judge order by judge"
            )
        }

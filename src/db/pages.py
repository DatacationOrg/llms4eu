from __future__ import annotations

import os
import sqlite3
from dataclasses import dataclass, replace
from datetime import datetime, timezone
from pathlib import Path

from src.shared.env import ROOT, load_local_env, load_yaml
from src.shared.geo_scope import GeoScope
from src.shared.geocode import Coordinates, haversine_km

CONFIG = load_yaml(Path(__file__).with_name("config.yaml"))


def raw_pages_db_path() -> Path:
    """Path to the page database.

    `PAGES_DB_PATH` overrides the configured location. Experiments that change
    the schema or write new chunk variants can then run against a copy under
    `.local/` instead of the durable database in `data/db/`.
    """
    load_local_env()
    path = ROOT / os.getenv("PAGES_DB_PATH", CONFIG["raw_pages_db"])
    path.parent.mkdir(parents=True, exist_ok=True)
    return path


def connect_pages() -> sqlite3.Connection:
    conn = sqlite3.connect(raw_pages_db_path())
    conn.execute("pragma foreign_keys = on")
    conn.row_factory = sqlite3.Row
    return conn


def initialize_page_artifacts_db() -> None:
    schema = (ROOT / "sql" / "eval.sql").read_text(encoding="utf-8")
    with connect_pages() as conn:
        conn.executescript(schema)
        question_columns = {
            row["name"] for row in conn.execute("pragma table_info(eval_questions)")
        }
        if "approved" not in question_columns:
            conn.execute(
                "alter table eval_questions add column approved integer not null default 1"
            )
        page_columns = {
            row["name"] for row in conn.execute("pragma table_info(page_metadata)")
        }
        if "language" not in page_columns:
            # Nullable and unset: a page falls back to its source's language
            # until detection fills this in.
            conn.execute("alter table page_metadata add column language text")
        chunk_columns = {
            row["name"] for row in conn.execute("pragma table_info(page_chunks)")
        }
        if "summary" in chunk_columns:
            _drop_page_chunk_summary(conn)
            chunk_columns = {
                row["name"] for row in conn.execute("pragma table_info(page_chunks)")
            }
        if "variant" not in chunk_columns:
            _add_page_chunk_variant(conn)
        conn.execute(
            "create index if not exists idx_page_chunks_variant on page_chunks(variant)"
        )
        sources = [
            row["source"]
            for row in conn.execute("select distinct source from page_metadata")
        ]
        conn.executemany(
            "insert or ignore into page_sources (source, language) values (?, ?)",
            [(source, CONFIG["default_source_language"]) for source in sources],
        )


def _add_page_chunk_variant(conn: sqlite3.Connection) -> None:
    """Give page_chunks a variant and character span, keeping existing ids.

    The unique constraint moves from (page_id, chunk_index) to
    (page_id, variant, chunk_index), which SQLite cannot express as an ALTER, so
    the table is rebuilt. Existing rows become the 'base' variant and keep their
    `{page_id}:{index}` ids, so the approved eval labels and every checkpointed
    ranking stay valid.
    """
    conn.commit()
    conn.execute("pragma foreign_keys = off")
    try:
        conn.executescript(
            """
            drop table if exists page_chunks_with_variant;

            create table page_chunks_with_variant (
              id text primary key,
              page_id text not null references page_metadata(id) on delete cascade,
              chunk_index integer not null,
              variant text not null default 'base',
              heading_path text,
              text text not null,
              char_count integer not null,
              start_char integer,
              end_char integer,
              unique(page_id, variant, chunk_index)
            );

            insert into page_chunks_with_variant (
              id, page_id, chunk_index, variant, heading_path, text, char_count
            )
            select id, page_id, chunk_index, 'base', heading_path, text, char_count
            from page_chunks;

            drop table page_chunks;
            alter table page_chunks_with_variant rename to page_chunks;

            create index if not exists idx_page_chunks_page_id
              on page_chunks(page_id);
            """
        )
    finally:
        conn.execute("pragma foreign_keys = on")


def _drop_page_chunk_summary(conn: sqlite3.Connection) -> None:
    try:
        conn.execute("alter table page_chunks drop column summary")
    except sqlite3.OperationalError:
        conn.commit()
        conn.execute("pragma foreign_keys = off")
        try:
            conn.executescript(
                """
                drop table if exists page_chunks_without_summary;

                create table page_chunks_without_summary (
                  id text primary key,
                  page_id text not null references page_metadata(id) on delete cascade,
                  chunk_index integer not null,
                  heading_path text,
                  text text not null,
                  char_count integer not null,
                  unique(page_id, chunk_index)
                );

                insert into page_chunks_without_summary (
                  id, page_id, chunk_index, heading_path, text, char_count
                )
                select id, page_id, chunk_index, heading_path, text, char_count
                from page_chunks;

                drop table page_chunks;
                alter table page_chunks_without_summary rename to page_chunks;

                create index if not exists idx_page_chunks_page_id
                  on page_chunks(page_id);
                """
            )
        finally:
            conn.execute("pragma foreign_keys = on")


# --- page chunks -----------------------------------------------------------

# One query hydrates chunks for indexing, BM25 and readiness alike, so the geo
# columns cannot drift between the dense and sparse corpora. The primary
# location is a left join: a page without one is a chunk without codes, never a
# missing chunk.
CHUNK_ROWS_SQL = """
    select c.id, c.page_id, c.chunk_index, c.heading_path, c.text,
           m.title, m.source, m.page_kind,
           coalesce(m.language, s.language) as language,
           l.name as location_name, l.country_code, l.nuts2, l.nuts3,
           l.nuts3_name, l.latitude, l.longitude
    from page_chunks c
    join page_metadata m on m.id = c.page_id
    left join page_sources s on s.source = m.source
    left join page_locations l on l.page_id = c.page_id and l.role = 'primary'
    where c.variant = ?
    order by c.id
"""


def load_chunk_rows(variant: str) -> list[sqlite3.Row]:
    with connect_pages() as conn:
        return conn.execute(CHUNK_ROWS_SQL, (variant,)).fetchall()


# --- page locations ---------------------------------------------------------


@dataclass(frozen=True)
class PageLocation:
    """One row of `page_locations`; see sql/eval.sql for the field contract."""

    page_id: str
    role: str
    location_key: str
    name: str | None = None
    wikidata_qid: str | None = None
    latitude: float | None = None
    longitude: float | None = None
    granularity: str = "point"
    country_code: str | None = None
    nuts2: str | None = None
    nuts3: str | None = None
    nuts3_name: str | None = None
    iso_3166_2: str | None = None
    confidence: float | None = None
    method: str = "manual"

    @property
    def coordinates(self) -> Coordinates | None:
        if self.latitude is None or self.longitude is None:
            return None
        return Coordinates(self.latitude, self.longitude)


_LOCATION_COLUMNS = tuple(PageLocation.__dataclass_fields__)


def replace_page_locations(page_id: str, locations: list[PageLocation]) -> None:
    """Make `locations` the complete location set of one page."""
    with connect_pages() as conn:
        conn.execute("delete from page_locations where page_id = ?", (page_id,))
        conn.executemany(
            f"insert into page_locations ({', '.join(_LOCATION_COLUMNS)}) "
            f"values ({', '.join('?' for _ in _LOCATION_COLUMNS)})",
            [
                tuple(getattr(location, column) for column in _LOCATION_COLUMNS)
                for location in locations
            ],
        )


def page_locations(page_ids: list[str] | None = None) -> dict[str, list[PageLocation]]:
    sql = f"select {', '.join(_LOCATION_COLUMNS)} from page_locations"
    params: tuple = ()
    if page_ids is not None:
        if not page_ids:
            return {}
        sql += f" where page_id in ({', '.join('?' for _ in page_ids)})"
        params = tuple(page_ids)
    sql += " order by page_id, role desc, location_key"
    out: dict[str, list[PageLocation]] = {}
    with connect_pages() as conn:
        for row in conn.execute(sql, params):
            location = PageLocation(**dict(row))
            out.setdefault(location.page_id, []).append(location)
    return out


def primary_locations() -> dict[str, PageLocation]:
    return {
        page_id: next(loc for loc in locations if loc.role == "primary")
        for page_id, locations in page_locations().items()
        if any(loc.role == "primary" for loc in locations)
    }


def load_chunk_locations(chunk_ids: list[str]) -> dict[str, PageLocation]:
    """Primary location (coordinates and codes) of the page behind each chunk.

    Chunks of unlocated pages are absent from the result, which the soft geo
    path reads as "neutral", never as "outside".
    """
    if not chunk_ids:
        return {}
    placeholders = ", ".join("?" for _ in chunk_ids)
    columns = ", ".join(f"l.{column}" for column in _LOCATION_COLUMNS)
    with connect_pages() as conn:
        rows = conn.execute(
            f"""
            select c.id as chunk_id, {columns}
            from page_chunks c
            join page_locations l on l.page_id = c.page_id and l.role = 'primary'
            where c.id in ({placeholders})
            """,
            chunk_ids,
        ).fetchall()
    result = {}
    for row in rows:
        fields = dict(row)
        chunk_id = fields.pop("chunk_id")
        result[chunk_id] = PageLocation(**fields)
    return result


def located_page_share(scope: GeoScope) -> float | None:
    """Share of primary-located pages inside the scope's most specific level.

    The selectivity gate: a level that keeps nearly every located page carries
    no information on this corpus. None when the scope filters nothing or no
    page is located.
    """
    in_scope = page_ids_in_scope(replace(scope, include_null=False))
    if in_scope is None:
        return None
    with connect_pages() as conn:
        total = conn.execute(
            "select count(*) from page_locations where role = 'primary'"
        ).fetchone()[0]
    if not total:
        return None
    return len(in_scope) / total


def load_chunk_coordinates(chunk_ids: list[str]) -> dict[str, Coordinates]:
    """Primary-location coordinates of the pages behind these chunks."""
    if not chunk_ids:
        return {}
    placeholders = ", ".join("?" for _ in chunk_ids)
    with connect_pages() as conn:
        rows = conn.execute(
            f"""
            select c.id as chunk_id, l.latitude, l.longitude
            from page_chunks c
            join page_locations l on l.page_id = c.page_id and l.role = 'primary'
            where c.id in ({placeholders})
              and l.latitude is not null and l.longitude is not null
            """,
            chunk_ids,
        ).fetchall()
    return {
        row["chunk_id"]: Coordinates(row["latitude"], row["longitude"]) for row in rows
    }


def page_ids_in_scope(scope: GeoScope) -> frozenset[str] | None:
    """Pages whose primary location falls inside the scope; None if unfiltered."""
    level = scope.level
    if level == "none":
        return None
    if level == "radius":
        lat_min, lat_max, lon_min, lon_max = scope.bounding_box()
        condition = "l.latitude between ? and ? and l.longitude between ? and ?"
        params: tuple = (lat_min, lat_max, lon_min, lon_max)
    else:
        column = {"nuts3": "nuts3", "nuts2": "nuts2", "country": "country_code"}[level]
        condition = f"l.{column} = ?"
        params = (getattr(scope, column),)
    sql = (
        "select l.page_id from page_locations l "
        f"where l.role = 'primary' and {condition}"
    )
    if scope.include_null:
        sql += (
            " union select m.id from page_metadata m where m.id not in "
            "(select page_id from page_locations where role = 'primary')"
        )
    with connect_pages() as conn:
        return frozenset(row["page_id"] for row in conn.execute(sql, params))


def pages_near(
    coordinates: Coordinates, radius_km: float, limit: int = 10
) -> list[tuple[PageLocation, float]]:
    """Pages by distance from a point, nearest first; primaries only."""
    scored = []
    for location in primary_locations().values():
        if location.coordinates is None:
            continue
        distance = haversine_km(coordinates, location.coordinates)
        if distance <= radius_km:
            scored.append((location, distance))
    scored.sort(key=lambda item: item[1])
    return scored[:limit]


def region_code_column(code: str) -> str | None:
    """The `page_locations` column a NUTS or ISO code is matched against.

    NUTS-1 (three characters, e.g. SI0) is not stored; it is the prefix of the
    stored NUTS-2 code, so it is matched with `like`.
    """
    code = (code or "").strip().upper()
    return {2: "country_code", 3: "nuts2", 4: "nuts2", 5: "nuts3"}.get(len(code))


def pages_in_region(code: str, limit: int = 50) -> list[PageLocation]:
    """Pages whose primary location sits in a NUTS or ISO country code."""
    code = (code or "").strip().upper()
    column = region_code_column(code)
    if column is None:
        return []
    condition, value = (
        (f"{column} like ?", f"{code}%") if len(code) == 3 else (f"{column} = ?", code)
    )
    with connect_pages() as conn:
        rows = conn.execute(
            f"select {', '.join(_LOCATION_COLUMNS)} from page_locations "
            f"where role = 'primary' and {condition} order by name limit ?",
            (value, limit),
        ).fetchall()
    return [PageLocation(**dict(row)) for row in rows]


def cached_geo_scope(query_norm: str) -> str | None:
    with connect_pages() as conn:
        row = conn.execute(
            "select scope_json from geo_scope_cache where query_norm = ?",
            (query_norm,),
        ).fetchone()
    return row["scope_json"] if row else None


def store_geo_scope(query_norm: str, scope_json: str) -> None:
    with connect_pages() as conn:
        conn.execute(
            "insert or replace into geo_scope_cache (query_norm, scope_json, created_at)"
            " values (?, ?, ?)",
            (query_norm, scope_json, datetime.now(timezone.utc).isoformat()),
        )

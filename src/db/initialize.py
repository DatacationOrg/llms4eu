from pathlib import Path
import sqlite3

from src.shared.env import ROOT, load_local_env, load_yaml, sqlite_path
from src.shared.schema import Place


__all__ = ["initialize_db"]

CONFIG = load_yaml(Path(__file__).with_name("config.yaml"))


def initialize_db() -> None:
    load_local_env()
    schema = (ROOT / CONFIG["schema_path"]).read_text()
    places = _read_seed_places(ROOT / CONFIG["seed_path"])

    with _connect() as conn:
        conn.executescript(schema)
        _add_missing_columns(conn)
        _truncate_places(conn)
        _insert_places(conn, places)

    print(f"loaded {len(places)} places")


# ---------- PRIVATE FUNCTIONS ----------


def _add_missing_columns(conn: sqlite3.Connection) -> None:
    existing = {row[1] for row in conn.execute("pragma table_info(places)")}
    for column in ("latitude", "longitude"):
        if column not in existing:
            conn.execute(f"alter table places add column {column} real")


def _read_seed_places(path: Path) -> list[Place]:
    # Pydantic catches fixture/schema drift before anything reaches Postgres.
    return [
        Place.model_validate_json(line)
        for line in path.read_text().splitlines()
        if line.strip()
    ]


def _connect() -> sqlite3.Connection:
    return sqlite3.connect(sqlite_path())


def _truncate_places(conn: sqlite3.Connection) -> None:
    conn.execute("delete from places")


def _insert_places(conn: sqlite3.Connection, places: list[Place]) -> None:
    columns = Place.db_columns()
    placeholders = ", ".join(["?"] * len(columns))
    conn.executemany(
        f"insert into places ({', '.join(columns)}) values ({placeholders})",
        [place.db_values() for place in places],
    )


if __name__ == "__main__":
    initialize_db()

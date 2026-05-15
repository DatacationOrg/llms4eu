import sqlite3

from src.shared.env import ROOT, load_local_env, load_yaml, sqlite_path
from src.shared.schema import Place


DB_CONFIG = load_yaml(ROOT / "src" / "db" / "config.yaml")


def upsert_places(places: list[Place]) -> int:
    if not places:
        return 0

    load_local_env()
    schema = (ROOT / DB_CONFIG["schema_path"]).read_text()
    columns = Place.db_columns()
    placeholders = ", ".join(["?"] * len(columns))
    updates = ", ".join(
        f"{column} = excluded.{column}" for column in columns if column != "id"
    )

    with _connect() as conn:
        conn.executescript(schema)
        conn.executemany(
            f"insert into places ({', '.join(columns)}) values ({placeholders}) "
            f"on conflict (id) do update set {updates}",
            [place.db_values() for place in places],
        )
    return len(places)


def _connect() -> sqlite3.Connection:
    return sqlite3.connect(sqlite_path())

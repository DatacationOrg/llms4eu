import sqlite3

from src.shared.schema import Place
from src.shared.env import sqlite_path


__all__ = ["load_places", "load_places_by_id"]


def load_places() -> list[Place]:
    """Load all place rows in stable order for indexing and inspection."""
    with _connect() as conn:
        rows = conn.execute("select id, place_description, summary from places order by id").fetchall()
    return [Place.model_validate(dict(row)) for row in rows]


def load_places_by_id(ids: list[str]) -> dict[str, Place]:
    """Load selected place rows as a dictionary keyed by place id."""
    if not ids:
        return {}

    placeholders = ", ".join("?" for _ in ids)
    with _connect() as conn:
        rows = conn.execute(
            f"select id, place_description, summary from places where id in ({placeholders})",
            ids,
        ).fetchall()
    return {row["id"]: Place.model_validate(dict(row)) for row in rows}


def _connect() -> sqlite3.Connection:
    conn = sqlite3.connect(sqlite_path())
    conn.row_factory = sqlite3.Row
    return conn

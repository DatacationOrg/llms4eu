import sqlite3

from src.shared.schema import Place
from src.shared.env import sqlite_path


__all__ = ["load_places", "load_places_by_id", "update_place_location"]

_PLACE_COLUMNS = "id, place_description, summary, latitude, longitude"


def load_places() -> list[Place]:
    """Load all place rows in stable order for indexing and inspection."""
    with _connect() as conn:
        rows = conn.execute(
            f"select {_PLACE_COLUMNS} from places order by id"
        ).fetchall()
    return [Place.model_validate(dict(row)) for row in rows]


def load_places_by_id(ids: list[str]) -> dict[str, Place]:
    """Load selected place rows as a dictionary keyed by place id."""
    if not ids:
        return {}

    placeholders = ", ".join("?" for _ in ids)
    with _connect() as conn:
        rows = conn.execute(
            f"select {_PLACE_COLUMNS} from places where id in ({placeholders})",
            ids,
        ).fetchall()
    return {row["id"]: Place.model_validate(dict(row)) for row in rows}


def update_place_location(place_id: str, latitude: float, longitude: float) -> None:
    with _connect() as conn:
        conn.execute(
            "update places set latitude = ?, longitude = ? where id = ?",
            (latitude, longitude, place_id),
        )


def _connect() -> sqlite3.Connection:
    conn = sqlite3.connect(sqlite_path())
    conn.row_factory = sqlite3.Row
    return conn

from pathlib import Path

from src.shared.schema import Place


def test_seed_data_shape():
    rows = [
        Place.model_validate_json(line)
        for line in Path("data/places.jsonl").read_text().splitlines()
        if line.strip()
    ]

    assert len(rows) == 10
    assert len({row.id for row in rows}) == 10
    for row in rows:
        assert row.id
        assert row.place_description
        assert row.summary


def test_schema_mentions_place_fields():
    schema = Path("sql/init.sql").read_text()

    for column in Place.db_columns():
        assert column in schema

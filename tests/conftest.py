import os
import tempfile

# Before anything imports src: every module's dataset and artifact root is a scratch
# folder for the whole session, so no test can write to the real data under /data.
os.environ["DATASET_DIR"] = tempfile.mkdtemp(prefix="dataset-")
os.environ["LLMS4EU_DATA"] = tempfile.mkdtemp(prefix="llms4eu-data-")

import sqlite3

import pytest

from src.shared.env import ROOT


@pytest.fixture
def page_db(monkeypatch, tmp_path):
    """Empty page database at the canonical artifact-store path, real sql/ schema."""
    monkeypatch.setenv("LLMS4EU_DATA", str(tmp_path))
    path = tmp_path / "db" / "pages.db"
    path.parent.mkdir(parents=True, exist_ok=True)
    with sqlite3.connect(path) as conn:
        for schema in ("raw_pages.sql", "eval.sql"):
            conn.executescript((ROOT / "sql" / schema).read_text(encoding="utf-8"))
    return path

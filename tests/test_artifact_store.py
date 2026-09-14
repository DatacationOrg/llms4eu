from pathlib import Path

from src.shared.env import chroma_path, data_path, okf_bundle, pages_db


def test_data_path_defaults_to_the_shared_store(monkeypatch):
    monkeypatch.delenv("LLMS4EU_DATA", raising=False)

    assert data_path("db", "pages.db") == Path("/data/llms4eu/db/pages.db")


def test_data_path_honours_the_env_override(monkeypatch, tmp_path):
    monkeypatch.setenv("LLMS4EU_DATA", str(tmp_path))

    assert data_path("checkpoints", "run.json") == tmp_path / "checkpoints/run.json"


def test_data_path_creates_the_parent_directory(monkeypatch, tmp_path):
    monkeypatch.setenv("LLMS4EU_DATA", str(tmp_path))

    path = data_path("answers", "nested", "answer.json")

    assert path.parent.is_dir()
    assert not path.exists()


def test_named_artifacts_live_in_the_store(monkeypatch, tmp_path):
    monkeypatch.setenv("LLMS4EU_DATA", str(tmp_path))

    assert pages_db() == tmp_path / "db/pages.db"
    assert okf_bundle() == tmp_path / "okf/tourism"
    assert chroma_path() == tmp_path / "chroma"
    assert chroma_path().is_dir()

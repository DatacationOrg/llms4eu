import os
from pathlib import Path

from src.shared.env import DEFAULT_DATA_ROOT, data_path


def test_data_path_defaults_to_the_shared_store(monkeypatch):
    """Assert the default without calling data_path, which would mkdir under /data."""
    monkeypatch.delenv("LLMS4EU_DATA", raising=False)

    assert Path(DEFAULT_DATA_ROOT) == Path("/data/llms4eu")
    assert os.getenv("LLMS4EU_DATA") is None


def test_data_path_honours_the_env_override(monkeypatch, tmp_path):
    monkeypatch.setenv("LLMS4EU_DATA", str(tmp_path))

    assert data_path("checkpoints", "run.json") == tmp_path / "checkpoints/run.json"


def test_data_path_creates_the_parent_directory(monkeypatch, tmp_path):
    monkeypatch.setenv("LLMS4EU_DATA", str(tmp_path))

    path = data_path("answers", "nested", "answer.json")

    assert path.parent.is_dir()
    assert not path.exists()

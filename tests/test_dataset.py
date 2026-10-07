import pyarrow.parquet as pq
import pytest

from src.db import dataset
from src.db.wiki_chunks import chunk_page
from src.db.wiki_notes import cap

MODELS = [
    dataset.Page,
    dataset.Chunk,
    dataset.Rag,
    dataset.Unanswerable,
    dataset.Compare,
    dataset.Meta,
    dataset.Tables,
]


@pytest.mark.parametrize("model", MODELS, ids=lambda m: m.file)
def test_file_matches_its_model(model):
    if not dataset.path(model).exists():
        pytest.skip("wiki data not on this machine")
    assert set(pq.read_schema(dataset.path(model)).names) == set(model.model_fields)
    assert next(dataset.read(model))


def test_chunks_keep_breadcrumbs_and_sizes():
    long = " ".join(["Die Burg wurde mehrfach umgebaut."] * 100)
    text = f"# Burg\n\nIntro.\n\n## Geschichte\n\nAlt.\n\n### Neuzeit\n\nNeu.\n\n## Lage\n\n{long}"
    chunks = chunk_page({"id": "dewiki/Q1", "title": "Burg", "text": text})
    small = [c for c in chunks if c["size"] == 256]
    assert all(c["tokens"] <= 256 for c in chunks if c["size"] == 256)
    assert small[0]["breadcrumb"] is None and "## Geschichte" in small[0]["text"]
    assert {c["breadcrumb"] for c in small[1:]} == {"Lage"}
    assert [c["id"] for c in small[:2]] == ["dewiki/Q1:256:0", "dewiki/Q1:256:1"]
    assert len([c for c in chunks if c["size"] == 2048]) == 1


def test_cap_keeps_whole_sentences_under_the_limit():
    assert cap("One. Two. Three. Four.", 3) == "One. Two. Three."
    assert cap("A" * 100 + ". " + "B" * 250 + ".", 3) == "A" * 100 + "."
    assert len(cap("word " * 100, 1)) <= 300

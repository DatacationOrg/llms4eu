import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

from src.db.dataset import DEFAULT_SIZE, SIZES, Chunk, load, vectors_path
from src.preprocess import chunker
from src.preprocess.chunker import chunk_page

LONG = " ".join(["Die Burg wurde mehrfach umgebaut."] * 100)
PAGE = {
    "id": "dewiki/Q1",
    "title": "Burg",
    "text": "# Burg\n\nIntro sentence about the castle and where it stands today.\n\n"
    "## Geschichte\n\n[Bearbeiten | Quelltext bearbeiten]Alt.\n\n### Neuzeit\n\nNeu.\n\n"
    "## Weblinks\n\n[Bearbeiten | Quelltext bearbeiten]\n\n"
    f"## Lage\n\n{LONG}",
}


def test_chunks_keep_breadcrumbs_and_sizes_without_edit_links():
    chunks = chunk_page(PAGE)
    small = [c for c in chunks if c["size"] == min(SIZES)]
    assert all(c["tokens"] <= min(SIZES) + chunker.MIN_TOKENS for c in small)
    assert small[0]["breadcrumb"] is None and "## Geschichte" in small[0]["text"]
    assert {c["breadcrumb"] for c in small[1:]} == {"Lage"}
    assert [c["n"] for c in small[:2]] == [0, 1]
    assert all(c["text"].endswith(".") for c in small)  # cut at sentence ends
    assert not any("Bearbeiten" in c["text"] or "Weblinks" in c["text"] for c in chunks)
    assert len([c for c in chunks if c["size"] == max(SIZES)]) == 1


def test_a_tiny_tail_joins_its_neighbour():
    page = dict(PAGE, text=f"# Burg\n\n## Lage\n\n{LONG}\n\n## Ende\n\nKurz.")
    last = [c for c in chunk_page(page) if c["size"] == min(SIZES)][-1]
    assert last["text"].endswith("Kurz.") and last["tokens"] > chunker.MIN_TOKENS


def test_rechunk_keeps_vectors_and_roles_of_unchanged_chunks(tiny_dataset, monkeypatch):
    monkeypatch.setattr(chunker, "SIZES", (DEFAULT_SIZE,))
    old = load(Chunk).to_pylist()
    old[1]["role"] = "Its trail."
    pq.write_table(pa.Table.from_pylist(old), tiny_dataset / "chunks.parquet")
    new = [dict(old[1], role=None),  # same text, now first: kept
           dict(old[0], text="Rewritten castle chunk.", role=None)]  # fmt: skip
    for written, target in chunker._carry_over(new):
        written.replace(target)
    vectors = np.load(vectors_path("qwen", DEFAULT_SIZE))
    assert vectors[0].tolist() == [0.0, 1.0]  # the forest chunk's vector moved along
    assert np.isnan(vectors[1]).all()  # changed: to be re-embedded
    assert [c["role"] for c in new] == ["Its trail.", None]

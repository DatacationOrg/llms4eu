# pyright: reportAttributeAccessIssue=false
# (pyarrow.compute functions are generated at import, unknown to the type checker)
"""The real dataset's contract, read only: every file in place, columns as its model
says, rows that validate, and ids and evidence that fit together."""

from functools import cache
from pathlib import Path

import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq
import pytest

from src.db import dataset

REAL = Path("/data/llms4eu/wiki")  # tests otherwise run on a scratch DATASET_DIR
MODELS = [
    dataset.Page,
    dataset.Rag,
    dataset.Unanswerable,
    dataset.Compare,
    dataset.Meta,
    dataset.Tables,
]
pytestmark = pytest.mark.skipif(not REAL.exists(), reason="wiki data not here")


def read(model, columns=None):
    return pq.read_table(REAL / model.file, columns=columns)


def longest(column) -> int:
    """Characters in the longest value of a text column (0 while it is all null)."""
    return pc.max(pc.utf8_length(column.cast(pa.string()))).as_py() or 0


@cache
def page_ids() -> set[str]:
    return set(read(dataset.Page, ["id"]).column("id").to_pylist())


def test_tests_never_see_the_real_data():
    assert dataset.ROOT != REAL


@pytest.mark.parametrize("model", MODELS, ids=lambda m: m.file)
def test_file_has_its_models_columns_and_rows_validate(model):
    file = pq.ParquetFile(REAL / model.file)
    assert set(file.schema_arrow.names) == set(model.model_fields)
    for row in next(file.iter_batches(batch_size=200)).to_pylist():
        model.model_validate(row)


def test_pages_are_unique_and_have_text():
    pages = read(dataset.Page, ["id", "text", "summary"])
    assert len(page_ids()) == pages.num_rows
    assert pc.min(pc.utf8_length(pages.column("text"))).as_py() > 0
    assert longest(pages.column("summary")) <= 300


def test_questions_point_at_existing_pages():
    assert set(read(dataset.Rag, ["id"]).column("id").to_pylist()) <= page_ids()
    assert (
        set(read(dataset.Unanswerable, ["id"]).column("id").to_pylist()) <= page_ids()
    )
    for model, column in ((dataset.Compare, "pages"), (dataset.Meta, "gold_pages")):
        pages = pc.list_flatten(read(model, [column]).column(column))
        assert set(pages.to_pylist()) <= page_ids()


@pytest.mark.xfail(
    strict=True,
    reason="QA data: about 1 in 4 verbatim spans miss their quote (footnote markers "
    "like [7] dropped from the quote, or the quote not in the page as is)",
)
def test_evidence_spans_point_at_the_quotes():
    rows = read(dataset.Rag, ["id", "evidence", "spans", "verbatim"]).slice(0, 2000)
    rows = [r for r in rows.to_pylist() if r["verbatim"]]
    pages = pq.read_table(
        REAL / dataset.Page.file,
        columns=["id", "text"],
        filters=[("id", "in", list({r["id"] for r in rows}))],
    )
    texts = dict(zip(pages.column("id").to_pylist(), pages.column("text").to_pylist()))
    for row in rows:
        for quote, (start, end) in zip(row["evidence"], row["spans"]):
            assert texts[row["id"]][start:end] == quote

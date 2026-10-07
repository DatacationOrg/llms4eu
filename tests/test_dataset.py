import pyarrow.parquet as pq
import pytest

from src.db import dataset

MODELS = [
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

import pyarrow.parquet as pq
import pytest

from src.db import wiki_qa

MODELS = [
    wiki_qa.Rag,
    wiki_qa.Unanswerable,
    wiki_qa.Compare,
    wiki_qa.Meta,
    wiki_qa.Tables,
]


@pytest.mark.parametrize("model", MODELS, ids=lambda m: m.file)
def test_file_matches_its_model(model):
    if not wiki_qa.path(model).exists():
        pytest.skip("wiki QA data not on this machine")
    assert set(pq.read_schema(wiki_qa.path(model)).names) == set(model.model_fields)
    assert next(wiki_qa.read(model))

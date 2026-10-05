import json

import pyarrow.parquet as pq
import pytest

from src.data_prep import wiki_qa


@pytest.mark.skipif(
    not wiki_qa.ROOT.exists(), reason="wiki QA release not on this machine"
)
def test_release_has_every_file_with_its_rows_and_columns():
    manifest = json.loads((wiki_qa.ROOT / "manifest.json").read_text())
    for name in wiki_qa.JSON_COLUMNS:
        meta = pq.read_metadata(wiki_qa.ROOT / f"wiki_qa_{name}.parquet")
        if f"wiki_qa_{name}.parquet" in manifest:
            assert meta.num_rows == manifest[f"wiki_qa_{name}.parquet"]["rows"]
        assert wiki_qa.JSON_COLUMNS[name] <= set(meta.schema.names)
    rag = set(pq.read_schema(wiki_qa.ROOT / "wiki_qa_rag.parquet").names)
    assert {"id", "question", "answer", "split", "answer_ok", "hits", "relevant"} <= rag

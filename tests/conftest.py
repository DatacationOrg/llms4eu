import os
import shutil
import tempfile
from pathlib import Path

# Before anything imports src: every module's dataset and artifact root is a scratch
# folder for the whole session, so no test can write to the real data under /data.
os.environ["DATASET_DIR"] = tempfile.mkdtemp(prefix="dataset-")
os.environ["LLMS4EU_DATA"] = tempfile.mkdtemp(prefix="llms4eu-data-")

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from src.db.dataset import DEFAULT_SIZE as SIZE

ROOT = Path(os.environ["DATASET_DIR"])


@pytest.fixture
def tiny_dataset(monkeypatch):
    """Two pages, one chunk each at DEFAULT_SIZE, and `qwen` vectors, in the scratch dir."""
    shutil.rmtree(ROOT)
    ROOT.mkdir()
    chunks = [
        {"id": f"enwiki/Q1:{SIZE}:0", "page_id": "enwiki/Q1", "size": SIZE, "n": 0,
         "title": "Castle Page", "breadcrumb": "History",
         "text": "Canonical castle chunk.", "tokens": 4, "role": None},
        {"id": f"enwiki/Q2:{SIZE}:0", "page_id": "enwiki/Q2", "size": SIZE, "n": 0,
         "title": "Forest Page", "breadcrumb": "Trail",
         "text": "Canonical forest chunk.", "tokens": 4, "role": None},
    ]  # fmt: skip
    pq.write_table(pa.Table.from_pylist(chunks), ROOT / "chunks.parquet")
    (ROOT / "embeddings" / "qwen").mkdir(parents=True)
    np.save(ROOT / f"embeddings/qwen/{SIZE}.npy", np.eye(2, dtype=np.float16))
    return ROOT

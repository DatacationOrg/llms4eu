import json
from pathlib import Path

from src.scraping.page_extract import FetchResult


def append_fetch_result(result: FetchResult, out: Path) -> None:
    """One JSON line per fetched page: its metadata and `text`, the Markdown (or "")."""
    row = result.metadata.model_dump(mode="json")
    row["text"] = result.markdown_content.markdown if result.markdown_content else ""
    with out.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(row, ensure_ascii=False) + "\n")

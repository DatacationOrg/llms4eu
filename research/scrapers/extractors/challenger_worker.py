"""HTML -> markdown for crawl4ai, markitdown and docling, run OUT OF PROCESS.

Why out of process: `crawl4ai` pulls 97 packages (Playwright, nltk, tiktoken, openai)
and `docling` pulls a document-AI stack. Adding either to `arena-env` would make the
shared environment un-resolvable for the other nine entrants. This module therefore
imports nothing from `research.scrapers` -- it is executed by
`uv run --no-project --with <tool>` and talks JSON over files.

All three read the SAME saved `raw.html` the other twelve entrants read. That is the
whole point: `crawl4ai` is a fetcher *and* an extractor, and letting it fetch its own
copy would confound the two layers -- worth 44% of pages, per the fetch-layer result.
So only its markdown generator is used, never its crawler.

Usage:  python challenger_worker.py <tool> <job.json> <out.json>
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path


def convert_markitdown(html: str):
    import io

    from markitdown import MarkItDown

    result = MarkItDown().convert_stream(
        io.BytesIO(html.encode("utf-8")), file_extension=".html"
    )
    return result.text_content, "markitdown"


def convert_crawl4ai(html: str):
    """Content-filtered markdown, which is crawl4ai's actual extraction product.

    `raw_markdown` is a straight tag-to-markdown transcription that keeps every nav
    and footer -- scoring that would test crawl4ai's converter while ignoring the
    extractor it ships. `PruningContentFilter` is the documented way to get content,
    so `fit_markdown` is what competes. It can legitimately come back empty on pages
    the filter prunes to nothing; that falls back to `raw_markdown` and is recorded,
    rather than silently scoring an empty string.
    """
    from crawl4ai.content_filter_strategy import PruningContentFilter
    from crawl4ai.markdown_generation_strategy import DefaultMarkdownGenerator

    generator = DefaultMarkdownGenerator(content_filter=PruningContentFilter())
    result = generator.generate_markdown(input_html=html, base_url="")
    fit = (result.fit_markdown or "").strip()
    if fit:
        return fit, "crawl4ai:fit"
    return (result.raw_markdown or "").strip(), "crawl4ai:raw-fallback"


def convert_docling(html: str):
    import io

    from docling.backend.html_backend import HTMLDocumentBackend
    from docling.datamodel.base_models import InputFormat
    from docling.datamodel.document import InputDocument

    data = html.encode("utf-8")
    in_doc = InputDocument(
        path_or_stream=io.BytesIO(data),
        format=InputFormat.HTML,
        backend=HTMLDocumentBackend,
        filename="page.html",
    )
    backend = HTMLDocumentBackend(in_doc=in_doc, path_or_stream=io.BytesIO(data))
    return backend.convert().export_to_markdown(), "docling"


CONVERTERS = {
    "markitdown": convert_markitdown,
    "crawl4ai": convert_crawl4ai,
    "docling": convert_docling,
}


def main() -> None:
    tool, job_path, out_path = sys.argv[1], sys.argv[2], sys.argv[3]
    convert = CONVERTERS[tool]
    jobs = json.loads(Path(job_path).read_text(encoding="utf-8"))

    results: dict[str, dict] = {}
    for index, job in enumerate(jobs, 1):
        html = Path(job["path"]).read_text(encoding="utf-8", errors="replace")
        started = time.perf_counter()
        try:
            text, variant = convert(html)
            results[job["id"]] = {
                "text": text or "",
                "ms": (time.perf_counter() - started) * 1000,
                "mode": variant,
                "error": None,
            }
        except Exception as exc:  # one bad page must not lose the batch
            results[job["id"]] = {
                "text": "",
                "ms": (time.perf_counter() - started) * 1000,
                "mode": None,
                "error": f"{type(exc).__name__}: {exc}"[:400],
            }
        print(f"{tool} {index}/{len(jobs)}", file=sys.stderr, flush=True)

    Path(out_path).write_text(json.dumps(results), encoding="utf-8")


if __name__ == "__main__":
    main()

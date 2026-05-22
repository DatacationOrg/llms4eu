from __future__ import annotations

import tempfile
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from urllib.parse import urljoin, urlparse

import httpx
from bs4 import BeautifulSoup
from docling.datamodel.base_models import ConversionStatus, InputFormat
from docling.datamodel.pipeline_options import PdfPipelineOptions
from docling.document_converter import DocumentConverter, PdfFormatOption

from src.scraping.extract_markdown import clean_markdown
from src.scraping.scraper import HEADERS
from src.scraping.settings import fetch_pages_config


config = fetch_pages_config()
DOCUMENT_EXTRACTOR_NAME = "docling"


@dataclass(frozen=True)
class DocumentLink:
    url: str
    label: str


@dataclass(frozen=True)
class DocumentMarkdownResult:
    markdown: str
    url: str
    bytes_read: int
    error: str | None = None


def extract_first_document_markdown(
    html: str,
    page_url: str,
    timeout: float,
    max_bytes: int | None = None,
    max_pages: int | None = None,
) -> DocumentMarkdownResult | None:
    max_bytes = max_bytes or config.max_document_bytes
    max_pages = max_pages or config.max_document_pages
    for link in find_document_links(html, page_url):
        result = document_url_to_markdown(
            link.url,
            timeout=timeout,
            max_bytes=max_bytes,
            max_pages=max_pages,
        )
        if result.markdown.strip():
            title = link.label.strip()
            prefix = f"# {title}\n\n" if title else ""
            return DocumentMarkdownResult(
                markdown=prefix + result.markdown,
                url=result.url,
                bytes_read=result.bytes_read,
                error=result.error,
            )
        if result.error:
            return result
    return None


def find_document_links(html: str, page_url: str) -> list[DocumentLink]:
    soup = BeautifulSoup(html, "lxml")
    links: list[DocumentLink] = []
    for anchor in soup.select("main a[href], article a[href], [role=main] a[href]"):
        href = anchor.get("href")
        if not href:
            continue
        absolute_url = urljoin(page_url, href)
        if not _looks_like_document_url(absolute_url):
            continue
        label = " ".join(anchor.get_text(" ", strip=True).split())
        links.append(DocumentLink(url=absolute_url, label=label))
    return _deduplicate_links(links)


def document_url_to_markdown(
    url: str,
    timeout: float,
    max_bytes: int | None = None,
    max_pages: int | None = None,
) -> DocumentMarkdownResult:
    max_bytes = max_bytes or config.max_document_bytes
    max_pages = max_pages or config.max_document_pages
    try:
        content = _download_document(url, timeout=timeout, max_bytes=max_bytes)
    except Exception as exc:
        return DocumentMarkdownResult(
            markdown="",
            url=url,
            bytes_read=0,
            error=f"document download failed: {type(exc).__name__}: {exc}",
        )

    suffix = Path(urlparse(url).path).suffix.lower() or ".pdf"
    try:
        with tempfile.NamedTemporaryFile(suffix=suffix) as tmp:
            tmp.write(content)
            tmp.flush()
            converter = _document_converter()
            result = converter.convert(
                tmp.name,
                max_file_size=max_bytes,
                max_num_pages=max_pages,
            )
    except Exception as exc:
        return DocumentMarkdownResult(
            markdown="",
            url=url,
            bytes_read=len(content),
            error=f"document conversion failed: {type(exc).__name__}: {exc}",
        )

    if result.status not in {
        ConversionStatus.SUCCESS,
        ConversionStatus.PARTIAL_SUCCESS,
    }:
        return DocumentMarkdownResult(
            markdown="",
            url=url,
            bytes_read=len(content),
            error=f"document conversion status: {result.status}",
        )

    markdown = clean_markdown(result.document.export_to_markdown())
    return DocumentMarkdownResult(markdown=markdown, url=url, bytes_read=len(content))


def _download_document(url: str, timeout: float, max_bytes: int) -> bytes:
    with httpx.Client(
        headers=HEADERS, follow_redirects=True, timeout=timeout
    ) as client:
        with client.stream("GET", url) as response:
            response.raise_for_status()
            expected_size = response.headers.get("content-length")
            if expected_size and int(expected_size) > max_bytes:
                raise ValueError(f"document exceeds {max_bytes} bytes")

            chunks: list[bytes] = []
            total = 0
            for chunk in response.iter_bytes():
                total += len(chunk)
                if total > max_bytes:
                    raise ValueError(f"document exceeds {max_bytes} bytes")
                chunks.append(chunk)
    return b"".join(chunks)


@lru_cache(maxsize=1)
def _document_converter() -> DocumentConverter:
    pdf_options = PdfPipelineOptions()
    pdf_options.do_ocr = False
    pdf_options.do_table_structure = True
    return DocumentConverter(
        allowed_formats=[
            InputFormat.PDF,
            InputFormat.DOCX,
            InputFormat.PPTX,
            InputFormat.XLSX,
            InputFormat.HTML,
            InputFormat.MD,
            InputFormat.CSV,
        ],
        format_options={
            InputFormat.PDF: PdfFormatOption(pipeline_options=pdf_options),
        },
    )


def _looks_like_document_url(url: str) -> bool:
    path = urlparse(url).path.lower()
    return any(path.endswith(extension) for extension in config.document_extensions)


def _deduplicate_links(links: list[DocumentLink]) -> list[DocumentLink]:
    seen: set[str] = set()
    unique_links: list[DocumentLink] = []
    for link in links:
        if link.url in seen:
            continue
        seen.add(link.url)
        unique_links.append(link)
    return unique_links

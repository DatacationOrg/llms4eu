import hashlib
from dataclasses import dataclass
from datetime import datetime, timezone

from src.scraping.document_markdown import (
    DOCUMENT_EXTRACTOR_NAME,
    extract_first_document_markdown,
)
from src.scraping.extract_markdown import (
    EXTRACTOR_NAME,
    extract_markdown,
    markdown_looks_like_contact_footer,
    markdown_needs_browser_render,
)
from src.scraping.page_fetch import FetchedPage, extract_language
from src.scraping.render import RenderedPage, render_html
from src.scraping.settings import fetch_pages_config
from src.scraping.structured_markdown import (
    LISTING_EXTRACTOR_NAME,
    PageKind,
    classify_page,
    extract_listing_markdown,
)
from src.shared.schema import PageMarkdownContent, PageMetadata


config = fetch_pages_config()


@dataclass(frozen=True)
class FetchResult:
    metadata: PageMetadata
    markdown_content: PageMarkdownContent | None = None


def extract_page(fetched_page: FetchedPage, timeout: float) -> FetchResult:
    if fetched_page.html is None:
        return FetchResult(metadata=_empty_metadata(fetched_page.metadata))
    return _extract_or_render(fetched_page.metadata, fetched_page.html, timeout)


def _extract_or_render(
    metadata: PageMetadata, html: str, timeout: float
) -> FetchResult:
    markdown = extract_markdown(html, url=metadata.final_url or metadata.url)
    result = _extract_from_html(metadata, html, markdown, timeout)
    if result:
        return result

    try:
        rendered = render_html(
            metadata.final_url or metadata.url,
            # --timeout can ask for more than the configured browser budget.
            max(int(timeout * 1000), config.browser_timeout_ms),
        )
    except Exception as exc:
        return FetchResult(
            metadata=_empty_metadata(
                metadata,
                error=f"browser render failed: {type(exc).__name__}: {exc}",
            )
        )

    rendered_metadata = _rendered_metadata(metadata, rendered)
    rendered_markdown = extract_markdown(
        rendered.html, url=rendered_metadata.final_url or rendered_metadata.url
    )
    result = _extract_from_html(
        rendered_metadata, rendered.html, rendered_markdown, timeout
    )
    if result:
        return result

    return FetchResult(
        metadata=_empty_metadata(
            rendered_metadata,
            error="Markdown extraction failed after browser render",
            markdown_chars=len(rendered_markdown),
        )
    )


def _extract_from_html(
    metadata: PageMetadata, html: str, markdown: str, timeout: float
) -> FetchResult | None:
    metadata = metadata.model_copy(update={"language": extract_language(html)})
    if markdown_looks_like_contact_footer(markdown):
        document_result = _document_result_from_html(metadata, html, timeout)
        if document_result:
            return document_result

    page_kind = classify_page(html, metadata.final_url or metadata.url, markdown)
    if page_kind == PageKind.LISTING:
        listing_markdown = extract_listing_markdown(
            html, metadata.final_url or metadata.url, metadata.title
        )
        if listing_markdown:
            return _markdown_result(
                metadata,
                listing_markdown,
                extractor=LISTING_EXTRACTOR_NAME,
                page_kind=PageKind.LISTING,
            )

    if not markdown_needs_browser_render(markdown):
        return _markdown_result(metadata, markdown, page_kind=PageKind.PROSE)

    return _document_result_from_html(metadata, html, timeout)


def _rendered_metadata(metadata: PageMetadata, rendered: RenderedPage) -> PageMetadata:
    return metadata.model_copy(
        update={
            "final_url": rendered.final_url,
            "fetched_at": datetime.now(timezone.utc),
            "title": rendered.title or metadata.title,
            "raw_bytes": len(rendered.html.encode("utf-8")),
            "fetch_method": "playwright",
            "error": None,
        }
    )


def _empty_metadata(
    metadata: PageMetadata, error: str | None = None, markdown_chars: int = 0
) -> PageMetadata:
    update = {
        "content_hash": None,
        "page_kind": PageKind.EMPTY.value,
        "markdown_chars": markdown_chars,
    }
    if error is not None:
        update["error"] = error
    return metadata.model_copy(update=update)


def _document_result_from_html(
    metadata: PageMetadata,
    html: str,
    timeout: float,
) -> FetchResult | None:
    document_result = extract_first_document_markdown(
        html,
        page_url=metadata.final_url or metadata.url,
        timeout=timeout,
    )
    if not document_result:
        return None
    if not document_result.markdown.strip():
        return None

    document_metadata = metadata.model_copy(
        update={
            "final_url": document_result.url,
            "fetched_at": datetime.now(timezone.utc),
            "raw_bytes": document_result.bytes_read,
            "fetch_method": f"{metadata.fetch_method}+document",
            "error": None,
        }
    )
    return _markdown_result(
        document_metadata,
        document_result.markdown,
        extractor=DOCUMENT_EXTRACTOR_NAME,
        page_kind=PageKind.DOCUMENT,
    )


def _markdown_result(
    metadata: PageMetadata,
    markdown: str,
    extractor: str = EXTRACTOR_NAME,
    page_kind: PageKind = PageKind.PROSE,
) -> FetchResult:
    content = markdown.encode("utf-8")
    metadata = metadata.model_copy(
        update={
            "content_hash": hashlib.sha256(content).hexdigest(),
            "extractor": extractor,
            "page_kind": page_kind.value,
            "markdown_chars": len(markdown),
            "error": None,
        }
    )
    return FetchResult(
        metadata=metadata,
        markdown_content=PageMarkdownContent(page_id=metadata.id, markdown=markdown),
    )

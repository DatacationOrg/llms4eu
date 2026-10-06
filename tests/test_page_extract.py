from datetime import UTC, datetime

from src.scraping.extract_markdown import extract_markdown
from src.scraping.page_extract import _extract_from_html
from src.scraping.schema import PageMetadata

ARTICLE = """
<html><body><article>
<h1>Rajhenburg Castle</h1>
<p>The castle stands above Brestanica and was first mentioned in 1131.
It later passed to the Trappist monks, who produced chocolate there.</p>
<h2>Opening hours</h2>
<p>Open Tuesday to Sunday from 10:00 to 18:00, closed on Mondays.</p>
</article></body></html>
"""


def _metadata() -> PageMetadata:
    return PageMetadata(
        id="p1",
        source="castle",
        url="https://example.test/castle",
        fetched_at=datetime.now(UTC),
    )


def test_ordinary_article_is_classified_as_prose():
    """Branch order here decides page_kind; a misfire mis-tags the whole corpus."""
    metadata = _metadata()
    markdown = extract_markdown(ARTICLE, url=metadata.url)

    result = _extract_from_html(metadata, ARTICLE, markdown, timeout=5.0)

    assert result is not None, "well-formed prose must not fall through to render"
    assert result.metadata.page_kind == "prose"
    assert result.markdown_content is not None
    assert "Brestanica" in result.markdown_content.markdown


def test_nav_only_page_is_rejected_rather_than_stored():
    """Junk markdown must fall through to the render fallback, not be kept."""
    chrome = "<html><body><nav>[Home](#) [About](#)</nav></body></html>"
    markdown = extract_markdown(chrome, url="https://example.test/nav")

    result = _extract_from_html(_metadata(), chrome, markdown, timeout=5.0)

    assert result is None


def test_extracted_page_records_the_language_the_site_declares():
    """The cross-language question must exclude the page's real language, not a default."""
    html = """
    <html lang="hu-HU"><head><title>Vár</title></head><body><main>
    <p>A vár 895-ben epult, a Sava folyo felett all, es ma is latogathato.</p>
    <p>A kiallitas a helyi tortenelmet mutatja be reszletes leirasokkal.</p>
    </main></body></html>
    """

    metadata = _metadata()
    markdown = extract_markdown(html, url=metadata.url)

    result = _extract_from_html(metadata, html, markdown, timeout=5.0)

    assert result is not None
    assert result.metadata.language == "hu"

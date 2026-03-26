from dataclasses import dataclass, field
from pathlib import Path
from urllib.parse import urljoin, urlparse

import httpx
from bs4 import BeautifulSoup, Tag
from fpdf import FPDF
from loguru import logger

URLS = [
    "https://www.holland.com",
    "https://www.toeristeninformatienederland.nl",
    "https://www.visitbrabant.com",
    "https://www.friesland.nl",
]

MAX_PAGES_PER_SITE = 50
MIN_PARAGRAPH_LENGTH = 40

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/131.0.0.0 Safari/537.36"
    ),
    "Accept-Language": "en-US,en;q=0.9,nl;q=0.8",
}

SKIP_EXTENSIONS = {
    ".pdf",
    ".jpg",
    ".jpeg",
    ".png",
    ".gif",
    ".svg",
    ".webp",
    ".css",
    ".js",
    ".zip",
    ".mp4",
    ".mp3",
    ".ico",
    ".xml",
}


@dataclass
class Heading:
    level: int
    text: str


@dataclass
class PageContent:
    url: str
    title: str
    meta_description: str
    # headings: list[Heading]
    paragraphs: list[str]
    internal_links: list[str]


@dataclass
class SiteResult:
    site_url: str
    pages: list[PageContent] = field(default_factory=list)
    error: str = ""


def _is_same_domain(url: str, base_domain: str) -> bool:
    """Check if a URL belongs to the same domain."""
    parsed = urlparse(url)
    return parsed.netloc == base_domain or parsed.netloc == ""


def _normalize_url(href: str, base_url: str) -> str | None:
    """Resolve a relative/absolute href into a full URL, filtering out non-page links."""
    if not href or href.startswith(("#", "mailto:", "tel:", "javascript:")):
        return None

    full = urljoin(base_url, href)
    parsed = urlparse(full)

    # Only http(s)
    if parsed.scheme not in ("http", "https"):
        return None

    # Skip file downloads / static assets
    path_lower = parsed.path.lower()
    if any(path_lower.endswith(ext) for ext in SKIP_EXTENSIONS):
        return None

    # Strip fragment
    return parsed._replace(fragment="").geturl()


def extract_page_content(html: str, url: str) -> PageContent:
    """Extract text content, metadata, and internal links from an HTML page."""
    soup = BeautifulSoup(html, "lxml")

    title = soup.title.string.strip() if soup.title and soup.title.string else ""

    meta_description = ""
    meta_tag = soup.find("meta", attrs={"name": "description"})
    if meta_tag and meta_tag.get("content"):
        meta_description = str(meta_tag["content"]).strip()

    # Collect all links before decomposing elements
    raw_links: list[str] = []
    base_domain = urlparse(url).netloc
    for a in soup.find_all("a", href=True):
        normalized = _normalize_url(str(a["href"]), url)
        if normalized and _is_same_domain(normalized, base_domain):
            raw_links.append(normalized)

    # Remove script/style/boilerplate elements before extracting text
    for element in soup(["script", "style", "nav", "footer", "header", "aside"]):
        element.decompose()

    # Remove newsletter sections and filter/result widgets
    to_remove: list[Tag] = []
    for section in soup.find_all("section"):
        css_classes = list(section.get_attribute_list("class"))
        if any(
            kw in str(cls)
            for cls in css_classes
            for kw in ("Newsletter", "newsletter", "filterbox")
        ):
            to_remove.append(section)
    for div in soup.find_all("div"):
        css_classes = list(div.get_attribute_list("class"))
        if any("filterbox" in str(cls) for cls in css_classes):
            to_remove.append(div)
    for el in to_remove:
        el.decompose()

    # Prefer <main> or <article> content to avoid site-wide boilerplate
    content_root = soup.find("main") or soup.find("article") or soup

    paragraphs = [
        p.get_text(strip=True)
        for p in content_root.find_all("p")
        if len(p.get_text(strip=True)) >= MIN_PARAGRAPH_LENGTH
    ]

    # headings: list[Heading] = []
    # for level in range(1, 4):
    #     for h in content_root.find_all(f"h{level}"):
    #         text = h.get_text(strip=True)
    #         if text:
    #             headings.append(Heading(level=level, text=text))

    return PageContent(
        url=url,
        title=title,
        meta_description=meta_description,
        # headings=headings,
        paragraphs=paragraphs,
        internal_links=list(
            dict[str, None].fromkeys(raw_links)
        ),  # deduplicated, order preserved
    )


def remove_boilerplate(
    pages: list[PageContent], threshold: float = 0.1
) -> list[PageContent]:
    """Remove paragraphs and headings that appear on more than `threshold` fraction of pages."""
    if len(pages) < 2:
        return pages

    min_occurrences = max(2, int(len(pages) * threshold))

    # Count how many pages each paragraph / heading text appears on
    paragraph_counts: dict[str, int] = {}
    # heading_counts: dict[str, int] = {}

    for page in pages:
        for text in set(page.paragraphs):
            paragraph_counts[text] = paragraph_counts.get(text, 0) + 1
        # for h in {h.text for h in page.headings}:
        #     heading_counts[h] = heading_counts.get(h, 0) + 1

    boilerplate_paragraphs = {
        t for t, c in paragraph_counts.items() if c >= min_occurrences
    }
    # boilerplate_headings = {
    #     t for t, c in heading_counts.items() if c >= min_occurrences
    # }

    # if boilerplate_paragraphs or boilerplate_headings:
    #     logger.info(
    #         f"Removing boilerplate: {len(boilerplate_paragraphs)} paragraphs, "
    #         + f"{len(boilerplate_headings)} headings (threshold={threshold})"
    #     )

    cleaned: list[PageContent] = []
    for page in pages:
        cleaned.append(
            PageContent(
                url=page.url,
                title=page.title,
                meta_description=page.meta_description,
                # headings=[
                #     h for h in page.headings if h.text not in boilerplate_headings
                # ],
                paragraphs=[
                    p for p in page.paragraphs if p not in boilerplate_paragraphs
                ],
                internal_links=page.internal_links,
            )
        )
    return cleaned


def crawl_site(
    start_url: str, client: httpx.Client, max_pages: int = MAX_PAGES_PER_SITE
) -> list[PageContent]:
    """Crawl an entire website following internal links, up to max_pages."""
    visited: set[str] = set()
    to_visit: list[str] = [start_url]
    pages: list[PageContent] = []

    while to_visit and len(visited) < max_pages:
        url = to_visit.pop(0)
        if url in visited:
            continue
        visited.add(url)

        try:
            response = client.get(url)
            response.raise_for_status()

            content_type = response.headers.get("content-type", "")
            if "text/html" not in content_type:
                continue

            page = extract_page_content(response.text, url)
            pages.append(page)
            logger.info(
                f"  [{len(visited)}/{max_pages}] {url} — {len(page.paragraphs)} paragraphs"
            )

            # Queue new internal links
            for link in page.internal_links:
                if link not in visited:
                    to_visit.append(link)

        except httpx.HTTPStatusError as e:
            logger.warning(f"  HTTP {e.response.status_code}: {url}")
        except httpx.RequestError as e:
            logger.warning(f"  Failed: {url} ({e})")

    return pages


def scrape_urls(
    urls: list[str] | None = None, max_pages: int = MAX_PAGES_PER_SITE
) -> list[SiteResult]:
    """Crawl each website fully (following internal links) and return results grouped by site."""
    urls = urls or URLS
    results: list[SiteResult] = []

    with httpx.Client(headers=HEADERS, follow_redirects=True, timeout=30.0) as client:
        for url in urls:
            logger.info(f"Crawling {url} (up to {max_pages} pages)...")
            pages = crawl_site(url, client, max_pages=max_pages)
            if pages:
                pages = remove_boilerplate(pages)
                logger.success(f"Finished {url} — {len(pages)} pages scraped")
                results.append(SiteResult(site_url=url, pages=pages))
            else:
                logger.error(f"No pages scraped from {url}")
                results.append(
                    SiteResult(site_url=url, error="No pages could be scraped")
                )

    return results


def _sanitize_filename(url: str) -> str:
    """Turn a URL into a safe filename."""
    parsed = urlparse(url)
    name = parsed.netloc.replace("www.", "").replace(".", "_")
    return name


def _add_wrapped_text(pdf: FPDF, text: str) -> None:
    """Add text to the PDF, handling encoding issues gracefully."""
    clean = text.encode("latin-1", errors="replace").decode("latin-1")
    pdf.multi_cell(0, 6, clean)
    pdf.ln(2)


def save_results_as_pdfs(
    results: list[SiteResult], output_dir: str = "data/scraped"
) -> list[Path]:
    """Save one PDF per website, containing all crawled pages."""
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    saved: list[Path] = []

    for site in results:
        if site.error:
            logger.warning(f"Skipping {site.site_url} (error: {site.error})")
            continue

        filename = _sanitize_filename(site.site_url)
        filepath = out / f"{filename}.pdf"

        pdf = FPDF()
        pdf.set_auto_page_break(auto=True, margin=20)

        for page in site.pages:
            pdf.add_page()

            # Page title
            pdf.set_font("Helvetica", "B", 16)
            _add_wrapped_text(pdf, page.title or "Untitled")

            # URL
            pdf.set_font("Helvetica", "", 8)
            pdf.set_text_color(100, 100, 100)
            _add_wrapped_text(pdf, page.url)
            pdf.set_text_color(0, 0, 0)
            pdf.ln(2)

            # Meta description
            if page.meta_description:
                pdf.set_font("Helvetica", "I", 10)
                _add_wrapped_text(pdf, page.meta_description)
                pdf.ln(2)

            # Headings
            # if page.headings:
            #     pdf.set_font("Helvetica", "B", 12)
            #     _add_wrapped_text(pdf, "Headings")
            #     pdf.set_font("Helvetica", "", 10)
            #     for h in page.headings:
            #         prefix = "  " * (h.level - 1)
            #         _add_wrapped_text(pdf, f"{prefix}H{h.level}: {h.text}")

            # Paragraphs
            if page.paragraphs:
                pdf.ln(2)
                pdf.set_font("Helvetica", "B", 12)
                _add_wrapped_text(pdf, "Content")
                pdf.set_font("Helvetica", "", 10)
                for p in page.paragraphs:
                    _add_wrapped_text(pdf, p)

        pdf.output(str(filepath))
        saved.append(filepath)
        logger.success(f"Saved PDF: {filepath} ({len(site.pages)} pages)")

    return saved


def print_summary(results: list[SiteResult]) -> None:
    """Print a readable summary of crawled results."""
    for site in results:
        print(f"\n{'=' * 60}")
        print(f"SITE: {site.site_url}")

        if site.error:
            print(f"  ERROR: {site.error}")
            continue

        total_paragraphs = sum(len(p.paragraphs) for p in site.pages)
        # total_headings = sum(len(p.headings) for p in site.pages)
        print(f"  Pages crawled: {len(site.pages)}")
        # print(f"  Total headings: {total_headings}")
        print(f"  Total paragraphs: {total_paragraphs}")

        for page in site.pages[:3]:
            print(f"    - {page.title} ({page.url})")
        if len(site.pages) > 3:
            print(f"    ... and {len(site.pages) - 3} more pages")


if __name__ == "__main__":
    data = scrape_urls([])
    save_results_as_pdfs(data)
    print_summary(data)

from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from urllib.parse import urljoin, urlparse

import httpx
from bs4 import BeautifulSoup, Tag
from loguru import logger

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
    paragraphs: list[str]
    internal_links: list[str]


@dataclass
class SiteResult:
    site_url: str
    pages: list[PageContent] = field(default_factory=list)
    error: str = ""


def _is_same_domain(url: str, base_domain: str) -> bool:
    parsed = urlparse(url)
    return parsed.netloc == base_domain or parsed.netloc == ""


def _normalize_url(href: str, base_url: str) -> str | None:
    if not href or href.startswith(("#", "mailto:", "tel:", "javascript:")):
        return None

    full = urljoin(base_url, href)
    parsed = urlparse(full)

    if parsed.scheme not in ("http", "https"):
        return None

    path_lower = parsed.path.lower()
    if any(path_lower.endswith(ext) for ext in SKIP_EXTENSIONS):
        return None

    return parsed._replace(fragment="").geturl()


def extract_page_content(html: str, url: str) -> PageContent:
    soup = BeautifulSoup(html, "lxml")

    title = soup.title.string.strip() if soup.title and soup.title.string else ""

    meta_description = ""
    meta_tag = soup.find("meta", attrs={"name": "description"})
    if meta_tag and meta_tag.get("content"):
        meta_description = str(meta_tag["content"]).strip()

    raw_links: list[str] = []
    base_domain = urlparse(url).netloc
    for anchor in soup.find_all("a", href=True):
        normalized = _normalize_url(str(anchor["href"]), url)
        if normalized and _is_same_domain(normalized, base_domain):
            raw_links.append(normalized)

    for element in soup(["script", "style", "nav", "footer", "header", "aside"]):
        element.decompose()

    to_remove: list[Tag] = []
    for section in soup.find_all("section"):
        css_classes = list(section.get_attribute_list("class"))
        if any(
            keyword in str(css_class)
            for css_class in css_classes
            for keyword in ("Newsletter", "newsletter", "filterbox")
        ):
            to_remove.append(section)
    for div in soup.find_all("div"):
        css_classes = list(div.get_attribute_list("class"))
        if any("filterbox" in str(css_class) for css_class in css_classes):
            to_remove.append(div)
    for element in to_remove:
        element.decompose()

    content_root = soup.find("main") or soup.find("article") or soup

    paragraphs = [
        paragraph.get_text(strip=True)
        for paragraph in content_root.find_all("p")
        if len(paragraph.get_text(strip=True)) >= MIN_PARAGRAPH_LENGTH
    ]

    return PageContent(
        url=url,
        title=title,
        meta_description=meta_description,
        paragraphs=paragraphs,
        internal_links=list(dict[str, None].fromkeys(raw_links)),
    )


def remove_boilerplate(
    pages: list[PageContent], threshold: float = 0.1
) -> list[PageContent]:
    if len(pages) < 2:
        return pages

    min_occurrences = max(2, int(len(pages) * threshold))
    paragraph_counts: dict[str, int] = {}

    for page in pages:
        for text in set(page.paragraphs):
            paragraph_counts[text] = paragraph_counts.get(text, 0) + 1

    boilerplate_paragraphs = {
        text for text, count in paragraph_counts.items() if count >= min_occurrences
    }

    cleaned: list[PageContent] = []
    for page in pages:
        cleaned.append(
            PageContent(
                url=page.url,
                title=page.title,
                meta_description=page.meta_description,
                paragraphs=[
                    paragraph
                    for paragraph in page.paragraphs
                    if paragraph not in boilerplate_paragraphs
                ],
                internal_links=page.internal_links,
            )
        )
    return cleaned


def crawl_site(
    start_url: str,
    client: httpx.Client,
    max_pages: int = MAX_PAGES_PER_SITE,
    on_progress: Callable[[str, int, int], None] | None = None,
) -> list[PageContent]:
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
                f"  [{len(visited)}/{max_pages}] {url} - {len(page.paragraphs)} paragraphs"
            )
            if on_progress:
                on_progress(url, len(visited), max_pages)

            for link in page.internal_links:
                if link not in visited:
                    to_visit.append(link)

        except httpx.HTTPStatusError as exc:
            logger.warning(f"  HTTP {exc.response.status_code}: {url}")
        except httpx.RequestError as exc:
            logger.warning(f"  Failed: {url} ({exc})")

    return pages


def scrape_urls(
    urls: list[str],
    max_pages: int = MAX_PAGES_PER_SITE,
    on_progress: Callable[[str, str, int, int], None] | None = None,
) -> list[SiteResult]:
    results: list[SiteResult] = []

    with httpx.Client(headers=HEADERS, follow_redirects=True, timeout=30.0) as client:
        for url in urls:
            logger.info(f"Crawling {url} (up to {max_pages} pages)...")

            def _page_progress(
                page_url: str, visited: int, total: int, _site_url: str = url
            ) -> None:
                if on_progress:
                    on_progress(_site_url, page_url, visited, total)

            pages = crawl_site(
                url, client, max_pages=max_pages, on_progress=_page_progress
            )
            if pages:
                pages = remove_boilerplate(pages)
                logger.success(f"Finished {url} - {len(pages)} pages scraped")
                results.append(SiteResult(site_url=url, pages=pages))
            else:
                logger.error(f"No pages scraped from {url}")
                results.append(
                    SiteResult(site_url=url, error="No pages could be scraped")
                )

    return results


def _sanitize_filename(url: str) -> str:
    parsed = urlparse(url)
    return parsed.netloc.replace("www.", "").replace(".", "_")


def save_results_as_txt(
    results: list[SiteResult], output_dir: str = "data/scraped"
) -> list[Path]:
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    saved: list[Path] = []

    for site in results:
        if site.error:
            logger.warning(f"Skipping {site.site_url} (error: {site.error})")
            continue

        filepath = out / f"{_sanitize_filename(site.site_url)}.txt"
        lines: list[str] = []
        lines.append(f"SITE: {site.site_url}")
        lines.append(f"Scraped: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        lines.append(f"Total pages: {len(site.pages)}")
        lines.append("=" * 60)

        for index, page in enumerate(site.pages, 1):
            lines.append(f"\nPAGE {index}: {page.title or 'Untitled'}")
            lines.append(f"URL: {page.url}")
            if page.meta_description:
                lines.append(f"Description: {page.meta_description}")
            lines.append("-" * 40)
            for paragraph in page.paragraphs:
                lines.append(paragraph)
                lines.append("")
            lines.append("=" * 60)

        filepath.write_text("\n".join(lines), encoding="utf-8")
        saved.append(filepath)
        logger.success(f"Saved: {filepath} ({len(site.pages)} pages)")

    return saved

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
from urllib.parse import urljoin, urlparse

from bs4 import BeautifulSoup, Tag

from src.scraping.extract_markdown import clean_markdown
from src.scraping.settings import fetch_pages_config


config = fetch_pages_config()
LISTING_EXTRACTOR_NAME = "main_links"


class PageKind(StrEnum):
    PROSE = "prose"
    LISTING = "listing"
    DOCUMENT = "document"
    EMPTY = "empty"


@dataclass(frozen=True)
class MainLink:
    label: str
    url: str
    context: str


def classify_page(html: str, page_url: str, prose_markdown: str) -> PageKind:
    if not prose_markdown.strip():
        items = extract_main_links(html, page_url, include_list_items=True)
        return PageKind.LISTING if items else PageKind.EMPTY

    items = extract_main_links(html, page_url, include_list_items=False)
    child_items = [item for item in items if _is_child_page(page_url, item.url)]
    if len(
        child_items
    ) >= config.listing_min_items and _listing_content_is_missing_from_prose(
        child_items, prose_markdown
    ):
        return PageKind.LISTING
    return PageKind.PROSE


def extract_listing_markdown(html: str, page_url: str, title: str | None = None) -> str:
    links = extract_main_links(html, page_url)
    if not links:
        return ""

    lines: list[str] = []
    if title:
        lines.extend([f"# {title}", ""])

    for link in links:
        lines.append(f"- [{link.label}]({link.url})")
        if link.context:
            lines.append(f"  {link.context}")

    return clean_markdown("\n".join(lines))


def extract_main_links(
    html: str, page_url: str, include_list_items: bool = True
) -> list[MainLink]:
    soup = BeautifulSoup(html, "lxml")
    main = soup.select_one("main, [role=main], article")
    if not main:
        return []

    selectors = ["article", "section", "div"]
    if include_list_items:
        selectors.append("li")

    for selector in selectors:
        items = _links_from_repeated_containers(main, selector, page_url)
        if items:
            return items
    return []


def _links_from_repeated_containers(
    main: Tag, selector: str, page_url: str
) -> list[MainLink]:
    links = [
        link
        for container in main.find_all(selector)
        if _is_leaf_item(container)
        for link in [_container_link(container, page_url)]
        if link and _has_listing_context(link)
    ]
    links = _deduplicate_links(links)
    return links if len(links) >= config.listing_min_items else []


def _is_leaf_item(container: Tag) -> bool:
    if container.find_parent(["nav", "header", "footer", "aside"]):
        return False
    if container.find(["article", "li", "section"]):
        return False
    return True


def _container_link(container: Tag, page_url: str) -> MainLink | None:
    anchors = [
        anchor
        for anchor in container.find_all("a", href=True)
        if _clean_text(anchor.get_text(" ", strip=True))
    ]
    if len(anchors) != 1:
        return None

    anchor = anchors[0]
    label = _clean_text(anchor.get_text(" ", strip=True))
    if _is_pagination_label(label):
        return None

    context = _clean_text(container.get_text(" ", strip=True))
    if context.startswith(label):
        context = context[len(label) :].lstrip(" -:,.")
    return MainLink(label=label, url=urljoin(page_url, anchor["href"]), context=context)


def _clean_text(text: str) -> str:
    return " ".join(text.split())


def _is_pagination_label(label: str) -> bool:
    return label.isdigit() or label in {"...", "‹", "›", "<", ">"}


def _has_listing_context(link: MainLink) -> bool:
    return len(link.context.split()) >= config.listing_min_context_words


def _listing_content_is_missing_from_prose(
    links: list[MainLink], prose_markdown: str
) -> bool:
    return any(link.label not in prose_markdown for link in links)


def _is_child_page(page_url: str, link_url: str) -> bool:
    page = urlparse(page_url)
    link = urlparse(link_url)
    page_path = page.path.rstrip("/")
    return page.netloc == link.netloc and link.path.startswith(f"{page_path}/")


def _deduplicate_links(links: list[MainLink]) -> list[MainLink]:
    seen: set[str] = set()
    unique_links: list[MainLink] = []
    for link in links:
        key = f"{link.label}\n{link.url}"
        if key in seen:
            continue
        seen.add(key)
        unique_links.append(link)
    return unique_links

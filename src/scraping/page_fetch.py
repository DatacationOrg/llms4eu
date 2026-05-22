import json
import random
import threading
import time
import uuid
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from urllib.error import URLError
from urllib.parse import urlparse
from urllib.request import Request, urlopen

import httpx
from bs4 import BeautifulSoup

from src.scraping.scraper import HEADERS
from src.scraping.settings import fetch_pages_config
from src.shared.schema import PageMetadata


config = fetch_pages_config()


@dataclass(frozen=True)
class SourceUrl:
    source: str
    url: str


@dataclass(frozen=True)
class FetchedPage:
    metadata: PageMetadata
    html: str | None = None


class DomainThrottle:
    def __init__(self, delay_seconds: float):
        self.delay_seconds = delay_seconds
        self._lock = threading.Lock()
        self._last_request_at: dict[str, float] = {}

    def wait(self, url: str) -> None:
        if self.delay_seconds <= 0:
            return

        domain = urlparse(url).netloc
        with self._lock:
            now = time.monotonic()
            earliest = self._last_request_at.get(domain, 0) + self.delay_seconds
            wait_seconds = max(0, earliest - now)
            self._last_request_at[domain] = now + wait_seconds

        if wait_seconds:
            time.sleep(wait_seconds + random.uniform(0, 0.25))


def read_source_urls(path: Path) -> list[SourceUrl]:
    text = path.read_text(encoding="utf-8")
    if not text.strip():
        return []

    if text.lstrip().startswith("["):
        rows = json.loads(text)
    else:
        rows = [json.loads(line) for line in text.splitlines() if line.strip()]

    source_urls: list[SourceUrl] = []
    for row in rows:
        source = str(row["source"]).strip()
        url = str(row["url"]).strip()
        if source and url:
            source_urls.append(SourceUrl(source=source, url=url))
    return source_urls


def fetch_page(
    source_url: SourceUrl, throttle: DomainThrottle, timeout: float
) -> FetchedPage:
    return _fetch_with_fallback(source_url, source_url.url, throttle, timeout)


def error_page(source_url: SourceUrl, error: str) -> FetchedPage:
    return FetchedPage(
        metadata=PageMetadata(
            id=_page_id(source_url.url),
            source=source_url.source,
            url=source_url.url,
            fetched_at=datetime.now(timezone.utc),
            error=error,
        )
    )


def _fetch_with_fallback(
    source_url: SourceUrl,
    fetch_url: str,
    throttle: DomainThrottle,
    timeout: float,
) -> FetchedPage:
    try:
        return _fetch_once(source_url, fetch_url, throttle, timeout)
    except httpx.RequestError as exc:
        if _should_retry_as_http(fetch_url, exc):
            http_url = fetch_url.replace("https://", "http://", 1)
            try:
                return _fetch_once(source_url, http_url, throttle, timeout)
            except httpx.RequestError as fallback_exc:
                return error_page(
                    source_url,
                    f"{type(fallback_exc).__name__}: {fallback_exc}",
                )
        return error_page(source_url, f"{type(exc).__name__}: {exc}")


def _fetch_once(
    source_url: SourceUrl,
    fetch_url: str,
    throttle: DomainThrottle,
    timeout: float,
) -> FetchedPage:
    with httpx.Client(
        headers=HEADERS, follow_redirects=True, timeout=timeout
    ) as client:
        response: httpx.Response | None = None
        for attempt in range(2):
            throttle.wait(fetch_url)
            response = client.get(fetch_url)
            if response.status_code not in config.retry_statuses or attempt == 1:
                break
            time.sleep(1.0 + random.uniform(0, 0.5))

    assert response is not None
    if response.status_code == 403 and "robot policy" in response.text.lower():
        return _fetch_once_urllib(source_url, fetch_url, throttle, timeout)

    html = response.text
    content_type = response.headers.get("content-type")
    metadata = PageMetadata(
        id=_page_id(source_url.url),
        source=source_url.source,
        url=source_url.url,
        final_url=str(response.url),
        fetched_at=datetime.now(timezone.utc),
        status_code=response.status_code,
        content_type=content_type,
        title=_extract_title(html) if html else None,
        raw_bytes=len(response.content),
        error=_response_error(response, html),
    )
    return FetchedPage(
        metadata=metadata,
        html=html if not metadata.error and _is_html(content_type) else None,
    )


def _fetch_once_urllib(
    source_url: SourceUrl,
    fetch_url: str,
    throttle: DomainThrottle,
    timeout: float,
) -> FetchedPage:
    try:
        throttle.wait(fetch_url)
        request = Request(fetch_url, headers=HEADERS)
        with urlopen(request, timeout=timeout) as response:
            content = response.read()
            final_url = response.url
            status_code = response.status
            content_type = response.headers.get("content-type")
    except URLError as exc:
        return error_page(source_url, f"{type(exc).__name__}: {exc}")

    encoding = "utf-8"
    if content_type and "charset=" in content_type:
        encoding = content_type.rsplit("charset=", 1)[1].split(";", 1)[0].strip()
    html = content.decode(encoding or "utf-8", errors="replace")
    error = (
        None
        if _is_html(content_type)
        else f"non-HTML content-type: {content_type or 'unknown'}"
    )
    metadata = PageMetadata(
        id=_page_id(source_url.url),
        source=source_url.source,
        url=source_url.url,
        final_url=final_url,
        fetched_at=datetime.now(timezone.utc),
        status_code=status_code,
        content_type=content_type,
        title=_extract_title(html) if html else None,
        raw_bytes=len(content),
        error=error,
    )
    return FetchedPage(metadata=metadata, html=html if not error else None)


def _response_error(response: httpx.Response, html: str) -> str | None:
    content_type = response.headers.get("content-type", "")
    if response.status_code >= 400:
        return f"HTTP {response.status_code}"
    if not _is_html(content_type):
        return f"non-HTML content-type: {content_type or 'unknown'}"
    lowered = html[:20_000].lower()
    if "cf-chl" in lowered or ("just a moment" in lowered and "cloudflare" in lowered):
        return "possible Cloudflare challenge page"
    return None


def _extract_title(html: str) -> str | None:
    soup = BeautifulSoup(html, "lxml")
    if soup.title and soup.title.string:
        return soup.title.string.strip()
    return None


def _is_html(content_type: str | None) -> bool:
    return bool(content_type and "text/html" in content_type.lower())


def _page_id(url: str) -> str:
    return str(uuid.uuid5(uuid.NAMESPACE_URL, url))


def _should_retry_as_http(url: str, exc: httpx.RequestError) -> bool:
    message = str(exc).lower()
    return url.startswith("https://") and (
        "certificate verify failed" in message
        or "hostname" in message
        or "certificate" in message
    )

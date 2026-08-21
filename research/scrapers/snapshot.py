"""Fetch the corpus exactly once and snapshot it to disk.

Every extractor later runs against these identical bytes. Re-fetching per
extractor would confound quality differences with snapshot luck (A/B tests,
rotating banners, edits), and would make the arena need network.

Order of work is deliberate: cheap httpx fetches for the whole candidate pool
first, then bucket selection, then the expensive Playwright render pass over
only the pages that were actually selected.
"""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import re
import sys
import time
import urllib.robotparser as robotparser
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from urllib.parse import parse_qs, urlparse, urlsplit

import httpx
import yaml
from loguru import logger

from research.scrapers import store
from research.scrapers.browser import prepare_library_path
from src.scraping.page_fetch import DomainThrottle
from src.scraping.scraper import HEADERS
from src.shared.env import ROOT


CONFIG = store.config()
FETCH = CONFIG["fetch"]
RENDER = CONFIG["render"]

MIN_HTML_BYTES = 2_000  # below this a "page" is an error page or a JS shell

INTERSTITIAL_MARKERS = (
    "just a moment",
    "performing security verification",
    "access to this resource on the server is denied",
    "litespeed technologies",
    "403 forbidden",
    "checking your browser",
    "enable javascript and cookies to continue",
)


def looks_like_interstitial(markup: str) -> str | None:
    """Detect anti-bot / error pages that are structurally valid HTML.

    The raw path already screened these; the rendered path did not, so three
    snapshots were 403 and Cloudflare pages. Those then scored as legitimate
    `ok` extractions of ~190 chars ("Access to this resource ... is denied") and
    counted as rendering "changing the output", making the browser look useful
    exactly where it had failed.
    """
    lowered = markup[:20_000].lower()
    for marker in INTERSTITIAL_MARKERS:
        if marker in lowered:
            return marker
    if "cf-chl" in lowered:
        return "cloudflare challenge"
    return None


def _sha256(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _relative(path: Path) -> str:
    return str(path.relative_to(ROOT))


def load_candidates(buckets: set[str] | None = None) -> list[dict]:
    """Flatten the bucket-keyed YAML pool into rows, preserving pool order.

    `buckets` restricts the pass to named buckets. Needed because a full re-fetch
    rewrites `raw.html` for every page, which invalidates every vote already cast --
    so repinning the two wiki buckets to `?oldid=` permalinks must not drag the two
    non-wiki buckets along with it.
    """
    document = yaml.safe_load((ROOT / CONFIG["corpus_path"]).read_text())
    candidates: list[dict] = []
    for bucket, spec in document["buckets"].items():
        if buckets and bucket not in buckets:
            continue
        for entry in spec["urls"]:
            candidates.append(
                {
                    "id": store.page_id(entry["url"]),
                    "url": entry["url"],
                    "bucket": bucket,
                    "lang": entry.get("lang", spec.get("lang", "en")),
                    "shape": entry.get("shape", "prose"),
                    "is_wiki": bool(spec.get("is_wiki", False)),
                    "note": entry.get("note"),
                }
            )
    return candidates


class RobotsCache:
    """One robots.txt lookup per origin, honoured for every fetch."""

    def __init__(self) -> None:
        self._parsers: dict[str, robotparser.RobotFileParser | None] = {}

    def allows(self, url: str) -> bool:
        if not FETCH.get("respect_robots", True):
            return True
        parts = urlsplit(url)
        origin = f"{parts.scheme}://{parts.netloc}"
        if origin not in self._parsers:
            self._parsers[origin] = self._load(origin)
        parser = self._parsers[origin]
        if parser is None:
            return True  # no robots.txt served -> nothing disallowed
        return parser.can_fetch(HEADERS.get("User-Agent", "*"), url)

    @staticmethod
    def _load(origin: str) -> robotparser.RobotFileParser | None:
        try:
            response = httpx.get(
                f"{origin}/robots.txt",
                headers=HEADERS,
                timeout=15.0,
                follow_redirects=True,
            )
        except httpx.HTTPError:
            return None
        if response.status_code >= 400 or not response.text.strip():
            return None
        parser = robotparser.RobotFileParser()
        parser.parse(response.text.splitlines())
        return parser


def _get(url: str, throttle: DomainThrottle) -> tuple[httpx.Response, float]:
    """Fetch, returning the response and the time spent *fetching*.

    The per-domain politeness delay is waited out before the timer starts. With it
    inside, `fetch_ms` measured the 1.5 s throttle rather than the request: pages
    on multi-candidate domains read ~3400 ms against ~150 ms for single-candidate
    ones, which would have understated the browser's relative cost by ~4x.
    """
    throttle.wait(url)
    started = time.perf_counter()
    with httpx.Client(
        headers=HEADERS, follow_redirects=True, timeout=FETCH["timeout_seconds"]
    ) as client:
        response = client.get(url)
    return response, (time.perf_counter() - started) * 1000


# Hosts whose certificate is known-bad and which are still worth fetching. Kept as an
# explicit list rather than a blanket "accept any bad certificate": corpus-wide, a
# silently downgraded or unverified fetch means substituted content can become a
# research result, and this corpus has exactly one such host.
CERT_EXEMPT_HOSTS = frozenset({"www.brestanica.com", "brestanica.com"})


def _cert_exempt(url: str) -> bool:
    from urllib.parse import urlsplit

    return (urlsplit(url).hostname or "").lower() in CERT_EXEMPT_HOSTS


def _is_certificate_error(exc: Exception) -> bool:
    message = str(exc).lower()
    return "certificate" in message or "hostname mismatch" in message


def _allows_framing(headers) -> bool:
    """Whether the live URL can be shown in the arena's reference iframe.

    Recorded at fetch time because the reference pane is a live iframe: the page the
    reviewer compares against should be the real site. 29 of the corpus's 52 domains
    refuse framing -- and they are almost exactly the volatile news and government
    pages -- while every Wikipedia host allows it, so the pinned wiki slice frames
    cleanly and the rest falls back to the saved DOM.

    `X-Frame-Options` has no "allow any origin" value: DENY and SAMEORIGIN both block
    us, and the obsolete ALLOW-FROM is ignored by every current browser. So any value
    at all means blocked. CSP `frame-ancestors` is only permissive if it names `*`.
    """
    if headers.get("x-frame-options", "").strip():
        return False
    for directive in headers.get("content-security-policy", "").split(";"):
        directive = directive.strip()
        if directive.lower().startswith("frame-ancestors"):
            return "*" in directive.split(None, 1)[-1].split()
    return True


def fetch_raw(candidate: dict, throttle: DomainThrottle, robots: RobotsCache) -> dict:
    """httpx GET -> raw.html. Returns a result dict; never raises."""
    url = candidate["url"]
    if not robots.allows(url):
        return {"error": "robots.txt disallows this URL"}

    try:
        response, elapsed_ms = _get(url, throttle)
    except httpx.HTTPError as exc:
        if (
            _is_certificate_error(exc)
            and url.startswith("https://")
            and _cert_exempt(url)
        ):
            try:
                response, elapsed_ms = _get(
                    url.replace("https://", "http://", 1), throttle
                )
            except httpx.HTTPError as fallback_exc:
                return {"error": f"{type(fallback_exc).__name__}: {fallback_exc}"}
        else:
            return {"error": f"{type(exc).__name__}: {exc}"}

    content_type = response.headers.get("content-type", "")
    result = {
        "status_code": response.status_code,
        "final_url": str(response.url),
        "content_type": content_type,
        "charset": response.encoding,
        "fetch_ms": elapsed_ms,
        "bytes": len(response.content),
        "frameable": int(_allows_framing(response.headers)),
    }
    if response.status_code >= 400:
        result["error"] = f"HTTP {response.status_code}"
        return result
    if "html" not in content_type.lower():
        result["error"] = f"non-HTML content-type: {content_type or 'unknown'}"
        return result

    html = response.text
    marker = looks_like_interstitial(html)
    if marker:
        result["error"] = f"interstitial page ({marker})"
        return result
    if len(response.content) < MIN_HTML_BYTES:
        result["error"] = f"suspiciously small ({len(response.content)} bytes)"
        return result

    result["html"] = html
    return result


def run_raw_pass(candidates: list[dict]) -> dict[str, dict]:
    throttle = DomainThrottle(FETCH["domain_delay_seconds"])
    robots = RobotsCache()
    results: dict[str, dict] = {}

    with ThreadPoolExecutor(max_workers=FETCH["workers"]) as pool:
        futures = {
            pool.submit(fetch_raw, candidate, throttle, robots): candidate
            for candidate in candidates
        }
        for future, candidate in futures.items():
            try:
                results[candidate["id"]] = future.result()
            except Exception as exc:  # defensive: a worker must never sink the run
                results[candidate["id"]] = {"error": f"worker crashed: {exc}"}

    for candidate in candidates:
        result = results[candidate["id"]]
        page_directory = store.snapshot_dir(candidate["id"])
        html = result.pop("html", None)
        path = None
        sha = None
        if html is not None:
            target = page_directory / "raw.html"
            target.write_text(html, encoding="utf-8")
            path = _relative(target)
            sha = _sha256(html)
        store.record_snapshot(candidate["id"], "raw", path=path, sha256=sha, **result)
        status = "ok" if html is not None else f"DROP {result.get('error')}"
        logger.info(
            f"raw  {candidate['bucket']:9s} {candidate['url'][:70]:70s} {status}"
        )

    return results


def select_pages(candidates: list[dict], results: dict[str, dict]) -> list[dict]:
    """Keep the first `target_per_bucket` candidates per bucket that fetched cleanly.

    Pool order is curation order, so this preserves the intended shape spread
    rather than skewing toward whichever sites happen to be fastest.
    """
    document = yaml.safe_load((ROOT / CONFIG["corpus_path"]).read_text())
    target = int(document["target_per_bucket"])

    selected: list[dict] = []
    counts: dict[str, int] = {}
    for candidate in candidates:
        bucket = candidate["bucket"]
        result = results[candidate["id"]]
        if result.get("error"):
            store.mark_dropped(candidate["id"], result["error"])
            continue
        if counts.get(bucket, 0) >= target:
            store.mark_dropped(candidate["id"], "surplus: bucket already full")
            continue
        counts[bucket] = counts.get(bucket, 0) + 1
        selected.append(candidate)

    for bucket, count in sorted(counts.items()):
        marker = "OK" if count >= target else f"SHORT by {target - count}"
        logger.info(f"bucket {bucket:9s} {count}/{target} {marker}")
    return selected


async def render_pages(pages: list[dict]) -> None:
    """Playwright pass: the rendered DOM for each page.

    `rendered.html` is an extractor *input*, so rewriting it invalidates every vote
    already cast on a `@rendered` entrant. That is why `--force` is opt-in and the
    default resumes rather than re-renders.
    """
    prepare_library_path()
    from playwright.async_api import async_playwright

    semaphore = asyncio.Semaphore(FETCH["workers"])

    async with async_playwright() as playwright:
        browser = await playwright.chromium.launch(headless=True)

        raw_final = {
            row["page_id"]: row["final_url"]
            for row in store.load_snapshots("raw")
            if row.get("final_url") and not row.get("error")
        }

        async def one(page_row: dict) -> None:
            async with semaphore:
                # brestanica.com serves a bad certificate and 403s over https in
                # Chromium; the raw pass reached it by downgrading to http. Using
                # the URL that actually worked keeps the two variants comparable.
                target = raw_final.get(page_row["id"]) or page_row["url"]
                context = await browser.new_context(
                    viewport={"width": RENDER["viewport_width"], "height": 1200},
                    # Only for CERT_EXEMPT_HOSTS. brestanica.com serves a certificate
                    # with a hostname mismatch, and the production fetcher downgrades
                    # it rather than dropping the page (page_fetch.py
                    # _should_retry_as_http), so accepting it here keeps the two
                    # snapshot variants consistent. Every other host is verified: a
                    # blanket exemption would let a substituted page become a result.
                    ignore_https_errors=_cert_exempt(target),
                    device_scale_factor=1,
                    locale="en-US",
                    extra_http_headers={
                        "Accept-Language": f"{page_row['lang']},en;q=0.8"
                    },
                )
                page = await context.new_page()
                directory = store.snapshot_dir(page_row["id"])
                try:
                    started = time.perf_counter()
                    try:
                        await page.goto(
                            target,
                            wait_until="networkidle",
                            timeout=FETCH["browser_timeout_ms"],
                        )
                    except Exception:
                        # Ad/tracker-heavy pages never reach networkidle. Settle for
                        # domcontentloaded plus a short grace period instead of
                        # losing the page entirely.
                        await page.goto(
                            target,
                            wait_until="domcontentloaded",
                            timeout=FETCH["browser_timeout_ms"],
                        )
                        await page.wait_for_timeout(3000)
                    html = await page.content()
                    elapsed_ms = (time.perf_counter() - started) * 1000
                    title = await page.title()

                    marker = looks_like_interstitial(html)
                    if marker:
                        store.record_snapshot(
                            page_row["id"],
                            "rendered",
                            error=f"interstitial page ({marker})",
                        )
                        logger.warning(
                            f"render REJECTED {page_row['url'][:56]}: {marker}"
                        )
                        return

                    target = directory / "rendered.html"
                    target.write_text(html, encoding="utf-8")
                    store.record_snapshot(
                        page_row["id"],
                        "rendered",
                        path=_relative(target),
                        sha256=_sha256(html),
                        bytes=len(html.encode("utf-8")),
                        final_url=page.url,
                        fetch_ms=elapsed_ms,
                    )
                    store.set_title(page_row["id"], title)
                    logger.info(
                        f"render {page_row['url'][:62]:62s} {elapsed_ms:7.0f}ms"
                    )
                except Exception as exc:
                    message = f"{type(exc).__name__}: {exc}"[:300]
                    store.record_snapshot(page_row["id"], "rendered", error=message)
                    logger.warning(f"render FAILED {page_row['url'][:60]}: {message}")
                finally:
                    await context.close()

        await asyncio.gather(*(one(row) for row in pages))
        await browser.close()


def fetch_wikitext(pages: list[dict]) -> None:
    """MediaWiki action=raw for the wiki slice -- Wikiextractor's only valid input.

    The `oldid` from the page URL is forwarded to `action=raw`, so wikiextractor reads
    *the same revision* the HTML extractors were given. Without it this pass fetched
    whatever was current when it ran -- a median 44 minutes after `raw.html` -- which
    made the wiki three-way a comparison across two possibly-different revisions of
    the same article. Checked after the fact: 0 of the 50 corpus articles were edited
    inside that window, so no result was actually contaminated. It was luck, and the
    corpus URLs are now pinned so it cannot recur.
    """
    wiki_pages = [row for row in pages if row["is_wiki"]]
    throttle = DomainThrottle(FETCH["domain_delay_seconds"])

    def one(page_row: dict) -> None:
        parts = urlparse(page_row["url"])
        if "/wiki/" not in parts.path:
            store.record_snapshot(page_row["id"], "wikitext", error="not a /wiki/ URL")
            return
        title = parts.path.split("/wiki/", 1)[1]
        origin = f"{parts.scheme}://{parts.netloc}"
        oldid = parse_qs(parts.query).get("oldid", [None])[0]

        def raw_wikitext(page_title: str) -> tuple[httpx.Response | None, float]:
            api = f"{origin}/w/index.php?title={page_title}&action=raw"
            if oldid:
                api += f"&oldid={oldid}"
            throttle.wait(api)
            started = time.perf_counter()
            try:
                response = httpx.get(
                    api,
                    headers=HEADERS,
                    timeout=FETCH["timeout_seconds"],
                    follow_redirects=True,
                )
            except httpx.HTTPError:
                return None, (time.perf_counter() - started) * 1000
            return response, (time.perf_counter() - started) * 1000

        response, elapsed_ms = raw_wikitext(title)
        if response is None:
            store.record_snapshot(
                page_row["id"],
                "wikitext",
                error="wikitext fetch failed",
                fetch_ms=elapsed_ms,
            )
            return
        if response.status_code >= 400 or not response.text.strip():
            store.record_snapshot(
                page_row["id"],
                "wikitext",
                error=f"HTTP {response.status_code}",
                fetch_ms=elapsed_ms,
            )
            return

        # action=raw returns the redirect stub itself, not the target. Three pages
        # were 40-90 byte "#REDIRECT [[X]]" documents, so wikiextractor was being
        # scored on nothing through no fault of its own. The keyword is localised
        # (#ÁTIRÁNYÍTÁS in Hungarian, #PREUSMERITEV in Slovene), so match the
        # link rather than the keyword.
        # A pinned revision is never a redirect stub, and following one would silently
        # unpin it (the target's *current* revision, not the one we snapshotted).
        text = response.text
        redirect = re.match(r"\s*#[A-Za-zÀ-ž]+\s*\[\[([^\]|#]+)", text)
        if redirect and len(text) < 500 and not oldid:
            followed, extra_ms = raw_wikitext(
                redirect.group(1).strip().replace(" ", "_")
            )
            elapsed_ms += extra_ms
            if (
                followed is not None
                and followed.status_code < 400
                and followed.text.strip()
            ):
                logger.info(
                    f"wikitext followed redirect: {title} -> "
                    f"{redirect.group(1).strip()}"
                )
                text = followed.text
                response = followed

        target = store.snapshot_dir(page_row["id"]) / "wikitext.txt"
        target.write_text(text, encoding="utf-8")
        store.record_snapshot(
            page_row["id"],
            "wikitext",
            path=_relative(target),
            sha256=_sha256(text),
            bytes=len(text.encode("utf-8")),
            status_code=response.status_code,
            fetch_ms=elapsed_ms,
        )

    with ThreadPoolExecutor(max_workers=FETCH["workers"]) as pool:
        list(pool.map(one, wiki_pages))
    logger.info(f"wikitext fetched for {len(wiki_pages)} wiki pages")


def verify() -> int:
    """Report snapshot gaps. Exit code is the number of problems found."""
    pages = store.load_pages()
    snapshots: dict[tuple[str, str], dict] = {
        (row["page_id"], row["variant"]): row for row in store.load_snapshots()
    }
    problems = 0
    # `raw` and `rendered` are the real extractor inputs, so a missing one is a
    # problem. The arena's reference pane is a live iframe of the (pinned) URL, or the
    # saved DOM where a site refuses framing -- no other artefact is required.
    for page in pages:
        for variant in ("raw", "rendered"):
            row = snapshots.get((page["id"], variant))
            if not row or row.get("error") or not row.get("path"):
                reason = (row or {}).get("error", "missing")
                logger.warning(
                    f"{page['bucket']:9s} {page['url'][:60]:60s} {variant}: {reason}"
                )
                problems += 1
        if page["is_wiki"]:
            row = snapshots.get((page["id"], "wikitext"))
            if not row or row.get("error"):
                logger.warning(f"{page['url'][:60]:60s} wikitext missing")
                problems += 1

    by_bucket: dict[str, int] = {}
    for page in pages:
        by_bucket[page["bucket"]] = by_bucket.get(page["bucket"], 0) + 1
    logger.info(f"selected pages: {len(pages)} {by_bucket}")
    logger.info(f"problems: {problems}")
    return problems


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--verify", action="store_true", help="report gaps and exit")
    parser.add_argument(
        "--force",
        action="store_true",
        help="re-render pages that already have snapshots",
    )
    parser.add_argument("--skip-render", action="store_true", help="raw pass only")
    parser.add_argument(
        "--buckets",
        default="",
        help="comma-separated bucket names to restrict the pass to; a full pass "
        "rewrites raw.html for every page and invalidates every existing vote",
    )
    parser.add_argument(
        "--render-only",
        action="store_true",
        help="render the already-selected pages, skipping the raw pass",
    )
    arguments = parser.parse_args()

    store.initialize()
    if arguments.verify:
        sys.exit(0 if verify() == 0 else 1)

    if arguments.render_only:
        selected = store.load_pages()
        if not arguments.force:
            # Resume: keep snapshots already captured so their bytes stay stable,
            # and only retry pages that failed or were never rendered.
            done = {
                row["page_id"]
                for row in store.load_snapshots("rendered")
                if row.get("path") and not row.get("error")
            }
            selected = [row for row in selected if row["id"] not in done]
        logger.info(f"rendering {len(selected)} pages")
        asyncio.run(render_pages(selected))
        fetch_wikitext(selected)
        verify()
        return

    buckets = {b.strip() for b in arguments.buckets.split(",") if b.strip()}
    candidates = load_candidates(buckets or None)
    if buckets:
        logger.info(f"restricted to buckets: {', '.join(sorted(buckets))}")
    store.upsert_pages(
        [
            store.Page(
                id=c["id"],
                url=c["url"],
                bucket=c["bucket"],
                lang=c["lang"],
                shape=c["shape"],
                is_wiki=c["is_wiki"],
                note=c["note"],
            )
            for c in candidates
        ]
    )
    logger.info(f"{len(candidates)} candidates in the pool")

    results = run_raw_pass(candidates)
    selected = select_pages(candidates, results)
    logger.info(f"{len(selected)} pages selected")

    if not arguments.skip_render:
        asyncio.run(render_pages(selected))
        fetch_wikitext(selected)
    verify()


if __name__ == "__main__":
    main()

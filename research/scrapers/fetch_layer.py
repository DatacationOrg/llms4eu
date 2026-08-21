"""Compare fetch layers: our httpx client vs Scrapy vs Scrapling.

Scrapy and Scrapling sit at a different layer from the extractors -- they
retrieve HTML and have no main-content model -- so voting on them would be
meaningless: `scrapy + html-text` emits output byte-identical to `html-text`.

Rather than assume two HTTP clients return the same bytes, this measures it:
every fetcher runs over the same URLs, and the normalised text of each response
is hashed and compared. Pages where the fetchers genuinely diverge are worth
promoting into the arena (charset mishandling, partial or blocked responses);
pages where they agree are a reported number, not a vote.

Runs in its own interpreter because scrapling[fetchers] pins a newer Playwright
than the project env:

    just arena-fetchers
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from research.scrapers import store  # noqa: E402


USER_AGENT = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/124.0 Safari/537.36"
)
TIMEOUT = 30.0

_SPACE = re.compile(r"\s+")


def normalised_text(markup: str) -> str:
    """Visible text, extracted with a real parser.

    A regex tag-strip is not good enough here and quietly produced a false
    finding: `<[^>]+>` terminates at the first `>` *inside* an attribute value or
    a comment, so Wikipedia's JSON-in-attribute payloads and any comment
    containing `>` leaked fragments like `of the United Nations"}},"i":0}}]}'>`
    and `--> ` into the "visible text". Scrapling returns an lxml-reserialized
    DOM in which those attributes are re-quoted and comments dropped, so the leak
    happened on one side only -- and the comparison reported 79 of 93 pages as
    fetcher disagreements when the fetchers had in fact returned the same content.

    Parsing both sides identically neutralises representation differences, which
    is the whole point: a genuine content difference still survives.
    """
    from lxml import etree, html as lxml_html

    try:
        tree = lxml_html.fromstring(markup)
    except (etree.ParserError, etree.XMLSyntaxError, ValueError):
        return _SPACE.sub(" ", markup).strip()
    for element in tree.xpath("//script | //style | //noscript | //comment()"):
        parent = element.getparent()
        if parent is not None:
            parent.remove(element)
    return _SPACE.sub(" ", tree.text_content()).strip()


def normalised_hash(markup: str) -> str:
    """Hash the visible text, so representation differences do not count.

    Whitespace is removed entirely rather than collapsed. `text_content()` inserts
    no separator at element boundaries, so whether two words end up as "foo bar" or
    "foobar" depends on whether an insignificant whitespace text node survived
    serialization -- and Scrapling returns a reserialized DOM, so it often did not.
    That produced a stream of single-space "differences" on pages whose content was
    identical. The cost is that a genuine difference consisting only of whitespace
    is invisible here, which is the right trade: no extractor's quality turns on it,
    and the question this answers is whether the fetchers retrieved the same content.
    """
    text = normalised_text(markup)
    return hashlib.sha256("".join(text.split()).encode("utf-8")).hexdigest()


def classify(status: int | None, html: str | None, error: str | None) -> str:
    if error:
        return "error"
    if status and status >= 400:
        return "blocked" if status in (401, 403, 429) else "http-error"
    if not html or len(html) < 2000:
        return "too-small"
    lowered = html[:20000].lower()
    if "cf-chl" in lowered or ("just a moment" in lowered and "cloudflare" in lowered):
        return "anti-bot"
    return "ok"


# ------------------------------------------------------------------- fetchers


def fetch_httpx(urls: list[str]) -> dict[str, dict]:
    import httpx

    results: dict[str, dict] = {}
    headers = {"User-Agent": USER_AGENT, "Accept-Language": "en;q=0.9"}
    for url in urls:
        started = time.perf_counter()
        try:
            with httpx.Client(
                headers=headers, follow_redirects=True, timeout=TIMEOUT
            ) as client:
                response = client.get(url)
            results[url] = {
                "status": response.status_code,
                "html": response.text,
                "charset": response.encoding,
                "error": None,
                "ms": (time.perf_counter() - started) * 1000,
            }
        except Exception as exc:
            results[url] = {
                "status": None,
                "html": None,
                "charset": None,
                "error": f"{type(exc).__name__}: {exc}"[:200],
                "ms": (time.perf_counter() - started) * 1000,
            }
    return results


def fetch_scrapy(urls: list[str]) -> dict[str, dict]:
    """A real minimal Scrapy spider, so this measures Scrapy and not a stand-in."""
    import scrapy
    from scrapy.crawler import CrawlerProcess

    results: dict[str, dict] = {}

    class CompareSpider(scrapy.Spider):
        name = "fetch-compare"
        custom_settings = {
            "ROBOTSTXT_OBEY": True,  # same courtesy as the snapshot pass
            "CONCURRENT_REQUESTS": 4,
            "DOWNLOAD_TIMEOUT": TIMEOUT,
            "USER_AGENT": USER_AGENT,
            "LOG_LEVEL": "ERROR",
            "RETRY_ENABLED": False,
            "COOKIES_ENABLED": True,
            "TELNETCONSOLE_ENABLED": False,
        }

        # Scrapy >= 2.13 replaced start_requests() with an async start(); defining
        # only the old hook silently yields zero requests and the crawl "succeeds"
        # having fetched nothing. Both are defined so this works either way.
        async def start(self):
            for url in urls:
                yield self._build(url)

        def start_requests(self):
            for url in urls:
                yield self._build(url)

        def _build(self, url):
            return scrapy.Request(
                url,
                callback=self.parse,
                errback=self.on_error,
                dont_filter=True,
                meta={"t0": time.perf_counter(), "src": url},
            )

        def parse(self, response):
            source = response.meta["src"]
            try:
                body = response.text
            except Exception:
                body = None
            results[source] = {
                "status": response.status,
                "html": body,
                "charset": getattr(response, "encoding", None),
                "error": None,
                "ms": (time.perf_counter() - response.meta["t0"]) * 1000,
            }

        def on_error(self, failure):
            source = failure.request.meta["src"]
            results[source] = {
                "status": None,
                "html": None,
                "charset": None,
                "error": repr(failure.value)[:200],
                "ms": (time.perf_counter() - failure.request.meta["t0"]) * 1000,
            }

    process = CrawlerProcess(settings={"LOG_LEVEL": "ERROR"})
    process.crawl(CompareSpider)
    process.start()  # blocks until the crawl finishes

    for url in urls:
        results.setdefault(
            url,
            {
                "status": None,
                "html": None,
                "charset": None,
                "error": "no response (robots.txt disallowed or dropped)",
                "ms": 0.0,
            },
        )
    return results


def fetch_scrapling(urls: list[str], stealthy: bool = False) -> dict[str, dict]:
    if stealthy:
        # StealthyFetcher launches a browser, so it needs the vendored
        # libasound.so.2 on the loader path just like the snapshot pass.
        from research.scrapers.browser import prepare_library_path

        prepare_library_path()
    from scrapling import Fetcher, StealthyFetcher

    results: dict[str, dict] = {}
    for url in urls:
        started = time.perf_counter()
        try:
            if stealthy:
                page = StealthyFetcher.fetch(url, headless=True, timeout=TIMEOUT * 1000)
            else:
                page = Fetcher.get(
                    url, timeout=TIMEOUT, stealthy_headers=True, follow_redirects=True
                )
            results[url] = {
                "status": getattr(page, "status", None),
                "html": page.html_content,
                "charset": getattr(page, "encoding", None),
                "error": None,
                "ms": (time.perf_counter() - started) * 1000,
            }
        except Exception as exc:
            results[url] = {
                "status": None,
                "html": None,
                "charset": None,
                "error": f"{type(exc).__name__}: {exc}"[:200],
                "ms": (time.perf_counter() - started) * 1000,
            }
    return results


# ----------------------------------------------------------------------- main


def recompute(urls: list[str], by_url: dict[str, dict]) -> None:
    """Re-run the comparison over saved bodies, keeping the measured timings.

    Success rates and latencies are properties of the fetch and are read back from
    the database; only the content comparison is recomputed. Anything the previous
    run failed to save simply drops out of the comparison rather than counting as a
    disagreement.
    """
    previous_path = store.fetch_layer_json()
    previous = (
        json.loads(previous_path.read_text(encoding="utf-8"))
        if previous_path.exists()
        else {}
    )
    if not previous.get("fetchers"):
        raise SystemExit("no previous fetch_layer.json to recompute from; run a pass")

    bodies: dict[str, dict[str, dict]] = {}
    missing = 0
    for name in previous["fetchers"]:
        bodies[name] = {}
        for url in urls:
            path = (
                store.data_dir()
                / "snapshots"
                / by_url[url]["id"]
                / f"fetchcmp-{name}.html"
            )
            if path.exists():
                bodies[name][url] = {
                    "html": path.read_text(encoding="utf-8", errors="replace")
                }
            else:
                missing += 1
    print(f"recomputing from saved bodies ({missing} not saved)", flush=True)

    for name, results in bodies.items():
        for url, row in results.items():
            store.record_snapshot(
                by_url[url]["id"],
                f"fetchcmp-{name}",
                sha256=normalised_hash(row["html"]),
            )

    report = {
        "urls": len(urls),
        "fetchers": previous["fetchers"],
        **compare_bodies(urls, by_url, bodies),
    }
    write_report(report, urls)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--limit", type=int, default=0, help="first N pages only")
    parser.add_argument(
        "--skip-stealthy",
        action="store_true",
        help="skip StealthyFetcher (a browser launch per URL)",
    )
    parser.add_argument("--skip-scrapy", action="store_true")
    parser.add_argument(
        "--recompute",
        action="store_true",
        help="re-derive divergence from persisted bodies, without fetching anything",
    )
    arguments = parser.parse_args()

    pages = store.load_pages()
    if arguments.limit:
        pages = pages[: arguments.limit]
    by_url = {page["url"]: page for page in pages}
    urls = list(by_url)
    print(f"comparing fetch layers over {len(urls)} URLs", flush=True)

    if arguments.recompute:
        recompute(urls, by_url)
        return

    passes: dict[str, dict[str, dict]] = {}
    print("  httpx (ours) ...", flush=True)
    passes["ours-httpx"] = fetch_httpx(urls)
    # Control: the same client, twice. Many pages embed timestamps, nonces, CSRF
    # tokens or rotating ad slots, so two sequential fetches differ even with an
    # identical client. Without this baseline, "the fetchers disagree" is
    # unfalsifiable -- page dynamism would masquerade as a fetcher difference.
    print("  httpx again (self-consistency control) ...", flush=True)
    passes["ours-httpx-repeat"] = fetch_httpx(urls)
    if not arguments.skip_scrapy:
        print("  scrapy ...", flush=True)
        passes["scrapy"] = fetch_scrapy(urls)
    print("  scrapling Fetcher ...", flush=True)
    passes["scrapling"] = fetch_scrapling(urls, stealthy=False)
    if not arguments.skip_stealthy:
        print("  scrapling StealthyFetcher ...", flush=True)
        passes["scrapling-stealthy"] = fetch_scrapling(urls, stealthy=True)

    summary: dict[str, dict] = {}
    for name, results in passes.items():
        outcomes: dict[str, int] = {}
        times: list[float] = []
        for url in urls:
            row = results.get(url, {})
            outcome = classify(row.get("status"), row.get("html"), row.get("error"))
            outcomes[outcome] = outcomes.get(outcome, 0) + 1
            if outcome == "ok":
                times.append(row["ms"])
            html = row.get("html")
            # Persist the body. The first run stored hashes only, so when the
            # divergence count looked wrong there was nothing to diff and the whole
            # comparison had to be re-fetched to find the bug. Cheap insurance:
            # .local is gitignored.
            saved = None
            if html:
                target = (
                    store.data_dir()
                    / "snapshots"
                    / by_url[url]["id"]
                    / f"fetchcmp-{name}.html"
                )
                target.parent.mkdir(parents=True, exist_ok=True)
                target.write_text(html, encoding="utf-8", errors="replace")
                saved = str(target)
            store.record_snapshot(
                by_url[url]["id"],
                f"fetchcmp-{name}",
                path=saved,
                bytes=len(html.encode("utf-8", "replace")) if html else 0,
                sha256=normalised_hash(html) if html else None,
                status_code=row.get("status"),
                charset=row.get("charset"),
                fetch_ms=row.get("ms"),
                error=row.get("error"),
            )
        times.sort()
        summary[name] = {
            "outcomes": outcomes,
            "ok": outcomes.get("ok", 0),
            "success_rate": round(100.0 * outcomes.get("ok", 0) / len(urls), 1),
            "median_ms": round(times[len(times) // 2], 1) if times else None,
            "total_s": round(
                sum(row.get("ms", 0) for row in results.values()) / 1000, 1
            ),
        }

    divergence = compare_bodies(
        urls, by_url, {name: results for name, results in passes.items()}
    )
    report = {
        "urls": len(urls),
        "fetchers": summary,
        **divergence,
    }
    write_report(report, urls)


def compare_bodies(urls, by_url, bodies: dict[str, dict]) -> dict:
    """Divergence, split into page dynamism vs genuine fetcher difference.

    Takes bodies rather than fetching, so `--recompute` can re-derive every number
    from the persisted snapshots. That matters: the first two runs of this
    comparison both reported a divergence count that turned out to be an artifact of
    how the text was normalised, and each diagnosis cost a full re-fetch of 100 URLs
    through five clients. Normalisation is now a pure function of bytes on disk.
    """
    compare_names = [name for name in bodies if name != "ours-httpx-repeat"]
    divergent: list[dict] = []
    agreed = 0
    dynamic: list[str] = []
    for url in urls:
        first = bodies["ours-httpx"].get(url, {}).get("html")
        repeat = bodies.get("ours-httpx-repeat", {}).get(url, {}).get("html")
        if first and repeat and normalised_hash(first) != normalised_hash(repeat):
            # The page changed between two identical fetches. Any cross-fetcher
            # difference here is uninformative, so it is excluded rather than
            # counted as a fetcher disagreement.
            dynamic.append(url)
            continue

        hashes = {}
        for name in compare_names:
            row = bodies[name].get(url, {})
            html = row.get("html")
            if html:
                hashes[name] = normalised_hash(html)
        if len(hashes) < 2:
            continue
        if len(set(hashes.values())) == 1:
            agreed += 1
        else:
            divergent.append(
                {
                    "url": url,
                    "page_id": by_url[url]["id"],
                    "lang": by_url[url]["lang"],
                    "distinct": len(set(hashes.values())),
                    "by_fetcher": {name: value[:12] for name, value in hashes.items()},
                    "chars": {
                        name: len(bodies[name][url]["html"] or "") for name in hashes
                    },
                }
            )

    return {
        "dynamic_pages": len(dynamic),
        "dynamic_urls": dynamic,
        "content_agreed": agreed,
        "content_divergent": len(divergent),
        "divergent_pages": divergent,
    }


def write_report(report: dict, urls: list[str]) -> None:
    summary = report["fetchers"]
    dynamic = report["dynamic_urls"]
    agreed = report["content_agreed"]
    divergent = report["divergent_pages"]
    target = store.fetch_layer_json()
    target.write_text(json.dumps(report, indent=2), encoding="utf-8")

    print("\n=== fetch layer summary ===", flush=True)
    for name, row in summary.items():
        print(
            f"  {name:22s} ok {row['ok']:3d}/{len(urls)} "
            f"({row['success_rate']:5.1f}%)  median {row['median_ms']} ms  "
            f"{row['outcomes']}",
            flush=True,
        )
    print(
        f"\npages excluded as dynamic (same client, two fetches differ): "
        f"{len(dynamic)}/{len(urls)}"
    )
    print(f"of the {agreed + len(divergent)} stable pages:")
    print(f"  content identical across fetchers: {agreed}")
    print(f"  content genuinely divergent:       {len(divergent)}")
    for row in divergent[:12]:
        print(f"  {row['lang']}  {row['url'][:64]}  chars={row['chars']}")
    print(f"\nwrote {target}")


if __name__ == "__main__":
    main()

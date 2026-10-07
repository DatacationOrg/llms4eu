import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from src.scraping.page_extract import FetchResult, extract_page
from src.scraping.page_fetch import (
    DomainThrottle,
    SourceUrl,
    error_page,
    fetch_page,
    read_source_urls,
)
from src.scraping.page_store import append_fetch_result
from src.scraping.settings import fetch_pages_config
from src.shared.env import data_path, load_local_env

config = fetch_pages_config()


def scrape_source_file(
    source_path: Path,
    out: Path,
    workers: int = config.workers,
    domain_delay_seconds: float = config.domain_delay_seconds,
    timeout: float = config.timeout_seconds,
) -> dict[str, int]:
    source_urls = read_source_urls(source_path)

    throttle = DomainThrottle(domain_delay_seconds)
    counts = {"total": len(source_urls), "ok": 0, "failed": 0, "non_html": 0}

    with ThreadPoolExecutor(max_workers=workers) as executor:
        futures = {
            executor.submit(
                _fetch_and_extract, source_url, throttle, timeout
            ): source_url
            for source_url in source_urls
        }
        for future in as_completed(futures):
            source_url = futures[future]
            try:
                result = future.result()
            except Exception as exc:
                result = extract_page(
                    error_page(source_url, f"{type(exc).__name__}: {exc}"),
                    timeout,
                )

            append_fetch_result(result, out)
            status = _record_count(counts, result)
            print(f"{status}: {source_url.url}", flush=True)

    return counts


def _fetch_and_extract(
    source_url: SourceUrl, throttle: DomainThrottle, timeout: float
) -> FetchResult:
    return extract_page(fetch_page(source_url, throttle, timeout), timeout)


def _record_count(counts: dict[str, int], result: FetchResult) -> str:
    if result.markdown_content and not result.metadata.error:
        counts["ok"] += 1
        return "ok"

    counts["failed"] += 1
    if result.metadata.error and result.metadata.error.startswith("non-HTML"):
        counts["non_html"] += 1
    return "failed"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("source_file", type=Path)
    parser.add_argument(
        "--out", type=Path, help="JSONL to append to (default: scraped/<source>.jsonl)"
    )
    parser.add_argument("--workers", type=int, default=config.workers)
    parser.add_argument(
        "--domain-delay",
        type=float,
        default=config.domain_delay_seconds,
        help="Minimum seconds between requests to the same domain.",
    )
    parser.add_argument("--timeout", type=float, default=config.timeout_seconds)
    args = parser.parse_args()

    load_local_env()
    out = args.out or data_path("scraped", f"{args.source_file.stem}.jsonl")
    counts = scrape_source_file(
        source_path=args.source_file,
        out=out,
        workers=args.workers,
        domain_delay_seconds=args.domain_delay,
        timeout=args.timeout,
    )
    print(
        "finished: "
        f"total={counts['total']} ok={counts['ok']} "
        f"failed={counts['failed']} non_html={counts['non_html']} out={out}"
    )


if __name__ == "__main__":
    main()

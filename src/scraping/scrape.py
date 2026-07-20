import argparse
from pathlib import Path

from src.preprocess.index import rebuild_vector_index
from src.scraping.ingest import upsert_places
from src.scraping.scraper import SiteResult, save_results_as_txt, scrape_urls
from src.scraping.transform import transform_results
from src.shared.env import ROOT, load_local_env, load_yaml


CONFIG = load_yaml(Path(__file__).with_name("config.yaml"))


def scrape() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--urls", nargs="+", required=True)
    parser.add_argument("--max-pages", type=int, default=CONFIG["max_pages"])
    parser.add_argument("--place-per", choices=["page", "site"], default=None)
    parser.add_argument("--skip-index", action="store_true")
    parser.add_argument("--skip-save", action="store_true")
    args = parser.parse_args()

    run_scrape(
        urls=args.urls,
        max_pages=args.max_pages,
        place_per=args.place_per,
        save_raw=not args.skip_save,
        rebuild_index=not args.skip_index,
    )


def run_scrape(
    urls: list[str],
    max_pages: int,
    place_per: str | None = None,
    save_raw: bool = True,
    rebuild_index: bool = True,
    on_progress=None,
    raw_output_dir: Path | None = None,
) -> dict[str, int | list[str] | list[SiteResult]]:
    load_local_env()
    results = scrape_urls(urls, max_pages=max_pages, on_progress=on_progress)

    saved_files: list[str] = []
    if save_raw and CONFIG["save_raw_text"]:
        output_dir = raw_output_dir or (ROOT / CONFIG["raw_output_dir"])
        saved_files = [
            str(path)
            for path in save_results_as_txt(results, output_dir=str(output_dir))
        ]

    places = transform_results(results, place_per=place_per)
    inserted = upsert_places(places)

    if rebuild_index and inserted:
        rebuild_vector_index()

    site_count = sum(1 for result in results if not result.error)
    print(f"added {inserted} places from {site_count} sites")

    return {
        "place_count": inserted,
        "site_count": site_count,
        "saved_files": saved_files,
        "results": results,
    }


if __name__ == "__main__":
    scrape()

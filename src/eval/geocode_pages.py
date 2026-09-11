import argparse
from pathlib import Path

from src.db.pages import (
    initialize_page_artifacts_db,
    load_pages_for_geocoding,
    upsert_page_location,
)
from src.shared.env import load_local_env, load_yaml
from src.shared.geocode import NominatimGeocoder, locate_text
from src.shared.llm import LocalOllamaStructuredLlm


__all__ = ["geocode_pages"]

CONFIG = load_yaml(Path(__file__).with_name("config.yaml"))


def geocode_pages(*, force: bool = False) -> None:
    load_local_env()
    initialize_page_artifacts_db()
    llm = LocalOllamaStructuredLlm(CONFIG["geocode_model"], method="function_calling")
    geocoder = NominatimGeocoder()

    pages = load_pages_for_geocoding(force=force)
    max_chars = CONFIG["geocode_max_chars"]
    updated = 0
    for page in pages:
        text = f"{page.title or ''}\n\n{page.markdown}"[:max_chars]
        coords = locate_text(llm, geocoder, text)
        if coords is None:
            continue
        upsert_page_location(page.id, coords.latitude, coords.longitude)
        updated += 1
        print(f"{page.id}: {coords.latitude:.5f}, {coords.longitude:.5f}")

    print(f"geocoded {updated} of {len(pages)} pages")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--force",
        action="store_true",
        help="Re-geocode pages that already have a stored location.",
    )
    args = parser.parse_args()
    geocode_pages(force=args.force)


if __name__ == "__main__":
    main()

import argparse
from pathlib import Path

from src.db.places import load_places, update_place_location
from src.shared.env import load_local_env, load_yaml
from src.shared.geocode import NominatimGeocoder, locate_text
from src.shared.llm import LocalOllamaStructuredLlm


__all__ = ["geocode_places"]

CONFIG = load_yaml(Path(__file__).with_name("config.yaml"))


def geocode_places(*, force: bool = False) -> None:
    """Look up and store coordinates for every place missing them.

    Each place's text is sent to a local LLM to name the one real-world place
    it describes, then that name is geocoded to coordinates via Nominatim.
    Places where no specific place can be identified are left without
    coordinates and simply skipped at query time.
    """
    load_local_env()
    # function_calling, not the json_schema default: gemma4 answers this
    # schema in plain prose under json_schema (see architecture-decisions.md,
    # "Local-Only Inference").
    llm = LocalOllamaStructuredLlm(CONFIG["geocode_model"], method="function_calling")
    geocoder = NominatimGeocoder()

    places = load_places()
    updated = 0
    for place in places:
        if not force and place.latitude is not None and place.longitude is not None:
            continue
        coords = locate_text(llm, geocoder, place.embedding_text)
        if coords is None:
            continue
        update_place_location(place.id, coords.latitude, coords.longitude)
        updated += 1
        print(f"{place.id}: {coords.latitude:.5f}, {coords.longitude:.5f}")

    print(f"geocoded {updated} of {len(places)} places")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--force",
        action="store_true",
        help="Re-geocode places that already have coordinates.",
    )
    args = parser.parse_args()
    geocode_places(force=args.force)


if __name__ == "__main__":
    main()

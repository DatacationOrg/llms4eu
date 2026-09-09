"""Eurostat NUTS 2024 boundaries: a point to its region, a name to its code.

NUTS is the one region hierarchy that is consistent across EU countries, which
is why pages carry `nuts2`/`nuts3` codes rather than national province names.
The codes are derived from coordinates here, so they can be recomputed from the
stored points when Eurostat revises the boundaries.

The GeoJSON is not tracked (16 MB); `just fetch-nuts` downloads it once into
`data/geo/`.
"""

from __future__ import annotations

import argparse
import json
import re
import unicodedata
from dataclasses import dataclass
from functools import cache
from pathlib import Path

import httpx

from src.shared.env import ROOT

__all__ = [
    "NUTS_FILE",
    "NutsIndex",
    "NutsRegion",
    "download_nuts",
    "normalize_place_name",
    "nuts_index",
    "nuts_parents",
]

NUTS_FILE = ROOT / "data" / "geo" / "NUTS_RG_03M_2024_4326.geojson"
NUTS_URL = (
    "https://gisco-services.ec.europa.eu/distribution/v2/nuts/geojson/"
    "NUTS_RG_03M_2024_4326.geojson"
)

# Words a user or a model puts around a region name that the gazetteer does not.
_FILLER = re.compile(
    r"\b(statisticna|statistical|regija|region|regio|pokrajina|province|"
    r"okraj|county|dezela|land)\b"
)


@dataclass(frozen=True)
class NutsRegion:
    code: str
    level: int
    country_code: str
    name: str
    bbox: tuple[float, float, float, float]  # lon_min, lat_min, lon_max, lat_max
    polygons: tuple[tuple[tuple[tuple[float, float], ...], ...], ...]

    def contains(self, latitude: float, longitude: float) -> bool:
        lon_min, lat_min, lon_max, lat_max = self.bbox
        if not (lon_min <= longitude <= lon_max and lat_min <= latitude <= lat_max):
            return False
        for rings in self.polygons:
            outer, *holes = rings
            if _in_ring(longitude, latitude, outer) and not any(
                _in_ring(longitude, latitude, hole) for hole in holes
            ):
                return True
        return False


@dataclass(frozen=True)
class NutsIndex:
    regions: dict[str, NutsRegion]

    def locate(self, latitude: float, longitude: float, level: int = 3) -> str | None:
        """The code of the level-`level` region containing the point, if any."""
        for region in self.regions.values():
            if region.level == level and region.contains(latitude, longitude):
                return region.code
        return None

    def name_of(self, code: str) -> str | None:
        region = self.regions.get(code)
        return region.name if region else None

    def find_by_name(self, name: str, country_code: str | None = None) -> list[str]:
        """Codes whose Latin name matches `name` after normalisation.

        Exact matches first, then prefix matches, then matches on word stems
        (the first five letters of each word), so 'Posavska statistična regija'
        and 'Posavje' both find SI036 and 'Vzhodna Slovenija' finds SI03,
        without a fuzzy-matching library. Slovenian region names inflect
        (Posavska / Posavje / Posavsko), which is what the stem tier absorbs.
        """
        needle = normalize_place_name(name)
        if not needle:
            return []
        exact, partial, stemmed = [], [], []
        needle_stems = _stems(needle)
        for region in self.regions.values():
            if country_code and region.country_code != country_code.upper():
                continue
            candidate = normalize_place_name(region.name)
            if candidate == needle:
                exact.append(region.code)
            elif candidate.startswith(needle) or needle.startswith(candidate):
                partial.append(region.code)
            elif needle_stems and _stems(candidate) == needle_stems:
                stemmed.append(region.code)
        # Most specific level first inside each tier.
        key = lambda code: -len(code)  # noqa: E731
        return (
            sorted(exact, key=key) + sorted(partial, key=key) + sorted(stemmed, key=key)
        )


def nuts_parents(code: str) -> dict[str, str | None]:
    """Every level's code implied by one NUTS code, e.g. SI036 -> SI, SI0, SI03."""
    return {
        "country_code": code[:2] if len(code) >= 2 else None,
        "nuts1": code[:3] if len(code) >= 3 else None,
        "nuts2": code[:4] if len(code) >= 4 else None,
        "nuts3": code if len(code) == 5 else None,
    }


def _stems(normalized: str) -> tuple[str, ...]:
    return tuple(word[:5] for word in normalized.split() if len(word) >= 4)


def normalize_place_name(name: str) -> str:
    folded = unicodedata.normalize("NFKD", name or "")
    ascii_only = "".join(ch for ch in folded if not unicodedata.combining(ch))
    lowered = re.sub(r"[^a-z0-9 ]+", " ", ascii_only.casefold())
    return re.sub(r"\s+", " ", _FILLER.sub(" ", lowered)).strip()


@cache
def nuts_index(path: Path = NUTS_FILE) -> NutsIndex:
    if not path.exists():
        raise RuntimeError(
            f"NUTS boundaries missing at {path}. Run `just fetch-nuts` "
            "(downloads the Eurostat GISCO 2024 GeoJSON once)."
        )
    with path.open(encoding="utf-8") as handle:
        collection = json.load(handle)
    regions = {}
    for feature in collection["features"]:
        properties = feature["properties"]
        polygons = _polygons(feature["geometry"])
        if not polygons:
            continue
        regions[properties["NUTS_ID"]] = NutsRegion(
            code=properties["NUTS_ID"],
            level=int(properties["LEVL_CODE"]),
            country_code=properties["CNTR_CODE"],
            name=properties.get("NAME_LATN") or properties.get("NUTS_NAME") or "",
            bbox=_bbox(polygons),
            polygons=polygons,
        )
    return NutsIndex(regions)


def download_nuts(path: Path = NUTS_FILE, force: bool = False) -> Path:
    if path.exists() and not force:
        return path
    path.parent.mkdir(parents=True, exist_ok=True)
    with httpx.stream(
        "GET", NUTS_URL, timeout=120.0, follow_redirects=True
    ) as response:
        response.raise_for_status()
        with path.open("wb") as handle:
            for chunk in response.iter_bytes():
                handle.write(chunk)
    return path


def _polygons(geometry: dict):
    if geometry["type"] == "Polygon":
        raw = [geometry["coordinates"]]
    elif geometry["type"] == "MultiPolygon":
        raw = geometry["coordinates"]
    else:
        return ()
    return tuple(
        tuple(tuple((float(x), float(y)) for x, y in ring) for ring in polygon)
        for polygon in raw
    )


def _bbox(polygons) -> tuple[float, float, float, float]:
    xs = [x for polygon in polygons for x, _ in polygon[0]]
    ys = [y for polygon in polygons for _, y in polygon[0]]
    return (min(xs), min(ys), max(xs), max(ys))


def _in_ring(x: float, y: float, ring) -> bool:
    """Ray casting over one closed ring."""
    inside = False
    count = len(ring)
    for index in range(count):
        x1, y1 = ring[index]
        x2, y2 = ring[(index + 1) % count]
        if (y1 > y) != (y2 > y):
            crossing = (x2 - x1) * (y - y1) / (y2 - y1) + x1
            if x < crossing:
                inside = not inside
    return inside


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--force", action="store_true", help="Re-download.")
    args = parser.parse_args()
    path = download_nuts(force=args.force)
    index = nuts_index(path)
    print(f"{path}: {len(index.regions)} regions")


if __name__ == "__main__":
    main()

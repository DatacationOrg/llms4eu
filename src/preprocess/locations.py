"""Locate each page: where it is about, as rows in `page_locations`.

Read-only by default. Locations feed the v3 embedding text, every chunk's
Chroma metadata and the geo filters, so the result is reported for review
before `--apply` writes it.

Three tiers, cheapest and most reliable first, each recorded in `method`:

1. `wikidata`: a Wikipedia page's Wikidata item says what it is (P31) and where
   (P625, P605, P300). A person or a concept gets no location, by design.
2. `source_default`: a single-site source (a castle's own website) is about
   that site; configured as a Wikidata QID per source in config.yaml.
3. `llm_nominatim`: a model names the place a page is about and the places it
   mentions; Nominatim resolves each and the hit is kept only if its name
   matches what the model said. The model never produces coordinates.

NUTS codes are then derived from the coordinates against the Eurostat
boundaries, so `--recompute-codes` can refresh them without any lookups.
"""

from __future__ import annotations

import argparse
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from urllib.parse import unquote, urlparse

from pydantic import BaseModel, Field

from src.db.pages import (
    PageLocation,
    connect_pages,
    initialize_page_artifacts_db,
    page_locations,
    replace_page_locations,
)
from src.preprocess.languages import prose_sample
from src.preprocess.rejections import (
    Rejection,
    classify_rejections,
    format_rejections,
    parse_rejections_table,
    wikidata_place_lookup,
)
from src.shared.env import ROOT, load_yaml
from src.shared.geocode import (
    Coordinates,
    GeocodeProvider,
    NominatimGeocoder,
    name_matches,
)
from src.shared.llm import StructuredLlm, run_structured_outputs
from src.shared.nuts import normalize_place_name, nuts_index, nuts_parents
from src.shared.wikidata import WikidataClient, WikidataEntity

CONFIG = load_yaml(Path(__file__).with_name("config.yaml"))
RETRIEVAL_CONFIG = load_yaml(ROOT / "src" / "retrieval" / "config.yaml")

SAMPLE_CHARS = 4000
MAX_MENTIONED = 5


class PagePlaces(BaseModel):
    """What a model reads off one page."""

    primary_place: str | None = Field(
        default=None,
        description=(
            "The one real-world place this page is about, when the page is about a "
            "place (a castle, church, town, region, natural site, venue). Null for a "
            "person, an organisation, an event, a concept, or a listing of many "
            "places. Written so a gazetteer finds it, e.g. 'Grad Rajhenburg, "
            "Brestanica, Slovenia'."
        ),
    )
    mentioned_places: list[str] = Field(
        default_factory=list,
        description=(
            "Up to five other specific, geocodable places central to the page: the "
            "castles a listing covers, the town a person's life is tied to. Not "
            "countries, not passing mentions."
        ),
    )


@dataclass(frozen=True)
class PageForLocating:
    id: str
    source: str
    url: str
    title: str | None
    markdown: str


@dataclass
class LocateSummary:
    pages: int = 0
    located: int = 0
    by_method: Counter = field(default_factory=Counter)
    mentioned: int = 0
    rejected: Counter = field(default_factory=Counter)
    llm_failures: int = 0
    unlocated: list[PageForLocating] = field(default_factory=list)
    primaries: dict[str, PageLocation] = field(default_factory=dict)
    # Every model-produced name the gazetteer refused, with the page that named
    # it, so the misses can be classified rather than only counted.
    rejections: list[Rejection] = field(default_factory=list)


def load_pages() -> list[PageForLocating]:
    initialize_page_artifacts_db()
    with connect_pages() as conn:
        rows = conn.execute(
            """
            select m.id, m.source, m.url, m.title, c.markdown
            from page_metadata m
            join page_markdown_content c on c.page_id = m.id
            where m.page_kind != 'empty' and m.error is null
            order by m.source, m.id
            """
        ).fetchall()
    return [PageForLocating(*row) for row in rows]


def locate_pages(
    pages: list[PageForLocating],
    *,
    wikidata: WikidataClient | None,
    geocoder: GeocodeProvider | None,
    llm: StructuredLlm | None,
    workers: int = 2,
) -> tuple[dict[str, list[PageLocation]], LocateSummary]:
    """Every page's location set, without writing anything."""
    summary = LocateSummary(pages=len(pages))
    located: dict[str, list[PageLocation]] = {page.id: [] for page in pages}
    wikidata_settled: set[str] = set()

    if wikidata is not None:
        _tier_wikidata(pages, wikidata, located, wikidata_settled, summary)
        _tier_source_default(pages, wikidata, located, summary)
    if llm is not None:
        _tier_llm(pages, llm, geocoder, located, wikidata_settled, summary, workers)

    for page in pages:
        primary = next((loc for loc in located[page.id] if loc.role == "primary"), None)
        if primary is None:
            summary.unlocated.append(page)
        else:
            summary.located += 1
            summary.by_method[primary.method] += 1
            summary.primaries[page.id] = primary
        summary.mentioned += sum(
            1 for loc in located[page.id] if loc.role == "mentioned"
        )
    return located, summary


def _tier_wikidata(pages, wikidata, located, settled, summary) -> None:
    by_language: dict[str, dict[str, PageForLocating]] = {}
    for page in pages:
        language, title = _wikipedia_title(page.url)
        if title:
            by_language.setdefault(language, {})[title] = page
    for language, titled in by_language.items():
        qids = wikidata.qids_for_wikipedia_titles(list(titled), language=language)
        entities = wikidata.entities(list(qids.values()))
        for title, page in titled.items():
            entity = entities.get(qids.get(title, ""))
            if entity is None:
                continue
            settled.add(page.id)
            location = _location_from_entity(page.id, entity, "wikidata", 1.0)
            if location is None:
                summary.rejected["wikidata: not a place"] += 1
                continue
            located[page.id].append(location)


def _tier_source_default(pages, wikidata, located, summary) -> None:
    defaults = CONFIG.get("source_locations") or {}
    if not defaults:
        return
    entities = wikidata.entities([entry["wikidata_qid"] for entry in defaults.values()])
    for page in pages:
        entry = defaults.get(page.source)
        if entry is None or any(loc.role == "primary" for loc in located[page.id]):
            continue
        entity = entities.get(entry["wikidata_qid"])
        if entity is None:
            summary.rejected["source_default: unknown QID"] += 1
            continue
        location = _location_from_entity(page.id, entity, "source_default", 0.9)
        if location is not None:
            if entry.get("name"):
                location = PageLocation(**{**location.__dict__, "name": entry["name"]})
            located[page.id].append(location)


def _tier_llm(pages, llm, geocoder, located, settled, summary, workers) -> None:
    tolerant = _Tolerant(llm)
    prompts = [page_places_prompt(page) for page in pages]
    answers = run_structured_outputs(tolerant, prompts, PagePlaces, workers=workers)
    summary.llm_failures = len(tolerant.failures)
    resolved: dict[str, PageLocation | None] = {}
    reasons: dict[str, str] = {}

    def resolve(name: str, role: str, page_id: str) -> PageLocation | None:
        key = normalize_place_name(name)
        if not key:
            return None
        if key not in resolved:
            resolved[key], reason = _geocode(name, geocoder, summary)
            if reason:
                reasons[key] = reason
        template = resolved[key]
        if template is None:
            if key in reasons:
                summary.rejections.append(Rejection(name, reasons[key], page_id, role))
            return None
        return PageLocation(**{**template.__dict__, "page_id": page_id, "role": role})

    for page, answer in zip(pages, answers):
        rows = located[page.id]
        has_primary = any(loc.role == "primary" for loc in rows)
        primary_name = (answer.primary_place or "").strip()
        if primary_name and not has_primary and page.id not in settled:
            location = resolve(primary_name, "primary", page.id)
            if location is not None:
                rows.append(location)
                has_primary = True
        elif primary_name and page.id in settled and not has_primary:
            summary.rejected["llm: primary overruled by wikidata class"] += 1
        keys = {loc.location_key for loc in rows}
        for name in answer.mentioned_places[:MAX_MENTIONED]:
            location = resolve(name.strip(), "mentioned", page.id)
            if location is None or location.location_key in keys:
                continue
            keys.add(location.location_key)
            rows.append(location)


def _geocode(name: str, geocoder, summary) -> tuple[PageLocation | None, str | None]:
    """The located template, or None plus the reason the gazetteer refused."""
    if geocoder is None:
        return None, None
    hit = geocoder.geocode(name)
    if hit is None:
        summary.rejected["nominatim: no hit"] += 1
        return None, "nominatim: no hit"
    if not name_matches(name, hit.display_name):
        summary.rejected["nominatim: name mismatch"] += 1
        return None, "nominatim: name mismatch"
    return _located(
        page_id="",
        role="mentioned",
        key=f"name:{normalize_place_name(name)}",
        name=name,
        qid=None,
        coordinates=hit.coordinates,
        granularity="point",
        country_code=hit.country_code,
        iso_3166_2=None,
        method="llm_nominatim",
        confidence=min(1.0, 0.5 + hit.importance),
    ), None


def _location_from_entity(
    page_id: str, entity: WikidataEntity, method: str, confidence: float
) -> PageLocation | None:
    granularity = entity.granularity
    if granularity is None:
        return None
    codes = None
    if entity.nuts_codes:
        codes = nuts_parents(max(entity.nuts_codes, key=len))
    return _located(
        page_id=page_id,
        role="primary",
        key=entity.qid,
        name=entity.label,
        qid=entity.qid,
        coordinates=entity.coordinates,
        granularity=granularity,
        country_code=None,
        iso_3166_2=entity.iso_3166_2,
        method=method,
        confidence=confidence,
        codes=codes,
    )


def _located(
    *,
    page_id: str,
    role: str,
    key: str,
    name: str | None,
    qid: str | None,
    coordinates: Coordinates | None,
    granularity: str,
    country_code: str | None,
    iso_3166_2: str | None,
    method: str,
    confidence: float,
    codes: dict | None = None,
) -> PageLocation:
    """Fill the NUTS hierarchy from explicit codes or from the point.

    A region keeps only the levels it genuinely is: a country has no NUTS-3, a
    traditional region spanning several NUTS-3 units keeps NUTS-2 from its
    centroid and no NUTS-3, so a NUTS-3 filter never claims a whole region.
    """
    index = nuts_index()
    if codes is None and coordinates is not None:
        nuts3 = index.locate(coordinates.latitude, coordinates.longitude)
        codes = nuts_parents(nuts3) if nuts3 else {}
    codes = codes or {}
    if granularity == "country":
        codes = {"country_code": codes.get("country_code")}
    elif granularity == "region" and codes.get("nuts3"):
        matched = name and codes["nuts3"] in index.find_by_name(
            name, codes.get("country_code")
        )
        if not matched:
            codes = {**codes, "nuts3": None}
    nuts3 = codes.get("nuts3")
    return PageLocation(
        page_id=page_id,
        role=role,
        location_key=key,
        name=name,
        wikidata_qid=qid,
        latitude=coordinates.latitude if coordinates else None,
        longitude=coordinates.longitude if coordinates else None,
        granularity=granularity,
        country_code=codes.get("country_code") or (country_code or None),
        nuts2=codes.get("nuts2"),
        nuts3=nuts3,
        nuts3_name=index.name_of(nuts3) if nuts3 else None,
        iso_3166_2=iso_3166_2,
        confidence=confidence,
        method=method,
    )


def _wikipedia_title(url: str) -> tuple[str, str | None]:
    parsed = urlparse(url)
    host = parsed.netloc
    if not host.endswith("wikipedia.org") or "/wiki/" not in parsed.path:
        return "", None
    language = host.split(".")[0]
    title = unquote(parsed.path.split("/wiki/", 1)[1]).replace("_", " ")
    return language, title or None


def page_places_prompt(page: PageForLocating) -> str:
    return (
        "Read this page and name the place it is about, if it is about a place.\n"
        "Rules: a biography, a club, an organisation, an event, a concept or a "
        "time zone is not about a place, so primary_place is null; a page about a "
        "castle, church, town, region or natural site is, so primary_place names "
        "it. mentioned_places lists up to five other specific places central to "
        "the page (for a biography: the town the person is tied to).\n\n"
        f"title: {page.title or ''}\n\n{prose_sample(page.markdown, SAMPLE_CHARS)}"
    )


@dataclass
class _Tolerant:
    inner: StructuredLlm
    failures: list[str] = field(default_factory=list)

    def structured_output(self, prompt, output_schema, *, retries: int = 3):
        try:
            return self.inner.structured_output(prompt, output_schema, retries=retries)
        except Exception as error:
            self.failures.append(f"{type(error).__name__}: {error}")
            return output_schema()


def recompute_codes() -> int:
    """Refresh NUTS codes from stored coordinates, e.g. after a NUTS revision."""
    updated = 0
    for page_id, rows in page_locations().items():
        refreshed = [
            _located(
                page_id=row.page_id,
                role=row.role,
                key=row.location_key,
                name=row.name,
                qid=row.wikidata_qid,
                coordinates=row.coordinates,
                granularity=row.granularity,
                country_code=row.country_code,
                iso_3166_2=row.iso_3166_2,
                method=row.method,
                confidence=row.confidence or 0.0,
                codes=(
                    {"country_code": row.country_code}
                    if row.granularity == "country"
                    else None
                ),
            )
            for row in rows
        ]
        replace_page_locations(page_id, refreshed)
        updated += len(refreshed)
    return updated


def copy_locations(source: Path) -> int:
    """Copy another database's page_locations into this one, page by page.

    Only pages present here receive rows, so a sweep snapshot cut from an older
    page set cannot gain locations for pages it does not hold.
    """
    import sqlite3

    source_path = source if source.is_absolute() else ROOT / source
    initialize_page_artifacts_db()
    with sqlite3.connect(f"file:{source_path}?mode=ro", uri=True) as conn:
        conn.row_factory = sqlite3.Row
        rows = conn.execute("select * from page_locations").fetchall()
    with connect_pages() as conn:
        known = {row["id"] for row in conn.execute("select id from page_metadata")}
    by_page: dict[str, list[PageLocation]] = {}
    for row in rows:
        if row["page_id"] in known:
            by_page.setdefault(row["page_id"], []).append(PageLocation(**dict(row)))
    return apply_locations(by_page)


def apply_locations(located: dict[str, list[PageLocation]]) -> int:
    written = 0
    for page_id, rows in located.items():
        replace_page_locations(page_id, rows)
        written += len(rows)
    return written


def format_report(
    summary: LocateSummary, located: dict[str, list[PageLocation]]
) -> str:
    lines = [
        f"pages scanned: {summary.pages}",
        f"pages with a primary location: {summary.located} "
        f"({', '.join(f'{k}={v}' for k, v in summary.by_method.most_common()) or 'none'})",
        f"mentioned locations: {summary.mentioned}",
        f"pages without a location: {len(summary.unlocated)}",
        f"llm failures: {summary.llm_failures}",
    ]
    if summary.rejected:
        lines.append(
            "rejected: "
            + ", ".join(f"{k}={v}" for k, v in summary.rejected.most_common())
        )
    granularity = Counter(loc.granularity for loc in summary.primaries.values())
    nuts3 = Counter(loc.nuts3 or "-" for loc in summary.primaries.values())
    lines.append(f"primary granularity: {dict(granularity)}")
    lines.append(f"primary NUTS-3: {dict(nuts3.most_common())}")
    lines.append("")
    lines.append("| Page | Method | Primary | Gran. | NUTS-3 | Mentioned |")
    lines.append("|---|---|---|---|---|---|")
    for page_id, rows in sorted(located.items(), key=lambda item: item[0]):
        primary = next((r for r in rows if r.role == "primary"), None)
        mentioned = ", ".join((r.name or "?") for r in rows if r.role == "mentioned")
        lines.append(
            f"| {page_id[:8]} | {primary.method if primary else '-'} | "
            f"{(primary.name or '') if primary else '-'} | "
            f"{primary.granularity if primary else '-'} | "
            f"{(primary.nuts3 or primary.nuts2 or primary.country_code or '') if primary else '-'} | "
            f"{mentioned[:80]} |"
        )
    if summary.unlocated:
        lines.append("")
        lines.append("Unlocated pages:")
        for page in summary.unlocated:
            lines.append(f"  - {page.source}: {(page.title or page.url)[:70]}")
    return "\n".join(lines)


def page_locations_flat() -> list[PageLocation]:
    """Stored primary rows, for the known-names tier of the classifier.

    Primaries only: the gazetteer's corpus tier reads `primary_locations`, and
    the mentioned rows are themselves Nominatim output whose correctness the
    classification is meant to question.
    """
    return [
        row
        for rows in page_locations().values()
        for row in rows
        if row.role == "primary"
    ]


def classify_summary_rejections(
    summary: LocateSummary,
    located: dict[str, list[PageLocation]],
    *,
    geocoder: NominatimGeocoder | None,
    wikidata: WikidataClient | None,
) -> str:
    """Classify every refused name by the cheapest fix that resolves it (WP4 gate)."""
    known = [
        loc.name
        for rows in located.values()
        for loc in rows
        if loc.name and loc.role == "primary"
    ]
    known += [region.name for region in nuts_index().regions.values()]
    classified = classify_rejections(
        summary.rejections,
        geocode_hinted=(lambda q: geocoder.geocode(q)) if geocoder else None,
        geocode_unhinted=(
            (lambda q: geocoder.geocode(q, country_codes=None)) if geocoder else None
        ),
        wikidata_lookup=wikidata_place_lookup(wikidata) if wikidata else None,
        known_names=known,
    )
    return format_rejections(classified)


def _llm() -> StructuredLlm:
    from src.retrieval.methods import _judge_config
    from src.retrieval.retrievers.agentic import default_judge

    return default_judge(
        _judge_config(
            {
                "agentic_judge_provider": CONFIG.get(
                    "locate_provider", RETRIEVAL_CONFIG["geo_resolver_provider"]
                ),
                "agentic_judge_structured_method": RETRIEVAL_CONFIG[
                    "geo_structured_method"
                ],
                "agentic_judge_num_predict": 512,
            }
        )
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--apply", action="store_true", help="Write page_locations.")
    parser.add_argument("--limit", type=int, help="Only the first N pages.")
    parser.add_argument("--source", help="Only pages of this source.")
    parser.add_argument("--no-llm", action="store_true", help="Skip tier 3.")
    parser.add_argument("--no-wikidata", action="store_true", help="Skip tiers 1-2.")
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument(
        "--rejections",
        metavar="PATH",
        help=(
            "Classify every place name the gazetteer refused (qualifier, "
            "inflected, outside the country hint, Wikidata-only, unresolved) and "
            "write the table to PATH. Retries each miss a few times against "
            "Nominatim and Wikidata; read-only."
        ),
    )
    parser.add_argument(
        "--reclassify",
        metavar="TABLE",
        help=(
            "Re-run the rejection classification from a table written by "
            "--rejections instead of locating again (no LLM calls). Writes to "
            "--rejections PATH, or overwrites TABLE when PATH is not given."
        ),
    )
    parser.add_argument(
        "--recompute-codes",
        action="store_true",
        help="Only refresh NUTS codes of stored rows from their coordinates.",
    )
    parser.add_argument(
        "--copy-from",
        metavar="DB",
        help=(
            "Copy page_locations from another pages database (e.g. data/db/pages.db) "
            "into PAGES_DB_PATH instead of locating. Page ids are uuid5 of the URL, "
            "so a sweep snapshot shares them with the durable database."
        ),
    )
    args = parser.parse_args()

    if args.recompute_codes:
        print(f"recomputed codes for {recompute_codes()} rows")
        return
    if args.reclassify:
        source = Path(args.reclassify)
        summary = LocateSummary()
        summary.rejections = parse_rejections_table(source.read_text(encoding="utf-8"))
        hint = RETRIEVAL_CONFIG.get("geo_country_hint") or None
        report = classify_summary_rejections(
            summary,
            {"": list(page_locations_flat())},
            geocoder=NominatimGeocoder(country_codes=hint),
            wikidata=WikidataClient(),
        )
        target = Path(args.rejections) if args.rejections else source
        target.write_text(report + "\n", encoding="utf-8")
        print("\n".join(report.splitlines()[:2]))
        print(f"rejections written to {target}")
        return
    if args.copy_from:
        print(f"copied {copy_locations(Path(args.copy_from))} location rows")
        return

    pages = load_pages()
    if args.source:
        pages = [page for page in pages if page.source == args.source]
    if args.limit:
        pages = pages[: args.limit]
    hint = RETRIEVAL_CONFIG.get("geo_country_hint") or None
    wikidata = None if args.no_wikidata else WikidataClient()
    geocoder = None if args.no_llm else NominatimGeocoder(country_codes=hint)
    located, summary = locate_pages(
        pages,
        wikidata=wikidata,
        geocoder=geocoder,
        llm=None if args.no_llm else _llm(),
        workers=args.workers,
    )
    print(format_report(summary, located))
    if args.rejections:
        report = classify_summary_rejections(
            summary, located, geocoder=geocoder, wikidata=wikidata or WikidataClient()
        )
        path = Path(args.rejections)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(report + "\n", encoding="utf-8")
        print(f"\n{report.splitlines()[0]}\n{report.splitlines()[1]}")
        print(f"rejections written to {path}")
    if not args.apply:
        print("\nread-only: re-run with --apply to write page_locations")
        return
    print(f"\nwrote {apply_locations(located)} location rows for {len(located)} pages")


if __name__ == "__main__":
    main()

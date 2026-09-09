"""Why a place name the model produced did not geocode, and whether a cheap retry would.

`just locate-pages` used to report only a count of Nominatim misses (141 on
2026-09-08). This module keeps the names, then classifies each miss by trying
the cheapest fixes in order and recording the first that resolves:

- `corpus`: the head of the name is a place the corpus already holds as a
  primary location or a NUTS region name ("Rajhenburg" when `Grad Rajhenburg`
  is a primary); no lookup was needed, the corpus-names tier of the gazetteer
  should have caught it. Mentioned rows are not consulted: they are Nominatim
  output themselves, and the second pass on 2026-09-09 showed them matching
  "Gradec, Austria" to a Slovenian village row.
- `qualifier`: the name carried a type word or a trailing qualifier the
  gazetteer does not index ("Grad Sevnica", "cerkev sv. Petra, Brestanica");
  stripping it resolves. A hit counts only when it names the head of the
  original ("Gradec, Austria" must resolve to Gradec, not to Austria).
- `inflected`: a Slovenian oblique form ("v Brestanici", "Celju", "Krškem");
  a nominative candidate resolves or matches a name the corpus already holds.
- `outside_hint`: the name resolves once the `geo_country_hint` restriction is
  lifted (Trst, Gradec, Dunaj: places abroad that Slovenian pages name in
  Slovenian).
- `wikidata`: Nominatim has nothing but a Slovenian-language Wikidata search
  finds an item with coordinates (Emona, Celeja: historical places; Dunaj,
  Budimpešta: Slovenian exonyms whose English label shares no token).
- `unresolved`: none of the above.

The categories are the acceptance test for the gazetteer work in
docs/architecture/geo-improvement-plan.md (WP4): each one names the fix that
would have recovered the footprint. Network calls are bounded per distinct name
and made only for the misses.
"""

from __future__ import annotations

import re
from collections import Counter
from collections.abc import Callable, Iterable
from dataclasses import dataclass

from src.shared.geocode import Coordinates, GeocodeHit, name_matches
from src.shared.nuts import normalize_place_name

__all__ = [
    "ClassifiedRejection",
    "Rejection",
    "classify_rejections",
    "deinflect",
    "format_rejections",
    "head_segment",
    "inflected_candidates",
    "parse_rejections_table",
    "retry_candidates",
    "wikidata_place_lookup",
]

# Type words that precede a proper name and that Nominatim rarely indexes as
# part of the name. Normalised (ASCII, lower case) like `normalize_place_name`.
_TYPE_WORDS = (
    "grad|dvorec|gradic|trdnjava|cerkev|zupnijska cerkev|bazilika|kapela|"
    "zupnija|samostan|opatija|muzej|galerija|sola|osnovna sola|gimnazija|"
    "univerza|fakulteta|bolnica|bolnisnica|zdravilisce|tovarna|elektrarna|"
    "termoelektrarna|zelezniska postaja|postaja|trg|ulica|cesta|most|jama|"
    "dolina|gora|hrib|planina|reka|jezero|slap|otok|obcina|mesto|naselje|"
    "vas|kraj|pokrajina|regija|okolica|castle|church|monastery|museum|town|"
    "village|river|mountain|valley|municipality|region"
)
_LEADING_TYPE = re.compile(rf"^(?:{_TYPE_WORDS})\b\s*(?:sv\s+|svetega\s+|svete\s+)?")
_TRAILING_TYPE = re.compile(rf"\s+\b(?:{_TYPE_WORDS})$")
_COUNTRY_TAIL = re.compile(
    r"\s*,\s*(?:slovenija|slovenia|slowenien|si)\s*$", re.IGNORECASE
)
_PARENS = re.compile(r"\s*\([^)]*\)")
_PREPOSITION = re.compile(
    r"^(?:v|na|pri|ob|iz|do|pod|nad|blizu|okoli)\s+", re.IGNORECASE
)

# Slovenian nominal endings (oblique form -> nominative candidates). Ordered by
# specificity; longer endings first so "Krškem" is tried as -em before -m.
_ENDINGS: tuple[tuple[str, tuple[str, ...]], ...] = (
    ("ami", ("a", "e")),  # Brežicami -> Brežice
    ("ah", ("e", "a")),  # Brežicah -> Brežice
    ("ih", ("i", "e")),  # Kostanjevih...
    ("em", ("o", "e", "")),  # Krškem -> Krško, Celjem -> Celje
    ("om", ("", "o")),  # Mariborom -> Maribor
    ("ju", ("j", "je", "")),  # Celju -> Celje, Bohorju -> Bohor
    ("ja", ("j", "je")),  # Celja -> Celje
    ("ci", ("ca",)),  # Brestanici -> Brestanica, Sevnici -> Sevnica
    ("ce", ("ca",)),  # Brestanice -> Brestanica
    ("co", ("ca",)),  # Brestanico -> Brestanica
    ("i", ("a", "", "e")),  # Ljubljani -> Ljubljana, Ptuji -> Ptuj
    ("u", ("", "o", "e")),  # Trstu -> Trst, Krškemu...
    ("e", ("a", "o")),  # Ljubljane -> Ljubljana, Krške -> Krško
    ("a", ("", "o")),  # Maribora -> Maribor, Krška -> Krško
)


@dataclass(frozen=True)
class Rejection:
    name: str
    reason: str  # "nominatim: no hit" | "nominatim: name mismatch"
    page_id: str
    role: str  # "primary" | "mentioned"


@dataclass(frozen=True)
class ClassifiedRejection:
    name: str
    reason: str
    category: str  # qualifier | inflected | outside_hint | wikidata | unresolved
    resolved_as: str | None  # the query or label that resolved
    coordinates: Coordinates | None
    pages: int
    roles: tuple[str, ...]


def retry_candidates(name: str) -> list[str]:
    """Simpler spellings of a name, most specific first, original excluded.

    'Grad Sevnica' -> ['Sevnica']; 'cerkev sv. Petra, Brestanica, Slovenija' ->
    ['sv. Petra, Brestanica', 'Brestanica']; 'v Brestanici' -> ['Brestanici'].
    """
    seen: list[str] = []

    def add(candidate: str) -> None:
        candidate = re.sub(r"\s+", " ", candidate).strip(" ,")
        if candidate and candidate.casefold() != name.casefold():
            if candidate.casefold() not in {c.casefold() for c in seen}:
                seen.append(candidate)

    base = _COUNTRY_TAIL.sub("", _PARENS.sub("", name or "")).strip()
    base = _PREPOSITION.sub("", base)
    add(base)
    segments = [segment.strip() for segment in base.split(",") if segment.strip()]
    stripped_first = _strip_type_words(segments[0]) if segments else ""
    if stripped_first:
        add(", ".join([stripped_first, *segments[1:]]))
        add(stripped_first)
    for start in range(1, len(segments)):
        add(", ".join(segments[start:]))
    if segments:
        add(segments[-1])
    return seen


def _strip_type_words(segment: str) -> str:
    normalized = normalize_place_name(segment)
    if not normalized:
        return ""
    without = _LEADING_TYPE.sub("", normalized)
    without = _TRAILING_TYPE.sub("", without)
    if without == normalized:
        return ""
    # Map back onto the original casing by dropping as many leading/trailing
    # words as normalisation removed.
    words = segment.split()
    dropped_front = len(normalized.split()) - len(without.split())
    if _LEADING_TYPE.match(normalized):
        return " ".join(words[dropped_front:]).strip(" ,.")
    return " ".join(words[: len(words) - dropped_front]).strip(" ,.")


def deinflect(word: str) -> list[str]:
    """Nominative candidates for one Slovenian oblique word form.

    Candidates only; the caller checks them against known names or a gazetteer.
    Words under four letters are returned unchanged.
    """
    if len(word) < 4:
        return []
    lowered = word.casefold()
    out: list[str] = []
    for ending, replacements in _ENDINGS:
        if lowered.endswith(ending) and len(lowered) - len(ending) >= 3:
            stem = word[: len(word) - len(ending)]
            for replacement in replacements:
                candidate = stem + replacement
                if candidate.casefold() != lowered and candidate not in out:
                    out.append(candidate)
    return out


def inflected_candidates(name: str) -> list[str]:
    """Deinflect the last word of a name (the head noun in Slovenian toponyms)."""
    words = (_PREPOSITION.sub("", (name or "").strip())).split()
    if not words:
        return []
    *head, last = words
    return [" ".join([*head, form]) for form in deinflect(last)]


Geocode = Callable[[str], GeocodeHit | None]
WikidataLookup = Callable[[str], tuple[str, Coordinates] | None]


def classify_rejections(
    rejections: Iterable[Rejection],
    *,
    geocode_hinted: Geocode | None,
    geocode_unhinted: Geocode | None,
    wikidata_lookup: WikidataLookup | None,
    known_names: Iterable[str] = (),
    max_retries: int = 3,
) -> list[ClassifiedRejection]:
    """One row per distinct rejected name, with the cheapest fix that resolves it."""
    grouped: dict[str, list[Rejection]] = {}
    for rejection in rejections:
        grouped.setdefault(normalize_place_name(rejection.name), []).append(rejection)
    known = {normalize_place_name(n) for n in known_names if n}
    out: list[ClassifiedRejection] = []
    for rows in grouped.values():
        first = rows[0]
        category, resolved_as, coordinates = _classify_one(
            first.name,
            geocode_hinted=geocode_hinted,
            geocode_unhinted=geocode_unhinted,
            wikidata_lookup=wikidata_lookup,
            known=known,
            max_retries=max_retries,
        )
        out.append(
            ClassifiedRejection(
                name=first.name,
                reason=first.reason,
                category=category,
                resolved_as=resolved_as,
                coordinates=coordinates,
                pages=len({r.page_id for r in rows}),
                roles=tuple(sorted({r.role for r in rows})),
            )
        )
    order = {
        "corpus": 0,
        "qualifier": 1,
        "inflected": 2,
        "outside_hint": 3,
        "wikidata": 4,
    }
    out.sort(key=lambda r: (order.get(r.category, 9), -r.pages, r.name.casefold()))
    return out


def head_segment(name: str) -> str:
    """The place the name is about: first comma segment, type words stripped.

    'Gradec, Austria' -> 'Gradec'; 'Grad Sevnica' -> 'Sevnica'; 'v Brestanici'
    -> 'Brestanici'. A retry hit must name this, not a qualifier.
    """
    base = _COUNTRY_TAIL.sub("", _PARENS.sub("", name or "")).strip()
    base = _PREPOSITION.sub("", base)
    first = base.split(",")[0].strip()
    return _strip_type_words(first) or first


def _classify_one(
    name: str,
    *,
    geocode_hinted: Geocode | None,
    geocode_unhinted: Geocode | None,
    wikidata_lookup: WikidataLookup | None,
    known: set[str],
    max_retries: int,
) -> tuple[str, str | None, Coordinates | None]:
    head = head_segment(name)

    def verified(candidate: str, geocode: Geocode | None) -> GeocodeHit | None:
        """A hit that names the head of the original, not merely the candidate."""
        if geocode is None:
            return None
        hit = geocode(candidate)
        if hit is None:
            return None
        if name_matches(head, hit.display_name) and name_matches(
            candidate, hit.display_name
        ):
            return hit
        return None

    known_match = _known_name_for(head, known)
    if known_match is not None:
        return "corpus", known_match, None
    for candidate in retry_candidates(name)[:max_retries]:
        hit = verified(candidate, geocode_hinted)
        if hit is not None:
            return "qualifier", candidate, hit.coordinates
    for candidate in inflected_candidates(head)[:max_retries]:
        known_match = _known_name_for(candidate, known)
        if known_match is not None:
            return "inflected", known_match, None
        hit = verified(candidate, geocode_hinted)
        if hit is not None:
            return "inflected", candidate, hit.coordinates
    for candidate in [name, *retry_candidates(name)[:1]]:
        hit = verified(candidate, geocode_unhinted)
        if hit is not None:
            return "outside_hint", hit.display_name, hit.coordinates
    if wikidata_lookup is not None:
        found = wikidata_lookup(name)
        if found is not None:
            label, coordinates = found
            return "wikidata", label, coordinates
    return "unresolved", None, None


def _known_name_for(candidate: str, known: set[str]) -> str | None:
    """A known name equal to the candidate, or containing all its tokens.

    'rajhenburg' is found in 'grad rajhenburg'; 'sevnica' is not found in
    'sevniska ravan'. Tokens under three characters do not count.
    """
    normalized = normalize_place_name(candidate)
    if not normalized:
        return None
    if normalized in known:
        return normalized
    tokens = {token for token in normalized.split() if len(token) >= 3}
    for name in known:
        name_tokens = set(name.split())
        # One shared token is not evidence ("Stara vas" is not "Stara Zagora"):
        # the candidate must be the whole of the known name's proper-name part,
        # or share at least two tokens with it.
        if tokens and tokens <= name_tokens:
            proper = {token for token in name_tokens if len(token) >= 3}
            if len(tokens) >= 2 or tokens == proper or _drop_types(proper) == tokens:
                return name
    return None


def _drop_types(tokens: set[str]) -> set[str]:
    type_words = set(_TYPE_WORDS.split("|"))
    return {token for token in tokens if token not in type_words}


def parse_rejections_table(text: str) -> list[Rejection]:
    """Rejections back out of a `format_rejections` table, for reclassifying.

    Page ids are synthetic (`row-N`), one per counted page, so page counts
    survive a round trip; the original ids are not in the table.
    """
    out: list[Rejection] = []
    for index, line in enumerate(text.splitlines()):
        if not line.startswith("| ") or line.startswith("| Name") or "---" in line:
            continue
        cells = [cell.strip() for cell in line.strip().strip("|").split("|")]
        if len(cells) < 6 or not cells[4].isdigit():
            continue
        name, reason, pages, roles = cells[0], cells[1], int(cells[4]), cells[5]
        role_list = roles.split("/") if roles else ["mentioned"]
        for page in range(pages):
            out.append(
                Rejection(
                    name=name,
                    reason=f"nominatim: {reason}",
                    page_id=f"row-{index}-{page}",
                    role=role_list[min(page, len(role_list) - 1)],
                )
            )
    return out


def format_rejections(classified: list[ClassifiedRejection]) -> str:
    counts = Counter(row.category for row in classified)
    mentions = sum(row.pages for row in classified)
    lines = [
        f"distinct rejected names: {len(classified)} (over {mentions} page mentions)",
        "by category: "
        + ", ".join(
            f"{category}={counts.get(category, 0)}"
            for category in (
                "qualifier",
                "inflected",
                "outside_hint",
                "wikidata",
                "unresolved",
            )
        ),
        "",
        "| Name | Reason | Category | Resolves as | Pages | Roles |",
        "|---|---|---|---|---|---|",
    ]
    for row in classified:
        lines.append(
            f"| {row.name} | {row.reason.split(': ', 1)[-1]} | {row.category} | "
            f"{row.resolved_as or ''} | {row.pages} | {'/'.join(row.roles)} |"
        )
    return "\n".join(lines)


def wikidata_place_lookup(
    client, languages: tuple[str, ...] = ("sl", "en")
) -> WikidataLookup:
    """A `WikidataLookup` over `WikidataClient`: first non-human item with coordinates."""

    def lookup(name: str) -> tuple[str, Coordinates] | None:
        query = head_segment(name) or name
        try:
            for language in languages:
                qids = client.search(query, language=language, limit=1)
                if not qids:
                    continue
                for entity in client.entities(qids).values():
                    if entity.is_human or entity.coordinates is None:
                        continue
                    # A Slovenian-language search already matched the Slovenian
                    # name (Dunaj -> Vienna); the English label need not share a
                    # token. An English search is verified as before.
                    exonym = language == "sl"
                    if (
                        not exonym
                        and entity.label
                        and not name_matches(query, entity.label)
                    ):
                        continue
                    return (entity.label or entity.qid), entity.coordinates
        except Exception as exc:  # network; the row stays unresolved
            print(f"wikidata lookup failed for {name!r}: {exc}", flush=True)
        return None

    return lookup

"""Build seed files that mirror the Brestanica corpus for other localities.

`data/brestanica.json` is one small heritage locality seen from four kinds of
source: the attraction's own website (`castle_rajhenburg`), the town's website
(`brestanica_webpage`), the national biographical lexicon's entries for people
from the area (`svn_biography`), and the national-language Wikipedia articles
the town article links to (`wikipedia`). `seeds.yaml` declares the same four
for a locality in another country; this module discovers the URLs and writes
one `{source, url, language}` seed file per locality, in the format
`fetch_pages.py` reads.

Discovery per kind:

- **site**: the sitemaps named in robots.txt or at the usual paths, page
  sitemaps before post sitemaps; when a site has none, a two-level crawl of
  its own links from the root.
- **wikipedia**: the wikilinks of each seed article in reading order, plus the
  members of any listed category, so the set holds the town, its region, its
  history and its people the way the Brestanica set does.
- **biography**: people born in the listed places (Wikidata P19, walked up
  P131) that have an entry in the country's lexicon, most-linked first, with
  the entry URL built from the lexicon property's formatter.

Every URL is fetched once and kept only if it answers 200 with a text body, so
a seed file never carries dead links into the page database.
"""

from __future__ import annotations

import argparse
import json
import re
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path
from urllib.parse import quote, urljoin, urlparse

import httpx
from trafilatura import extract as extract_text

from src.scraping.page_fetch import DomainThrottle
from src.shared.env import ROOT, load_yaml

SPEC_PATH = Path(__file__).with_name("seeds.yaml")
# The reference corpus. Its URLs are kept out of every new seed: a page id is
# derived from its URL, so re-fetching one under a new source would move the
# gold-standard page out of its source and refresh its text under the labels.
REFERENCE_SEED = ROOT / "data" / "brestanica.json"
USER_AGENT = "llms4eu-seed-builder/0.1 (tourism RAG research corpus)"
TIMEOUT = 25.0
# A P131* walk over a whole district can take Wikidata's query service a while.
SPARQL_TIMEOUT = 120.0
DOMAIN_DELAY = 1.0

# A page whose extracted text is shorter than this is a stub (a lexicon index
# record, a photo gallery, an app link) and is rejected at seed time.
MIN_TEXT_CHARS = 300
SITEMAP_PATHS = ("/sitemap.xml", "/sitemap_index.xml", "/wp-sitemap.xml")
# A sitemap this small is a stub (language roots only); the site is crawled instead.
MIN_SITEMAP_URLS = 10
CRAWL_DEPTH = 2
ASSET_SUFFIXES = (
    ".jpg", ".jpeg", ".png", ".gif", ".svg", ".webp", ".ico", ".css", ".js",
    ".pdf", ".doc", ".docx", ".xls", ".xlsx", ".ppt", ".pptx", ".zip", ".rar",
    ".mp3", ".mp4", ".avi", ".woff", ".woff2", ".ttf", ".xml", ".json", ".rss",
)  # fmt: skip
WIKIDATA_API = "https://www.wikidata.org/w/api.php"
WIKIDATA_SPARQL = "https://query.wikidata.org/sparql"
_WIKILINK = re.compile(r"\[\[([^\]|#]+)(?:[|#][^\]]*)?\]\]")
_LOC = re.compile(r"<loc>\s*(.*?)\s*</loc>", re.S)
_HREF = re.compile(r"""href\s*=\s*["']([^"'#]+)""", re.I)
_SITEMAP_DIRECTIVE = re.compile(r"(?im)^sitemap:\s*(\S+)")


@dataclass(frozen=True)
class SiteSpec:
    source: str
    root: str
    max_pages: int
    # `attraction` or `town`: which Brestanica source it is compared with.
    kind: str = "attraction"
    include: tuple[str, ...] = ()
    exclude: tuple[str, ...] = ()


@dataclass(frozen=True)
class WikipediaSpec:
    source: str
    seeds: tuple[str, ...]
    max_links: int
    categories: tuple[str, ...] = ()
    max_category: int = 0


@dataclass(frozen=True)
class BiographySpec:
    source: str
    property_id: str
    places: tuple[str, ...]
    max_people: int


@dataclass(frozen=True)
class Cluster:
    name: str
    language: str
    sites: tuple[SiteSpec, ...] = ()
    wikipedia: WikipediaSpec | None = None
    biography: BiographySpec | None = None


@dataclass
class SeedRow:
    source: str
    url: str
    language: str

    def as_dict(self) -> dict[str, str]:
        return {"source": self.source, "url": self.url, "language": self.language}


@dataclass
class BuildReport:
    """What discovery found and what verification kept, per source."""

    discovered: Counter = field(default_factory=Counter)
    kept: Counter = field(default_factory=Counter)
    rejected: Counter = field(default_factory=Counter)


def load_clusters(spec_path: Path = SPEC_PATH) -> list[Cluster]:
    spec = load_yaml(spec_path)
    defaults = spec.get("defaults", {})
    clusters = []
    for name, raw in spec["clusters"].items():
        language = str(raw["language"])
        sites = tuple(
            SiteSpec(
                source=site["source"],
                root=site["root"],
                max_pages=int(
                    site.get("max_pages", defaults.get("site_max_pages", 45))
                ),
                kind=site.get("kind", "attraction" if index == 0 else "town"),
                include=tuple(site.get("include", ())),
                exclude=tuple(site.get("exclude", ())),
            )
            for index, site in enumerate(raw.get("sites", ()))
        )
        wikipedia = None
        if "wikipedia" in raw:
            wiki = raw["wikipedia"]
            wikipedia = WikipediaSpec(
                source=wiki.get("source", f"{name}_wikipedia"),
                seeds=tuple(wiki["seeds"]),
                max_links=int(
                    wiki.get("max_links", defaults.get("wikipedia_max_links", 45))
                ),
                categories=tuple(wiki.get("categories", ())),
                max_category=int(
                    wiki.get("max_category", defaults.get("category_max", 8))
                ),
            )
        biography = None
        if "biography" in raw:
            bio = raw["biography"]
            biography = BiographySpec(
                source=bio.get("source", f"{name}_biography"),
                property_id=bio["property"],
                places=tuple(bio["places"]),
                max_people=int(
                    bio.get("max_people", defaults.get("biography_max", 50))
                ),
            )
        clusters.append(Cluster(name, language, sites, wikipedia, biography))
    return clusters


# --- sites -----------------------------------------------------------------


def same_host(url: str, root: str) -> bool:
    """`www.` is the same site; a different subdomain is not."""

    def strip(host: str) -> str:
        return host.lower().removeprefix("www.")

    return strip(urlparse(url).netloc) == strip(urlparse(root).netloc)


def is_page_url(url: str) -> bool:
    parsed = urlparse(url)
    if parsed.scheme not in ("http", "https") or parsed.query:
        return False
    if parsed.path.startswith("/cdn-cgi/"):  # Cloudflare's email-protection stub
        return False
    return not parsed.path.lower().endswith(ASSET_SUFFIXES)


def unique_pages(urls: list[str]) -> list[str]:
    """First spelling of each page; `https://x` and `https://x/` are one page."""
    seen: set[str] = set()
    unique = []
    for url in urls:
        key = url.rstrip("/")
        if key not in seen:
            seen.add(key)
            unique.append(url)
    return unique


def matches(url: str, site: SiteSpec) -> bool:
    path = urlparse(url).path or "/"
    if any(re.search(pattern, path) for pattern in site.exclude):
        return False
    return not site.include or any(re.search(p, path) for p in site.include)


def parse_sitemap(xml: str) -> tuple[list[str], list[str]]:
    """(child sitemaps, page URLs) of one sitemap document."""
    locs = [loc.strip() for loc in _LOC.findall(xml)]
    if "<sitemapindex" in xml:
        return locs, []
    return [], locs


def sitemap_order(url: str) -> tuple[int, str]:
    """Page sitemaps first: they hold the site's structure, post sitemaps its news."""
    name = urlparse(url).path.lower()
    return (0 if "page" in name else 1 if "post" not in name else 2, name)


def sitemap_urls(root: str, client: httpx.Client, wanted: int) -> list[str]:
    """Page URLs from the site's sitemaps, stopping once `wanted` are collected."""
    candidates: list[str] = []
    try:
        robots = client.get(urljoin(root, "/robots.txt"))
        if robots.status_code == 200:
            # The directive should be absolute; some sites write "/sitemap.xml".
            candidates.extend(
                urljoin(root, found)
                for found in _SITEMAP_DIRECTIVE.findall(robots.text)
            )
    except httpx.HTTPError:
        pass
    candidates.extend(urljoin(root, path) for path in SITEMAP_PATHS)

    pages: list[str] = []
    seen: set[str] = set()
    queue = list(dict.fromkeys(candidates))
    while queue and len(pages) < wanted:
        sitemap = queue.pop(0)
        if sitemap in seen:
            continue
        seen.add(sitemap)
        try:
            response = client.get(sitemap)
        except (httpx.HTTPError, ValueError):
            # ValueError: a malformed sitemap URL; treat it as unreachable.
            continue
        if response.status_code != 200 or "<loc>" not in response.text:
            continue
        children, urls = parse_sitemap(response.text)
        queue = sorted(children, key=sitemap_order) + queue
        pages.extend(urls)
    return list(dict.fromkeys(pages))


def crawl(
    root: str, client: httpx.Client, cap: int, depth: int = CRAWL_DEPTH
) -> list[str]:
    """Breadth-first over the site's own page links, reading order preserved."""
    found = [root.rstrip("/")]
    frontier = [root]
    for _ in range(depth):
        next_frontier: list[str] = []
        for url in frontier:
            try:
                response = client.get(url)
            except httpx.HTTPError:
                continue
            if response.status_code != 200:
                continue
            for href in _HREF.findall(response.text):
                target = urljoin(str(response.url), href.strip()).rstrip("/")
                if target in found or not same_host(target, root):
                    continue
                if not is_page_url(target):
                    continue
                found.append(target)
                next_frontier.append(target)
                if len(found) >= cap:
                    return found
        frontier = next_frontier
    return found


def discover_site(site: SiteSpec, client: httpx.Client) -> list[str]:
    wanted = site.max_pages * 3
    urls = sitemap_urls(site.root, client, wanted)
    if len(urls) < MIN_SITEMAP_URLS:
        urls = crawl(site.root, client, cap=wanted)
    urls = [
        url
        for url in urls
        if same_host(url, site.root) and is_page_url(url) and matches(url, site)
    ]
    return unique_pages([site.root, *urls])[: site.max_pages]


# --- wikipedia ---------------------------------------------------------------


def wikitext_links(wikitext: str) -> list[str]:
    """Article-namespace link targets in reading order, each once."""
    titles: list[str] = []
    for match in _WIKILINK.finditer(wikitext):
        target = match.group(1).strip()
        if not target or ":" in target:
            continue
        title = target[0].upper() + target[1:]
        if title not in titles:
            titles.append(title)
    return titles


def wikipedia_api(client: httpx.Client, language: str, **params) -> dict:
    params.update(action="query", format="json", formatversion="2")
    response = client.get(f"https://{language}.wikipedia.org/w/api.php", params=params)
    response.raise_for_status()
    return response.json()


def article_links(client: httpx.Client, language: str, title: str) -> list[str]:
    data = wikipedia_api(
        client, language, prop="revisions", rvprop="content", rvslots="main",
        titles=title, redirects="1",
    )  # fmt: skip
    pages = data.get("query", {}).get("pages", [])
    if not pages or pages[0].get("missing"):
        return []
    wikitext = pages[0]["revisions"][0]["slots"]["main"]["content"]
    return wikitext_links(wikitext)


def category_members(
    client: httpx.Client, language: str, category: str, limit: int
) -> list[str]:
    data = wikipedia_api(
        client, language, list="categorymembers", cmtitle=category,
        cmnamespace="0", cmlimit=str(max(limit, 1)),
    )  # fmt: skip
    members = data.get("query", {}).get("categorymembers", [])
    return [m["title"] for m in members][:limit]


def wikipedia_url(language: str, title: str) -> str:
    path = quote(title.replace(" ", "_"), safe="(),")
    return f"https://{language}.wikipedia.org/wiki/{path}"


def discover_wikipedia(
    spec: WikipediaSpec, language: str, client: httpx.Client
) -> list[str]:
    titles: list[str] = []
    for seed in spec.seeds:
        for title in [seed, *article_links(client, language, seed)]:
            if title not in titles:
                titles.append(title)
    titles = titles[: spec.max_links]
    for category in spec.categories:
        for title in category_members(client, language, category, spec.max_category):
            if title not in titles:
                titles.append(title)
    return [wikipedia_url(language, title) for title in titles]


# --- biography -----------------------------------------------------------------

_SPARQL = """
SELECT DISTINCT ?id ?sitelinks WHERE {{
  VALUES ?place {{ {places} }}
  ?person wdt:P19/wdt:P131* ?place ;
          wdt:{property} ?id ;
          wikibase:sitelinks ?sitelinks .
}}
ORDER BY DESC(?sitelinks) LIMIT {limit}
"""


def property_formatters(client: httpx.Client, property_id: str) -> list[str]:
    """Every formatter URL (P1630) of the property, in Wikidata's order."""
    response = client.get(
        WIKIDATA_API,
        params={
            "action": "wbgetentities",
            "ids": property_id,
            "props": "claims",
            "format": "json",
        },
    )
    response.raise_for_status()
    claims = response.json()["entities"][property_id].get("claims", {})
    formatters = [
        str(claim["mainsnak"]["datavalue"]["value"])
        for claim in claims.get("P1630", [])
        if "datavalue" in claim["mainsnak"]
    ]
    if not formatters:
        raise ValueError(f"{property_id} has no formatter URL (P1630)")
    return formatters


def live_formatter(formatters: list[str], sample_id: str, is_live) -> str:
    """The first formatter whose URL for `sample_id` answers.

    A property can keep a superseded formatter first (HBL lists its retired
    `clanak.aspx?id=` form before the current `clanak/` one), and one probe
    tells them apart.
    """
    for formatter in formatters:
        if is_live(formatter.replace("$1", sample_id)):
            return formatter
    return formatters[0]


def _sparql_ids(
    client: httpx.Client, spec: BiographySpec, places: tuple[str, ...]
) -> list[str]:
    query = _SPARQL.format(
        places=" ".join(f"wd:{qid}" for qid in places),
        property=spec.property_id,
        limit=spec.max_people,
    )
    response = client.get(
        WIKIDATA_SPARQL,
        params={"query": query, "format": "json"},
        headers={"Accept": "application/sparql-results+json"},
        timeout=SPARQL_TIMEOUT,
    )
    response.raise_for_status()
    rows = response.json()["results"]["bindings"]
    return list(dict.fromkeys(row["id"]["value"] for row in rows))


def lexicon_ids(client: httpx.Client, spec: BiographySpec) -> list[str]:
    """Lexicon IDs for the places together; place by place if that times out."""
    try:
        return _sparql_ids(client, spec, spec.places)
    except httpx.HTTPError:
        ids: list[str] = []
        for place in spec.places:
            try:
                ids.extend(_sparql_ids(client, spec, (place,)))
            except httpx.HTTPError as exc:
                print(
                    f"  {spec.source}: {place} skipped ({type(exc).__name__})",
                    flush=True,
                )
        return list(dict.fromkeys(ids))[: spec.max_people]


def discover_biography(spec: BiographySpec, client: httpx.Client) -> list[str]:
    ids = lexicon_ids(client, spec)
    if not ids:
        return []

    def is_live(url: str) -> bool:
        try:
            return client.get(url).status_code == 200
        except httpx.HTTPError:
            return False

    formatter = live_formatter(
        property_formatters(client, spec.property_id), ids[0], is_live
    )
    return [formatter.replace("$1", entry_id) for entry_id in ids]


# --- verification and output ---------------------------------------------------


def make_client() -> httpx.Client:
    return httpx.Client(
        headers={"User-Agent": USER_AGENT, "Accept-Language": "*"},
        timeout=TIMEOUT,
        follow_redirects=True,
    )


def verify(
    rows: list[SeedRow],
    workers: int,
    report: BuildReport,
    min_text_chars: int = MIN_TEXT_CHARS,
) -> list[SeedRow]:
    """Keep rows that answer 200 with enough extracted text, in their original order."""
    throttle = DomainThrottle(DOMAIN_DELAY)
    client = make_client()

    def check(row: SeedRow) -> str | None:
        throttle.wait(row.url)
        try:
            response = client.get(row.url)
        except httpx.HTTPError as exc:
            return type(exc).__name__
        if response.status_code != 200:
            return f"HTTP {response.status_code}"
        content_type = response.headers.get("content-type", "")
        if not any(marker in content_type for marker in ("html", "xml", "text/")):
            return f"content-type {content_type.split(';')[0] or 'unknown'}"
        text = extract_text(response.text) or ""
        if len(text) < min_text_chars:
            return f"text < {min_text_chars} chars"
        return None

    with ThreadPoolExecutor(max_workers=workers) as pool:
        verdicts = list(pool.map(check, rows))
    kept = []
    for row, verdict in zip(rows, verdicts):
        if verdict is None:
            report.kept[row.source] += 1
            kept.append(row)
        else:
            report.rejected[f"{row.source}: {verdict}"] += 1
    return kept


def build_cluster(
    cluster: Cluster, client: httpx.Client, report: BuildReport
) -> list[SeedRow]:
    rows: list[SeedRow] = []

    def add(source: str, urls: list[str]) -> None:
        report.discovered[source] += len(urls)
        rows.extend(SeedRow(source, url, cluster.language) for url in urls)

    def attempt(source: str, discover) -> None:
        # One unreachable service must not lose the cluster's other sources.
        try:
            add(source, discover())
        except (httpx.HTTPError, ValueError, KeyError) as exc:
            report.rejected[f"{source}: discovery failed ({type(exc).__name__})"] += 1

    for site in cluster.sites:
        attempt(site.source, lambda site=site: discover_site(site, client))
    if cluster.wikipedia:
        wiki = cluster.wikipedia
        attempt(wiki.source, lambda: discover_wikipedia(wiki, cluster.language, client))
    if cluster.biography:
        bio = cluster.biography
        attempt(bio.source, lambda: discover_biography(bio, client))
    return rows


def reference_urls(path: Path = REFERENCE_SEED) -> frozenset[str]:
    if not path.exists():
        return frozenset()
    return frozenset(
        row["url"].rstrip("/") for row in json.loads(path.read_text("utf-8"))
    )


def without_reference(rows: list[SeedRow], reference: frozenset[str]) -> list[SeedRow]:
    return [row for row in rows if row.url.rstrip("/") not in reference]


def write_seed(path: Path, rows: list[SeedRow]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = [row.as_dict() for row in rows]
    text = json.dumps(payload, ensure_ascii=False, indent=1)
    path.write_text(text + "\n", encoding="utf-8")


def merge_seeds(out_dir: Path) -> list[SeedRow]:
    """Every cluster file in `out_dir`, so a one-cluster rebuild updates the merged file too."""
    rows: dict[str, SeedRow] = {}
    for path in sorted(out_dir.glob("*.json")):
        for row in json.loads(path.read_text(encoding="utf-8")):
            # Neighbouring clusters share Wikipedia articles; the first keeps it,
            # since page_metadata.url is unique and a re-upsert would just move
            # the page between sources.
            rows.setdefault(
                row["url"], SeedRow(row["source"], row["url"], row.get("language", ""))
            )
    return list(rows.values())


def format_report(report: BuildReport) -> str:
    lines = [f"{'source':36s} {'found':>6s} {'kept':>6s}"]
    for source in report.discovered:
        lines.append(
            f"{source:36s} {report.discovered[source]:6d} {report.kept[source]:6d}"
        )
    if report.rejected:
        lines.append("rejected:")
        lines.extend(
            f"  {reason}: {count}" for reason, count in report.rejected.most_common()
        )
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--spec", type=Path, default=SPEC_PATH)
    parser.add_argument("--out-dir", type=Path, default=ROOT / "data" / "seeds")
    parser.add_argument(
        "--merged",
        type=Path,
        default=ROOT / "data" / "eu_neighbours.json",
        help="One file with every cluster's rows, for a single fetch-pages run.",
    )
    parser.add_argument("--cluster", action="append", help="Build only these clusters.")
    parser.add_argument(
        "--no-verify", action="store_true", help="Skip the liveness check."
    )
    parser.add_argument("--workers", type=int, default=8)
    args = parser.parse_args()

    clusters = load_clusters(args.spec)
    defaults = load_yaml(args.spec).get("defaults", {})
    min_text_chars = int(defaults.get("min_text_chars", MIN_TEXT_CHARS))
    if args.cluster:
        clusters = [c for c in clusters if c.name in set(args.cluster)]
    report = BuildReport()
    with make_client() as client:
        for cluster in clusters:
            rows = build_cluster(cluster, client, report)
            if args.no_verify:
                report.kept.update(row.source for row in rows)
            else:
                rows = verify(rows, args.workers, report, min_text_chars)
            rows = without_reference(rows, reference_urls())
            write_seed(args.out_dir / f"{cluster.name}.json", rows)
            print(f"{cluster.name}: {len(rows)} urls", flush=True)
    write_seed(args.merged, merge_seeds(args.out_dir))
    print(format_report(report))


if __name__ == "__main__":
    main()

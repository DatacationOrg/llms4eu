"""Score the neighbouring-country sources against the Brestanica sources.

Brestanica is the gold standard: its four sources were chosen by hand and carry
the approved eval labels. Every new source in `seeds.yaml` is one of the same
four kinds (attraction site, town site, biography lexicon, Wikipedia), so each
is compared with the Brestanica source of its kind on measures that need no
model: how many pages, how long they are, how much of each page is prose rather
than navigation, how many are near-empty, duplicates, and whether the detected
language matches the declared one.

Read-only by default. `--prune-below N` deletes new-source pages with fewer
than N characters of prose, so stubs never reach chunking; a reference page is
never touched.
"""

from __future__ import annotations

import argparse
import statistics
from collections import Counter, defaultdict
from dataclasses import dataclass
from datetime import date
from pathlib import Path

from src.db.pages import connect_pages, initialize_page_artifacts_db, raw_pages_db_path
from src.preprocess.languages import prose_sample
from src.scraping.seed_urls import SPEC_PATH, Cluster, load_clusters
from src.shared.language import detect_language
from src.shared.env import load_yaml

# A page with less prose than this is a stub: an index record, a gallery, a
# contact card. Same bar as the seed builder's `min_text_chars`.
STUB_PROSE_CHARS = 300
# Beyond these ratios to the reference a source is flagged rather than passed.
THIN_RATIO = 0.5
STUB_RATIO = 2.0
LANGUAGE_MISMATCH_SHARE = 0.4


@dataclass
class PageStats:
    id: str
    source: str
    chars: int
    prose_chars: int
    page_kind: str
    content_hash: str | None
    declared: str | None
    detected: str | None


@dataclass
class SourceStats:
    source: str
    kind: str
    pages: int
    median_chars: int
    median_prose: int
    prose_share: float
    stub_share: float
    listing_share: float
    duplicate_share: float
    language_match: float | None

    @classmethod
    def build(cls, source: str, kind: str, pages: list[PageStats]) -> "SourceStats":
        n = len(pages)
        hashes = Counter(p.content_hash for p in pages if p.content_hash)
        duplicates = sum(count - 1 for count in hashes.values())
        judged = [p for p in pages if p.detected and p.declared]
        return cls(
            source=source,
            kind=kind,
            pages=n,
            median_chars=int(statistics.median(p.chars for p in pages)),
            median_prose=int(statistics.median(p.prose_chars for p in pages)),
            prose_share=sum(p.prose_chars for p in pages)
            / max(sum(p.chars for p in pages), 1),
            stub_share=sum(p.prose_chars < STUB_PROSE_CHARS for p in pages) / n,
            listing_share=sum(p.page_kind != "prose" for p in pages) / n,
            duplicate_share=duplicates / n,
            language_match=(
                sum(p.detected == p.declared for p in judged) / len(judged)
                if judged
                else None
            ),
        )


def verdict(stats: SourceStats, reference: SourceStats) -> str:
    """One word a reader can sort on, and the reason after it."""
    flags = []
    if stats.median_prose < THIN_RATIO * reference.median_prose:
        flags.append(
            f"thin: median prose {stats.median_prose} vs {reference.median_prose}"
        )
    if stats.stub_share > max(STUB_RATIO * reference.stub_share, 0.1):
        flags.append(f"stubs: {stats.stub_share:.0%} vs {reference.stub_share:.0%}")
    if stats.duplicate_share > max(STUB_RATIO * reference.duplicate_share, 0.1):
        flags.append(
            f"duplicates: {stats.duplicate_share:.0%} vs {reference.duplicate_share:.0%}"
        )
    if (
        stats.language_match is not None
        and stats.language_match < 1 - LANGUAGE_MISMATCH_SHARE
    ):
        flags.append(f"language: {stats.language_match:.0%} match declared")
    return "; ".join(flags) if flags else "ok"


def load_page_stats() -> list[PageStats]:
    initialize_page_artifacts_db()
    with connect_pages() as conn:
        rows = conn.execute(
            """
            select m.id, m.source, m.markdown_chars, m.page_kind, m.content_hash,
                   coalesce(m.language, s.language) as declared, c.markdown
            from page_metadata m
            join page_markdown_content c on c.page_id = m.id
            left join page_sources s on s.source = m.source
            """
        ).fetchall()
    stats = []
    for row in rows:
        prose = prose_sample(row["markdown"], limit=len(row["markdown"]) + 1)
        stats.append(
            PageStats(
                id=row["id"],
                source=row["source"],
                chars=row["markdown_chars"],
                prose_chars=len(prose),
                page_kind=row["page_kind"],
                content_hash=row["content_hash"],
                declared=row["declared"],
                detected=detect_language(prose[:4000]),
            )
        )
    return stats


def source_kinds(clusters: list[Cluster]) -> dict[str, str]:
    kinds: dict[str, str] = {}
    for cluster in clusters:
        for site in cluster.sites:
            kinds[site.source] = site.kind
        if cluster.wikipedia:
            kinds[cluster.wikipedia.source] = "wikipedia"
        if cluster.biography:
            kinds[cluster.biography.source] = "biography"
    return kinds


def compare(
    pages: list[PageStats], kinds: dict[str, str], reference: dict[str, str]
) -> tuple[dict[str, SourceStats], list[tuple[SourceStats, SourceStats, str]]]:
    by_source: dict[str, list[PageStats]] = defaultdict(list)
    for page in pages:
        by_source[page.source].append(page)
    references = {
        kind: SourceStats.build(source, kind, by_source[source])
        for kind, source in reference.items()
        if by_source.get(source)
    }
    rows = []
    for source, kind in kinds.items():
        if not by_source.get(source) or kind not in references:
            continue
        stats = SourceStats.build(source, kind, by_source[source])
        rows.append((stats, references[kind], verdict(stats, references[kind])))
    return references, rows


def format_row(stats: SourceStats, verdict_text: str = "") -> str:
    match = "-" if stats.language_match is None else f"{stats.language_match:.0%}"
    return (
        f"| {stats.source} | {stats.kind} | {stats.pages} | {stats.median_chars:,} | "
        f"{stats.median_prose:,} | {stats.prose_share:.0%} | {stats.stub_share:.0%} | "
        f"{stats.listing_share:.0%} | {stats.duplicate_share:.0%} | {match} | {verdict_text} |"
    )


HEADER = (
    "| source | kind | pages | median chars | median prose | prose share | stubs | "
    "non-prose | duplicates | language match | verdict |\n"
    "|---|---|---|---|---|---|---|---|---|---|---|"
)


def format_report(references: dict[str, SourceStats], rows) -> str:
    lines = [
        f"# Seed quality against the Brestanica sources, {date.today().isoformat()}",
        "",
        f"Database: `{raw_pages_db_path()}`. Prose is the text left after stripping",
        "links, images, markup and lines that are mostly not letters; a stub has",
        f"under {STUB_PROSE_CHARS} prose characters. A source is `thin` when its median",
        f"prose is under {THIN_RATIO:.0%} of its reference, `stubs` when its stub share is",
        f"over {STUB_RATIO:.0f}x the reference (and over 10%), `duplicates` likewise for",
        "pages whose content hash repeats within the source, `language` when under",
        f"{1 - LANGUAGE_MISMATCH_SHARE:.0%} of its pages detect as the declared language.",
        "",
        "## Reference (gold standard)",
        "",
        HEADER,
        *(format_row(ref, "reference") for ref in references.values()),
        "",
        "## New sources",
        "",
        HEADER,
        *(
            format_row(stats, text)
            for stats, _, text in sorted(rows, key=lambda r: (r[0].kind, r[0].source))
        ),
    ]
    flagged = [stats.source for stats, _, text in rows if text != "ok"]
    lines += [
        "",
        f"Flagged: {len(flagged)} of {len(rows)} sources"
        + (": " + ", ".join(flagged) if flagged else "."),
    ]
    return "\n".join(lines) + "\n"


def stub_ids(pages: list[PageStats], kinds: dict[str, str], below: int) -> set[str]:
    """New-source pages with fewer than `below` prose characters."""
    return {p.id for p in pages if p.source in kinds and p.prose_chars < below}


def duplicate_ids(pages: list[PageStats], kinds: dict[str, str]) -> set[str]:
    """Every new-source page whose content repeats within its source.

    Identical content across different URLs is not the page, it is what the
    extractor fell back to (a cookie notice, a navigation block), so no copy is
    worth keeping.
    """
    groups: dict[tuple[str, str], list[str]] = defaultdict(list)
    for page in pages:
        if page.source in kinds and page.content_hash:
            groups[(page.source, page.content_hash)].append(page.id)
    return {page_id for ids in groups.values() if len(ids) > 1 for page_id in ids}


def delete_pages(page_ids: set[str]) -> int:
    if not page_ids:
        return 0
    with connect_pages() as conn:
        conn.executemany(
            "delete from page_metadata where id = ?", [(v,) for v in sorted(page_ids)]
        )
    return len(page_ids)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--spec", type=Path, default=SPEC_PATH)
    parser.add_argument(
        "--output", type=Path, help="Also write the Markdown report here."
    )
    parser.add_argument(
        "--prune-below",
        type=int,
        metavar="N",
        help="Delete new-source pages with fewer than N prose characters.",
    )
    parser.add_argument(
        "--drop-duplicates",
        action="store_true",
        help="Delete new-source pages whose content repeats within the source.",
    )
    args = parser.parse_args()

    spec = load_yaml(args.spec)
    kinds = source_kinds(load_clusters(args.spec))
    pages = load_page_stats()
    doomed: set[str] = set()
    if args.prune_below is not None:
        doomed |= stub_ids(pages, kinds, args.prune_below)
        print(f"stubs under {args.prune_below} prose chars: {len(doomed)}")
    if args.drop_duplicates:
        duplicates = duplicate_ids(pages, kinds)
        print(f"duplicate-content pages: {len(duplicates)}")
        doomed |= duplicates
    if doomed:
        print(f"deleted {delete_pages(doomed)} new-source pages")
        pages = [p for p in pages if p.id not in doomed]
    references, rows = compare(pages, kinds, spec["reference"])
    report = format_report(references, rows)
    print(report)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(report, encoding="utf-8")


if __name__ == "__main__":
    main()

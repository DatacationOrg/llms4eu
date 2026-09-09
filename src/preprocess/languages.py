"""Detect the language of each scraped page.

Read-only by default. Changing a stored language changes the v2 embedding text
and every chunk's Chroma metadata, so the distribution is reported for review
before anything is written.

Pages keep their source's configured language until detection is applied; the
column is a per-page override, not a replacement for `page_sources`.
"""

from __future__ import annotations

import argparse
import re
from collections import Counter
from dataclasses import dataclass

from src.db.pages import connect_pages as connect
from src.db.pages import initialize_page_artifacts_db
from src.shared.language import detect_language_confidence

# Detection reads a prose sample: enough signal to be decisive, cheap enough to
# run over the whole corpus in seconds.
SAMPLE_CHARS = 4000

# Markdown boilerplate is not language. Feeding raw Markdown to the detector
# makes an address-and-phone-number page look Croatian, because the only thing
# a detector can see is the shape of the punctuation. These strip the parts of a
# page that carry no linguistic signal.
_IMAGE = re.compile(r"!\[[^\]]*\]\([^)]*\)")
_LINK = re.compile(r"\[([^\]]*)\]\([^)]*\)")
_URL = re.compile(r"https?://\S+|www\.\S+|\S+@\S+\.\S+")
_CODE = re.compile(r"```.*?```|`[^`]*`", re.DOTALL)
_MARKUP = re.compile(r"[#*_>|\[\]]+")

# A line needs to be mostly letters to count as prose. Tables, phone numbers and
# opening-hours listings fall below this and are dropped.
MIN_LETTER_RATIO = 0.65
MIN_PROSE_LINE_CHARS = 24


def prose_sample(markdown: str, limit: int = SAMPLE_CHARS) -> str:
    """Strip Markdown boilerplate down to the text that actually carries language."""
    text = _CODE.sub(" ", markdown)
    text = _IMAGE.sub(" ", text)
    text = _LINK.sub(r"\1", text)
    text = _URL.sub(" ", text)
    text = _MARKUP.sub(" ", text)

    kept: list[str] = []
    size = 0
    for line in text.splitlines():
        line = re.sub(r"\s+", " ", line).strip()
        if len(line) < MIN_PROSE_LINE_CHARS or _letter_ratio(line) < MIN_LETTER_RATIO:
            continue
        kept.append(line)
        size += len(line) + 1
        if size >= limit:
            break
    # Truncate as well as stop: a page can be one very long line, and the
    # per-line check alone would hand the detector the whole page.
    return "\n".join(kept)[:limit]


def _letter_ratio(line: str) -> float:
    letters = sum(1 for character in line if character.isalpha() or character == " ")
    return letters / len(line) if line else 0.0


@dataclass(frozen=True)
class PageLanguage:
    page_id: str
    source: str
    title: str | None
    stamped: str | None
    detected: str | None
    confidence: float

    @property
    def disagrees(self) -> bool:
        return self.detected is not None and self.detected != self.stamped


def detect_page_languages() -> list[PageLanguage]:
    """Detect every non-empty page's language without writing anything."""
    initialize_page_artifacts_db()
    with connect() as conn:
        rows = conn.execute(
            """
            select m.id, m.source, m.title, m.language as page_language,
                   s.language as source_language, c.markdown
            from page_metadata m
            join page_markdown_content c on c.page_id = m.id
            left join page_sources s on s.source = m.source
            where m.page_kind != 'empty'
            order by m.id
            """
        ).fetchall()

    results = []
    for row in rows:
        detected, confidence = detect_language_confidence(prose_sample(row["markdown"]))
        results.append(
            PageLanguage(
                page_id=row["id"],
                source=row["source"],
                title=row["title"],
                stamped=row["page_language"] or row["source_language"],
                detected=detected,
                confidence=confidence,
            )
        )
    return results


def apply_page_languages(results: list[PageLanguage]) -> int:
    """Write confident detections onto page_metadata.language."""
    rows = [
        (result.detected, result.page_id)
        for result in results
        if result.detected is not None
    ]
    with connect() as conn:
        conn.executemany(
            "update page_metadata set language = ? where id = ?",
            rows,
        )
    return len(rows)


def format_report(results: list[PageLanguage]) -> str:
    detected = Counter(result.detected or "undetermined" for result in results)
    stamped = Counter(result.stamped or "unset" for result in results)
    disagreements = [result for result in results if result.disagrees]

    lines = [
        f"pages scanned: {len(results)}",
        f"stamped:  {_counts(stamped)}",
        f"detected: {_counts(detected)}",
        f"disagreements: {len(disagreements)}",
    ]
    if disagreements:
        lines.append("")
        lines.append("| Page | Source | Stamped | Detected | Confidence | Title |")
        lines.append("|---|---|---|---|---:|---|")
        for result in sorted(disagreements, key=lambda item: -item.confidence)[:40]:
            lines.append(
                f"| {result.page_id} | {result.source} | {result.stamped} | "
                f"{result.detected} | {result.confidence:.2f} | "
                f"{(result.title or '').replace('|', '/')[:60]} |"
            )
        if len(disagreements) > 40:
            lines.append(f"\n({len(disagreements) - 40} further disagreements omitted)")
    return "\n".join(lines)


def _counts(counter: Counter) -> str:
    return ", ".join(f"{key}={value}" for key, value in counter.most_common())


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--apply",
        action="store_true",
        help="Write detected languages to page_metadata.language.",
    )
    args = parser.parse_args()

    results = detect_page_languages()
    print(format_report(results))
    if not args.apply:
        print("\nread-only: re-run with --apply to write page_metadata.language")
        return
    written = apply_page_languages(results)
    print(f"\nwrote language for {written} pages")


if __name__ == "__main__":
    main()

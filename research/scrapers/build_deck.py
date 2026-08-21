"""Check the deck's numbers against their source, then inline the figures.

Two jobs, in this order:

1. **Verify.** Every similarity number in `deck.src.html` is a literal, hand-copied
   out of `deck_data.json` (what `just arena-similarity` writes). Hand-copied numbers
   drift, and a deck that quietly disagrees with its own data is worse than no deck.
   So the arrays are parsed back out of the source and compared, cell by cell, and a
   mismatch fails the build.

2. **Inline.** The published artifact must be self-contained -- the viewer's CSP
   blocks every external host except Google Fonts -- so the four reference figures are
   embedded as data URIs rather than linked. Kept as a build step instead of editing
   deck.html by hand because a 1.6 MB file with base64 blobs in it is not something
   anyone should be diffing.

    uv run python -m research.scrapers.build_deck
"""

from __future__ import annotations

import base64
import json
import re
from pathlib import Path

HERE = Path(__file__).parent
FIGURES = {
    "__INFOBOX__": "infobox.png",
    "__WIKITABLE__": "wikitable.png",
    "__PYDOCS__": "pydocs.png",
    "__ARENA__": "arena.png",
}

# Deck label -> the name `similarity.py` uses. None means "measured outside the
# similarity pass", which has to be declared here rather than silently skipped:
# `ours@committed` is not in the run that produced deck_data.json, and the whole point
# of this check is that an unexplained number fails the build.
FETCHER_NAMES = {
    "httpx": "ours-httpx",
    "Scrapy": "scrapy",
    "Scrapling": "scrapling",
    "Stealthy": "scrapling-stealthy",
}
EXTRACTOR_NAMES = {
    "goose3": "goose3",
    "justext": "justext",
    "ours · proposed": "ours",
    "trafilatura": "trafilatura",
    "resiliparse": "resiliparse",
    "html-text": "html-text",
    "ours · now": None,
    "readability": "readability-lxml",
    "crawl4ai": "crawl4ai",
    "scrapling-md": "scrapling-md",
    "markitdown": "markitdown",
    "docling": "docling",
}


def data_uri(path: Path) -> str:
    encoded = base64.b64encode(path.read_bytes()).decode("ascii")
    return f"data:image/png;base64,{encoded}"


def _js_array(html: str, name: str) -> list:
    """Pull one `const NAME = [...];` literal out of the deck and parse it.

    The literals are JSON apart from trailing commas, which the deck uses for
    readable diffs, so they are stripped before parsing.
    """
    match = re.search(rf"const {name} = (\[.*?\]);", html, re.DOTALL)
    if not match:
        raise SystemExit(f"{name} not found in deck.src.html")
    text = re.sub(r",(\s*[\]}])", r"\1", match.group(1))
    return json.loads(text)


def _check_layer(
    html: str, layer: str, data: dict, chars_var: str, labels_var: str, sim_var: str,
    names: dict[str, str | None], chars_key: str,
) -> list[str]:
    """Compare one matrix and its char counts against the similarity output.

    `chars_key` differs by layer, deliberately. The fetcher bars answer "does this
    client return the same document", which is only meaningful over the pages every
    client handled, so they use `mean_common`. The extractor bars answer "how much
    does this tool keep", which is a property of the tool, so they use `mean_own`.
    """
    problems: list[str] = []
    labels = _js_array(html, labels_var)
    # Group headings carry a null value instead of a number; they are layout, not data.
    chars = [row for row in _js_array(html, chars_var) if row[1] is not None]
    matrix = _js_array(html, sim_var)

    unknown = [label for label in labels if label not in names]
    if unknown:
        problems.append(f"{layer}: labels not mapped in build_deck.py: {unknown}")
        return problems
    if len(chars) != len(labels):
        problems.append(
            f"{layer}: {chars_var} has {len(chars)} bars for {len(labels)} labels"
        )
        return problems

    for (deck_label, deck_value, _), axis_label in zip(chars, labels):
        name = names[axis_label]
        if name is None:
            continue
        expected = data["chars"][name][chars_key]
        if deck_value != expected:
            problems.append(
                f"{layer}: {deck_label} chars {deck_value} != {expected} ({chars_key})"
            )

    for i, a_label in enumerate(labels):
        for j, b_label in enumerate(labels):
            a, b = names[a_label], names[b_label]
            if a is None or b is None:
                continue
            cell = data["cosine"].get(f"{a}|{b}") or data["cosine"].get(f"{b}|{a}")
            if cell is None or cell.get("pct") is None:
                problems.append(f"{layer}: no cosine for {a_label} vs {b_label}")
                continue
            if round(cell["pct"], 1) != matrix[i][j]:
                problems.append(
                    f"{layer}: {a_label} vs {b_label} is {matrix[i][j]} in the deck, "
                    f"{round(cell['pct'], 1)} in deck_data.json"
                )
    return problems


# Every custom property the slides read. Checked because the palette lives in one
# `:root` block: lose it and the deck still builds, still has 24 sections, and renders
# with every colour resolving to nothing.
REQUIRED_TOKENS = (
    "--paper", "--raise", "--ink", "--slate",
    "--line", "--teal", "--rust", "--pencil", "--hm-on", "--shadow",
)


def check_structure(html: str) -> None:
    """Fail on a source that is not a whole document.

    Cheap, and it catches the failure mode that is invisible downstream: a truncated
    or half-edited source produces a deck.html of the right size with the right number
    of sections, and only looks wrong to a human opening it.
    """
    problems = []
    if html.count("<style>") != 1 or html.count("</style>") != 1:
        problems.append(
            f"expected one <style>...</style>, found {html.count('<style>')} open / "
            f"{html.count('</style>')} close"
        )
    if "<title>" not in html:
        problems.append("no <title>")
    css = html[: html.index("</style>")] if "</style>" in html else ""
    # The light palette is the base; the dark blocks only re-map it.
    base = css.split("@media", 1)[0]
    missing = [token for token in REQUIRED_TOKENS if f"{token}:" not in base]
    if missing:
        problems.append(f"base :root palette is missing {', '.join(missing)}")
    sections = html.count("<section>")
    if sections < 20:
        problems.append(f"only {sections} sections -- the deck has 24")
    if problems:
        raise SystemExit("deck.src.html is not intact:\n  " + "\n  ".join(problems))


def verify(html: str) -> None:
    """Fail the build if the deck and deck_data.json disagree."""
    path = HERE / "deck_data.json"
    if not path.exists():
        raise SystemExit(
            f"missing {path.name} -- run `just arena-similarity` before building"
        )
    data = json.loads(path.read_text(encoding="utf-8"))

    problems = _check_layer(
        html, "fetchers", data["fetchers"],
        "FETCH_CHARS", "FETCH_LABELS", "FETCH_SIM", FETCHER_NAMES, "mean_common",
    ) + _check_layer(
        html, "extractors", data["extractors"],
        "EXT_CHARS", "EXT_LABELS", "EXT_SIM", EXTRACTOR_NAMES, "mean_own",
    )
    if problems:
        raise SystemExit(
            "deck numbers disagree with deck_data.json:\n  "
            + "\n  ".join(problems)
        )

    external = sorted(k for k, v in EXTRACTOR_NAMES.items() if v is None)
    checked = sum(1 for v in EXTRACTOR_NAMES.values() if v) + len(FETCHER_NAMES)
    print(f"verified {checked} entrants against deck_data.json", end="")
    print(f" (externally sourced, unchecked: {', '.join(external)})" if external else "")


def main() -> None:
    source = HERE / "deck.src.html"
    html = source.read_text(encoding="utf-8")

    check_structure(html)
    verify(html)

    for placeholder, filename in FIGURES.items():
        path = HERE / "figures" / filename
        if not path.exists():
            raise SystemExit(f"missing figure: {path}")
        if placeholder not in html:
            raise SystemExit(f"{placeholder} not present in {source.name}")
        html = html.replace(placeholder, data_uri(path))

    left = [name for name in FIGURES if name in html]
    if left:
        raise SystemExit(f"placeholders survived substitution: {left}")

    out = HERE / "deck.html"
    out.write_text(html, encoding="utf-8")
    print(
        f"{out.relative_to(HERE.parents[1])}  "
        f"{out.stat().st_size / 1_048_576:.2f} MB  "
        f"{html.count('<section>')} sections"
    )


if __name__ == "__main__":
    main()

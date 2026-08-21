"""Run the `wikiextractor-v2` entrant over the wiki slice.

Separate script, separate interpreter. Wikiextractor-V2 uses inline regex global
flags -- `(?i)` mid-pattern -- which Python 3.11+ rejects, so it only imports on
Python <= 3.10. Rather than patch a third-party AGPL tool (which would change
what we measure), it runs unmodified under its own Python:

    uv run --python 3.10 --no-project --with pylatexenc --with babel \
        --with beautifulsoup4 --with pyyaml --with python-dotenv \
        research/scrapers/wiki_pass.py

Timings from this pass are therefore measured in a different interpreter than the
other entrants and are indicative only; the report says so.
"""

from __future__ import annotations

import hashlib
import html as html_module
import io
import statistics
import sys
import tempfile
import time
from pathlib import Path
from urllib.parse import unquote

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from research.scrapers import store  # noqa: E402


ENTRANT = "wikiextractor-v2"
VENDOR_SUBPATH = "vendor/Wikiextractor-V2"


def _vendor_root() -> Path:
    root = store.data_dir() / VENDOR_SUBPATH
    if not root.is_dir():
        raise SystemExit(
            f"Wikiextractor-V2 missing at {root}. Clone it with:\n"
            "  git clone --depth 1 "
            f"https://github.com/langtech-bsc/Wikiextractor-V2.git {root}"
        )
    return root


def build_dump(wikitext: str, title: str, base_url: str) -> str:
    """A minimal single-page MediaWiki dump.

    One XML tag per line is required: preprocess_dump() runs a single regex match
    per line and breaks on `</siteinfo>`, so tags sharing a line get missed and
    the loop would swallow the whole document.
    """
    lang = base_url.split("//")[-1].split(".")[0] or "en"
    return (
        f'<mediawiki xml:lang="{lang}">\n'
        "<siteinfo>\n"
        "<sitename>Wikipedia</sitename>\n"
        f"<base>{html_module.escape(base_url)}</base>\n"
        "<namespaces>\n"
        '<namespace key="0" case="first-letter" />\n'
        '<namespace key="10" case="first-letter">Template</namespace>\n'
        "</namespaces>\n"
        "</siteinfo>\n"
        "<page>\n"
        f"<title>{html_module.escape(title)}</title>\n"
        "<ns>0</ns>\n"
        "<id>1</id>\n"
        "<revision>\n"
        "<id>2</id>\n"
        f'<text xml:space="preserve">{html_module.escape(wikitext)}</text>\n'
        "</revision>\n"
        "</page>\n"
        "</mediawiki>\n"
    )


def bootstrap():
    """Import the tool and bind to the Extractor class its own logic uses.

    Subtle and load-bearing: `WikiExtractor.py` prepends `wikiextractor/extract/`
    to `sys.path` and does `from extract import Extractor`, so `extract.py` is
    imported a *second* time under the module name `extract`. Importing
    `wikiextractor.extract.extract` here yields a different class object, and
    every attribute set on it is invisible to `preprocess_dump()` and to the
    discard logic that reads `Extractor.discardTemplates`. Configuring the wrong
    twin left those attributes as the literal config *paths*, so membership tests
    like `title.lower() in self.discardTemplates` degraded into substring tests
    against `"config/discard_templates.txt"` -- which silently discarded any page
    containing a `{{ca|...}}` template, because "ca" occurs in "dis<b>ca</b>rd".
    Beethoven and nl/Amsterdam extracted to zero bytes for exactly that reason.
    """
    root = _vendor_root()
    sys.path.insert(0, str(root))
    import wikiextractor.WikiExtractor as extractor_main

    Extractor = extractor_main.Extractor

    Extractor.generator = True
    Extractor.to_txt = True
    Extractor.to_json = False
    Extractor.markdown = True
    Extractor.keepSections = True
    Extractor.keepLinks = False
    return extractor_main, Extractor


def load_language_config(extractor_main, Extractor, lang: str, cache: dict) -> None:
    """Point Extractor at the discard/ignore config for `lang`.

    The shipped config files are keyed by language (`+ LANGUAGE (ISO-639-1): sl`)
    and parsed against `Extractor.language`, which `preprocess_dump()` derives
    from the dump's `<base>` URL. A real run processes one wiki, hence one
    language, so the tool loads this exactly once; our corpus spans seven, so we
    have to reload per language or six of them silently get English section
    names. Results are cached, keeping config parsing out of the timed loop.
    """
    if lang not in cache:
        Extractor.discardSections = extractor_main.CONFIG_DISCARD_SECTIONS_PATH
        Extractor.discardTemplates = extractor_main.CONFIG_DISCARD_TEMPLATES_PATH
        Extractor.ignoreTemplates = extractor_main.CONFIG_IGNORE_TEMPLATES_PATH
        dump = build_dump(
            "test", "Test", f"https://{lang}.wikipedia.org/wiki/Main_Page"
        )
        with tempfile.NamedTemporaryFile(
            "w", suffix=".xml", delete=False, encoding="utf-8"
        ) as handle:
            handle.write(dump)
            path = handle.name
        # expand_templates=True is what turns the config paths into sets; with it
        # False the parsing block never runs. Safe here: the template *file* is
        # None, so no dump-wide template collection happens.
        opened, _ = extractor_main.preprocess_dump(path, None, expand_templates=True)
        opened.close()
        Path(path).unlink(missing_ok=True)
        cache[lang] = (
            Extractor.language,
            set(Extractor.discardSections),
            set(Extractor.discardTemplates),
            set(Extractor.ignoreTemplates),
        )

    language, sections, discard, ignore = cache[lang]
    Extractor.language = language
    Extractor.discardSections = sections
    Extractor.discardTemplates = discard
    Extractor.ignoreTemplates = ignore


def extract_once(extractor_main, Extractor, wikitext: str, page: dict) -> str:
    # The MediaWiki page title is the URL slug, so take it from there rather than
    # from the HTML <title>. The stored title is the browser one and carries a
    # localised site suffix -- "France Prešeren - Wikipedija, prosta enciklopedija"
    # -- which would be emitted as this entrant's first heading on every
    # non-English wiki page. Stripping those suffixes by list only ever covers the
    # languages someone remembered; the slug is exact and language-independent.
    # Strip the query before taking the slug: corpus wiki URLs are pinned as
    # `/wiki/Title?oldid=N`, and without this the title becomes
    # "Ada Lovelace?oldid=1370153024" and is emitted as this entrant's first heading.
    slug = page["url"].split("?", 1)[0].split("#", 1)[0].rsplit("/", 1)[-1]
    title = unquote(slug).replace("_", " ") or (page.get("title") or "").strip()

    origin = page["url"].split("/wiki/")[0]
    dump = build_dump(wikitext, title, f"{origin}/wiki/Main_Page")

    # collect_pages() directly rather than preprocess_dump(): the per-language
    # bootstrap already primed namespaces and the config sets, so calling
    # preprocess_dump again would try to os.path.join() a set. Its only other
    # outputs are the opened handle and urlbase, both trivial to supply here --
    # which also keeps redundant header parsing out of the timed section.
    collected = list(extractor_main.collect_pages(io.StringIO(dump)))
    if not collected:
        return ""
    # collect_pages yields 5-tuples (id, revid, title, page, metadata). Upstream's
    # own process_dump_generator() unpacks only 4, so its --generator mode raises;
    # that is why this adapter drives collect_pages/Extractor directly.
    page_id, revid, page_title, lines, metadata = collected[0]
    result = Extractor(
        page_id, revid, f"{origin}/wiki", page_title, lines, metadata=metadata
    ).extract(out=None, html_safe=True)
    if not result:
        return ""
    return result[4] if isinstance(result, tuple) else str(result)


def _page_language(page: dict) -> str:
    """The wiki's own subdomain, not the corpus `lang` label.

    They usually agree, but the URL is what the tool keys its config on.
    """
    host = page["url"].split("//")[-1].split("/")[0]
    subdomain = host.split(".")[0]
    return subdomain if subdomain and subdomain != "www" else (page["lang"] or "en")


def main() -> None:
    extractor_main, Extractor = bootstrap()
    timing = store.config()["timing"]
    config_cache: dict = {}

    pages = [row for row in store.load_pages() if row["is_wiki"]]
    wikitext_snapshots = store.load_snapshots("wikitext")
    print(f"wikiextractor-v2 over {len(pages)} wiki pages", flush=True)

    ok = empty = failed = 0
    for index, page in enumerate(pages, start=1):
        wikitext = store.snapshot_text(page["id"], "wikitext")
        wikitext_fetch_ms = next(
            (
                float(row["fetch_ms"] or 0.0)
                for row in wikitext_snapshots
                if row["page_id"] == page["id"]
            ),
            0.0,
        )
        load_language_config(
            extractor_main, Extractor, _page_language(page), config_cache
        )
        if not wikitext:
            store.record_run(
                page["id"],
                ENTRANT,
                variant="wikitext",
                status="error",
                error="no wikitext snapshot",
            )
            failed += 1
            continue
        try:
            for _ in range(timing["warmup_runs"]):
                extract_once(extractor_main, Extractor, wikitext, page)
            samples: list[float] = []
            text = ""
            for _ in range(timing["timed_runs"]):
                started = time.perf_counter()
                text = extract_once(extractor_main, Extractor, wikitext, page)
                samples.append((time.perf_counter() - started) * 1000)
        except Exception as exc:
            store.record_run(
                page["id"],
                ENTRANT,
                variant="wikitext",
                status="error",
                error=f"{type(exc).__name__}: {exc}"[:300],
            )
            failed += 1
            print(
                f"  [{index}/{len(pages)}] ERROR {page['url'][:60]}: {exc}"[:150],
                flush=True,
            )
            continue

        status = "ok" if text.strip() else "empty"
        ok += status == "ok"
        empty += status == "empty"
        store.record_run(
            page["id"],
            ENTRANT,
            variant="wikitext",
            output=text,
            output_chars=len(text),
            output_sha256=hashlib.sha256(text.encode("utf-8")).hexdigest(),
            extract_ms=statistics.median(samples),
            extract_ms_all=samples,
            # The wikitext API fetch is this entrant's input cost; leaving it at
            # zero would make it look free next to entrants charged for a fetch.
            input_ms=wikitext_fetch_ms,
            status=status,
        )
        print(
            f"  [{index}/{len(pages)}] {status:5s} {len(text):7d} chars "
            f"{statistics.median(samples):8.1f}ms  {page['url'][:52]}",
            flush=True,
        )

    print(f"done: ok={ok} empty={empty} failed={failed}", flush=True)


if __name__ == "__main__":
    main()

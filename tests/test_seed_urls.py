from pathlib import Path

from src.scraping.page_fetch import read_source_urls
import httpx

from src.scraping.seed_urls import (
    SPEC_PATH,
    SiteSpec,
    is_page_url,
    load_clusters,
    matches,
    parse_sitemap,
    same_host,
    sitemap_order,
    sitemap_urls,
    wikipedia_url,
    wikitext_links,
)


def test_wikitext_links_keep_reading_order_and_skip_namespaces():
    wikitext = (
        "'''Veliki Tabor''' je [[dvorac]] u [[Hrvatsko zagorje|Zagorju]] "
        "([[Datoteka:Tabor.jpg|thumb]]) blizu [[Desinić]]a, [[dvorac|dvorci]] "
        "iz [[15. stoljeće|15. stoljeća]]. Vidi [[#Povijest]]."
    )
    assert wikitext_links(wikitext) == [
        "Dvorac",
        "Hrvatsko zagorje",
        "Desinić",
        "15. stoljeće",
    ]


def test_wikipedia_url_encodes_like_the_reference_seed():
    url = wikipedia_url("sl", "Bazilika Lurške Matere Božje, Brestanica")
    assert url == (
        "https://sl.wikipedia.org/wiki/"
        "Bazilika_Lur%C5%A1ke_Matere_Bo%C5%BEje,_Brestanica"
    )
    assert wikipedia_url("de", "Riegersburg (Burg)").endswith("/Riegersburg_(Burg)")


def test_parse_sitemap_tells_indexes_from_url_sets():
    index = "<sitemapindex><sitemap><loc>https://x/page-sitemap.xml</loc></sitemap></sitemapindex>"
    urlset = "<urlset><url><loc> https://x/a </loc></url><url><loc>https://x/b</loc></url></urlset>"
    assert parse_sitemap(index) == (["https://x/page-sitemap.xml"], [])
    assert parse_sitemap(urlset) == ([], ["https://x/a", "https://x/b"])


def test_sitemap_order_prefers_pages_over_posts():
    names = [
        "https://x/post-sitemap.xml",
        "https://x/wp-sitemap-posts-page-1.xml",
        "https://x/other.xml",
    ]
    ordered = sorted(names, key=sitemap_order)
    assert ordered[0].endswith("page-1.xml")
    assert ordered[-1].endswith("post-sitemap.xml")


def test_page_url_and_host_filters():
    assert same_host("https://www.desinic.hr/a", "https://desinic.hr/")
    assert not same_host("https://shop.desinic.hr/a", "https://desinic.hr/")
    assert is_page_url("https://x/o-gradu/zgodovina")
    assert not is_page_url("https://x/brochure.pdf")
    assert not is_page_url("https://x/page?lang=en")
    site = SiteSpec("s", "https://x/", 10, include=("^/hu/",), exclude=("/hu/hirek/",))
    assert matches("https://x/hu/var", site)
    assert not matches("https://x/en/castle", site)
    assert not matches("https://x/hu/hirek/1", site)


def test_spec_loads_with_one_language_per_cluster_and_unique_sources():
    clusters = load_clusters(SPEC_PATH)
    assert len(clusters) >= 8
    sources = [site.source for c in clusters for site in c.sites]
    sources += [c.wikipedia.source for c in clusters if c.wikipedia]
    sources += [c.biography.source for c in clusters if c.biography]
    assert len(sources) == len(set(sources))
    for cluster in clusters:
        assert len(cluster.language) == 2
        assert cluster.sites, cluster.name


def test_seed_rows_carry_an_optional_language(tmp_path: Path):
    seed = tmp_path / "seed.json"
    seed.write_text(
        '[{"source": "a", "url": "https://a/", "language": "hr"},'
        ' {"source": "b", "url": "https://b/"}]'
    )
    rows = read_source_urls(seed)
    assert rows[0].language == "hr"
    assert rows[1].language is None


def test_live_formatter_skips_a_retired_form():
    from src.scraping.seed_urls import live_formatter

    formatters = [
        "https://hbl.lzmk.hr/clanak.aspx?id=$1",
        "https://hbl.lzmk.hr/clanak/$1",
    ]
    chosen = live_formatter(formatters, "2896", lambda url: "clanak/" in url)
    assert chosen == "https://hbl.lzmk.hr/clanak/$1"
    assert live_formatter(formatters, "1", lambda url: False) == formatters[0]


def test_sitemap_urls_resolves_a_relative_robots_directive():
    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/robots.txt":
            return httpx.Response(200, text="User-agent: *\nSitemap: /map.xml\n")
        if request.url.path == "/map.xml":
            return httpx.Response(
                200, text="<urlset><url><loc>https://x/a</loc></url></urlset>"
            )
        return httpx.Response(404)

    client = httpx.Client(transport=httpx.MockTransport(handler))
    assert sitemap_urls("https://x/", client, wanted=10) == ["https://x/a"]


def test_unique_pages_treats_trailing_slash_as_the_same_page():
    from src.scraping.seed_urls import unique_pages

    urls = ["https://x/", "https://x", "https://x/a/", "https://x/a", "https://x/b"]
    assert unique_pages(urls) == ["https://x/", "https://x/a/", "https://x/b"]
    assert not is_page_url("https://x/cdn-cgi/l/email-protection")


def test_site_kinds_default_by_position_and_reference_covers_every_kind():
    from src.shared.env import load_yaml

    clusters = load_clusters(SPEC_PATH)
    reference = load_yaml(SPEC_PATH)["reference"]
    for cluster in clusters:
        assert [site.kind for site in cluster.sites][:2] == ["attraction", "town"]
    assert set(reference) == {"attraction", "town", "biography", "wikipedia"}


def test_quality_verdict_flags_thin_and_stub_sources():
    from src.scraping.seed_quality import PageStats, SourceStats, verdict

    def pages(source, prose_lengths):
        return [
            PageStats(f"{source}{i}", source, n + 100, n, "prose", None, "sl", "sl")
            for i, n in enumerate(prose_lengths)
        ]

    reference = SourceStats.build(
        "svn_biography", "biography", pages("ref", [2000, 3000, 4000])
    )
    good = SourceStats.build("new", "biography", pages("new", [1800, 2500, 5000]))
    stubs = SourceStats.build("stub", "biography", pages("stub", [120, 150, 2000, 200]))
    assert verdict(good, reference) == "ok"
    assert verdict(stubs, reference).startswith("thin")
    assert "stubs" in verdict(stubs, reference)


def test_duplicate_ids_take_every_copy_but_only_in_new_sources():
    from src.scraping.seed_quality import PageStats, duplicate_ids, stub_ids

    def page(pid, source, h, prose=1000):
        return PageStats(pid, source, prose + 50, prose, "prose", h, "sl", "sl")

    pages = [
        page("a", "new", "h1"),
        page("b", "new", "h1"),
        page("c", "new", "h2"),
        page("d", "castle_rajhenburg", "h3"),
        page("e", "castle_rajhenburg", "h3"),
        page("f", "new", "h4", prose=100),
    ]
    kinds = {"new": "town"}
    assert duplicate_ids(pages, kinds) == {"a", "b"}
    assert stub_ids(pages, kinds, 300) == {"f"}


def test_reference_urls_are_kept_out_of_new_seeds(tmp_path: Path):
    from src.scraping.seed_urls import SeedRow, reference_urls, without_reference

    ref = tmp_path / "ref.json"
    ref.write_text(
        '[{"source": "wikipedia", "url": "https://sl.wikipedia.org/wiki/Slovenija"}]'
    )
    rows = [
        SeedRow("x_wikipedia", "https://sl.wikipedia.org/wiki/Slovenija", "sl"),
        SeedRow("x_wikipedia", "https://sl.wikipedia.org/wiki/Ptuj", "sl"),
    ]
    assert [r.url for r in without_reference(rows, reference_urls(ref))] == [
        "https://sl.wikipedia.org/wiki/Ptuj"
    ]

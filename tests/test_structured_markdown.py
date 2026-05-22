from src.scraping.structured_markdown import (
    PageKind,
    classify_page,
    extract_listing_markdown,
)


def test_repeated_cards_are_classified_as_listing():
    html = """
    <html><body>
      <main>
        <article><a href="/events/a">Concert</a><p>Sunday, castle hall.</p></article>
        <article><a href="/events/b">Film night</a><p>Tuesday, Romanesque hall.</p></article>
        <article><a href="/events/c">Workshop</a><p>Friday, museum room.</p></article>
        <article><a href="/events/d">Talk</a><p>Saturday, gallery room.</p></article>
        <article><a href="/events/e">Tour</a><p>Monday, castle courtyard.</p></article>
      </main>
    </body></html>
    """

    assert classify_page(html, "https://example.com/events", "") == PageKind.LISTING

    markdown = extract_listing_markdown(html, "https://example.com/events", "Events")

    assert "# Events" in markdown
    assert "- [Concert](https://example.com/events/a)" in markdown
    assert "Sunday, castle hall." in markdown
    assert "- [Workshop](https://example.com/events/c)" in markdown


def test_link_heavy_article_stays_prose():
    html = """
    <html><body>
      <main>
        <nav>
          <a href="https://af.wikipedia.org/wiki/Test">Afrikaans</a>
          <a href="https://de.wikipedia.org/wiki/Test">Deutsch</a>
          <a href="https://fr.wikipedia.org/wiki/Test">Français</a>
          <a href="https://it.wikipedia.org/wiki/Test">Italiano</a>
          <a href="https://es.wikipedia.org/wiki/Test">Español</a>
        </nav>
        <article>
          <h1>Slovenci</h1>
          <p>Slovenci so južnoslovanski narod, ki večinoma živi v Sloveniji.</p>
          <p>Članek vsebuje veliko notranjih povezav, vendar je primarno proza.</p>
        </article>
      </main>
    </body></html>
    """
    markdown = (
        "# Slovenci\n\nSlovenci so južnoslovanski narod, ki večinoma živi v Sloveniji."
    )

    assert (
        classify_page(html, "https://sl.wikipedia.org/wiki/Slovenci", markdown)
        == PageKind.PROSE
    )


def test_empty_page_without_repeated_items_is_empty():
    html = """
    <html><body>
      <main>
        <a href="/one">One</a>
        <a href="/two">Two</a>
      </main>
    </body></html>
    """

    assert classify_page(html, "https://example.com", "") == PageKind.EMPTY

import uuid

from src.scraping.scraper import PageContent, SiteResult
from src.scraping.transform import transform_results


class StubSummary:
    def __init__(self, summary: str):
        self.summary = summary


class StubSummarizer:
    def invoke(self, prompt: str) -> StubSummary:
        return StubSummary("Short tourism summary")


def test_transform_results_page_mode_uses_deterministic_page_ids():
    results = [
        SiteResult(
            site_url="https://example.com",
            pages=[
                PageContent(
                    url="https://example.com/forest",
                    title="Forest Walk",
                    meta_description="Quiet woodland path",
                    paragraphs=["A long peaceful walk beside a narrow stream."],
                    internal_links=[],
                )
            ],
        )
    ]

    places = transform_results(results, place_per="page", summarizer=StubSummarizer())

    assert len(places) == 1
    assert places[0].id == str(
        uuid.uuid5(uuid.NAMESPACE_URL, "https://example.com/forest")
    )
    assert places[0].place_description
    assert places[0].summary == "Short tourism summary"


def test_transform_results_site_mode_aggregates_pages():
    results = [
        SiteResult(
            site_url="https://example.com",
            pages=[
                PageContent(
                    url="https://example.com/a",
                    title="Lake",
                    meta_description="",
                    paragraphs=["A calm lakeside stop for an easy afternoon walk."],
                    internal_links=[],
                ),
                PageContent(
                    url="https://example.com/b",
                    title="Viewpoint",
                    meta_description="",
                    paragraphs=["A short climb leads to wide valley views."],
                    internal_links=[],
                ),
            ],
        )
    ]

    places = transform_results(results, place_per="site", summarizer=StubSummarizer())

    assert len(places) == 1
    assert places[0].id == str(uuid.uuid5(uuid.NAMESPACE_URL, "https://example.com"))
    assert "Lake" in places[0].place_description
    assert "Viewpoint" in places[0].place_description
